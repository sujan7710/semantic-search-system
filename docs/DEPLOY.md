# Deploying CacheGuard

CacheGuard ships as one Docker image (see [Dockerfile](../Dockerfile)): the API, the demo page and both models
(MiniLM embeddings and the cross-encoder verifier) are baked in, so a container needs no downloads at startup.

## What the host must provide

| Need | Value | Why |
|---|---|---|
| Memory | **1 GB minimum, 2 GB comfortable** | Measured 370 MiB–1.0 GiB in the container under Docker Desktop, depending on recent load (~660 MB natively on Windows) |
| CPU | 1–2 vCPU | The verifier is CPU-bound (~30–50 ms per check); see the [load test](../benchmarks/results/phase5_load/README.md) |
| Port | `$PORT` (default 8000) | The container listens on whatever `PORT` the platform sets |
| Secret | `GROQ_API_KEY` | Set as a platform secret, never in the image or the repo |
| Disk (optional) | a volume at `/data` | Keeps the cache (`/data/cache.sqlite3`) across restarts; without it the cache starts empty after each deploy |
| Startup | ~15–40 s | Ready in 16 s locally (loading and warming up both models); give the health check a start period |

Recommended environment for a public demo:

```
GROQ_API_KEY=<secret>
CACHEGUARD_ADMIN_TOKEN=<secret>      # protects cache admin endpoints and /admin/whoami
CACHEGUARD_PROXY_HOPS=1              # proxies the platform puts in front (check with /admin/whoami)
CACHEGUARD_DAILY_BACKEND_CALLS=300   # caps Groq spend per day; cached answers keep working after it
CACHEGUARD_RATE_LIMIT_PER_MINUTE=30
```

**Client IPs behind a proxy.** Rate limits are per client IP, and behind a platform proxy the server only sees the
proxy. Each proxy appends the address it received the request from to `X-Forwarded-For`, so with N proxies the real
client is the N-th entry from the right; everything further left is whatever the visitor sent and can be forged.
CacheGuard reads exactly that entry (`CACHEGUARD_PROXY_HOPS`, default 0 = ignore the header). After deploying, call
`/admin/whoami` with `X-Admin-Token` and a made-up `X-Forwarded-For` header: the reported client IP must be your real
address, not the made-up one.

Free tiers that cap memory at 512 MB will not fit this image as it is. Making it fit would mean running both models
through ONNX Runtime without PyTorch, and re-checking the guard's calibration with the converted models.

## Run locally

```bash
docker build -t cacheguard .
```

```bash
docker run --rm -p 8000:8000 --env-file .env -v cacheguard-data:/data cacheguard
```

Open http://127.0.0.1:8000/.

`docker --env-file` does not strip quotes (`GROQ_API_KEY="gsk_..."` arrives with the quote marks). CacheGuard strips
one pair of surrounding quotes from its secrets for this reason, but other variables are taken literally, so leave
quotes out of env files used with Docker.

## Hosting options

Prices and free-tier rules change often. Check each provider's pricing page before choosing. The notes below
reflect what was checked when this was written (November 2026).

| Option | Fits 1 GB? | Notes |
|---|---|---|
| **Google Cloud Run** | Yes (set memory to 2 GiB) | Has a monthly free allowance and scales to zero, so a low-traffic demo may cost little or nothing. Requires a Google Cloud billing account. The first request after idle waits for a cold start (~20–40 s model load) unless min instances is 1, which is billed. |
| **Hugging Face Spaces** (Docker) | Yes (CPU Basic: 2 vCPU, 16 GB) | Hugging Face's docs now say new Docker Spaces need a paid plan. Natural home for an ML demo; sleeps when idle. Needs the Space README header (`sdk: docker`, `app_port: 8000`). |
| **Railway / Fly.io / Render (paid tiers)** | Yes on 1–2 GB plans | Deploy straight from the Dockerfile; usage-based or fixed monthly pricing. Render's free tier is 512 MB, too small. |
| **A small VM** (any cloud, including always-free VM offers) | Yes | Most control and often cheapest; you manage updates, TLS (e.g. Caddy in front) and restarts yourself. |

## Google Cloud Run

### One-time setup (in your own Google account)

1. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) and run `gcloud init` to sign in and
   create or pick a project.
2. Link a billing account to the project in the Cloud Console (Billing). Strongly recommended: create a **budget with
   an email alert** (Billing → Budgets & alerts), e.g. $5/month.
3. Enable the services the deploy uses:

   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
   ```

4. Create the two secrets in Secret Manager (Console → Security → Secret Manager → Create secret):
   - `groq-api-key`: your Groq key, **without quotes**.
   - `cacheguard-admin-token`: a long random string (e.g. output of `python -c "import secrets; print(secrets.token_urlsafe(32))"`). Keep it somewhere safe; it unlocks the admin endpoints.
5. Let the Cloud Run service account read them (replace `PROJECT_NUMBER`, shown on the console's project dashboard):

   ```bash
   gcloud secrets add-iam-policy-binding groq-api-key --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com --role=roles/secretmanager.secretAccessor
   ```

   ```bash
   gcloud secrets add-iam-policy-binding cacheguard-admin-token --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com --role=roles/secretmanager.secretAccessor
   ```

### Deploy

From the repository root (Cloud Build builds the Dockerfile remotely; the first build takes several minutes):

```bash
gcloud run deploy cacheguard --source . --region us-central1 --port 8000 --memory 2Gi --cpu 1 --cpu-boost --concurrency 8 --min-instances 0 --max-instances 2 --timeout 60 --allow-unauthenticated --set-env-vars CACHEGUARD_PROXY_HOPS=1,CACHEGUARD_DAILY_BACKEND_CALLS=300,CACHEGUARD_RATE_LIMIT_PER_MINUTE=30 --set-secrets GROQ_API_KEY=groq-api-key:latest,CACHEGUARD_ADMIN_TOKEN=cacheguard-admin-token:latest
```

Why these settings:

| Setting | Reason |
|---|---|
| `--memory 2Gi --cpu 1` | Room for both models; one CPU is enough for a demo, and more instances (not cores) handle bursts |
| `--concurrency 8` | The verifier is CPU-bound; more parallel requests per instance only queue (see the load test) |
| `--min-instances 0` | No cost while idle; the first visit after idle waits for a cold start (~15–40 s) |
| `--max-instances 2` | Caps cost and Groq usage if the link gets popular |
| `--cpu-boost` | Extra CPU during startup, so the models load faster |
| `CACHEGUARD_DAILY_BACKEND_CALLS=300` | Caps Groq calls per day per instance |

Cost notes: Cloud Run has a monthly free allowance, and an idle service scales to zero. The ~2.8 GB image is stored in
Artifact Registry, which has a small free storage allowance; storage beyond it is billed monthly. Cloud Build minutes
beyond its free allowance are billed too. Check current pricing, and keep the budget alert from step 2.

Cloud Run's filesystem is in memory and per instance, so the cache starts empty on each new instance (after a deploy
or a scale-from-zero). That is fine for a demo; persistent caching would need a mounted volume or an external store.

## After deploying

- `GET /health` shows the backend and the guard settings that loaded.
- `GET /metrics` is Prometheus text: cache outcomes by stage and reason, latency histograms, tokens and dollars saved, daily budget left.
- Logs are one JSON line per question (outcome, stage, similarity, latencies). The question text itself is never logged, only a short hash.
