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
CACHEGUARD_TRUST_PROXY=on            # platforms put a proxy in front; rate limits need the real client IP
CACHEGUARD_ADMIN_TOKEN=<random>      # protects cache admin endpoints
CACHEGUARD_DAILY_BACKEND_CALLS=500   # caps Groq spend per day; cached answers keep working after it
CACHEGUARD_RATE_LIMIT_PER_MINUTE=30
```

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

### Google Cloud Run (example)

With the gcloud CLI installed and a project selected:

```bash
gcloud run deploy cacheguard --source . --region us-central1 --memory 2Gi --cpu 2 --port 8000 --allow-unauthenticated --set-env-vars CACHEGUARD_TRUST_PROXY=on,CACHEGUARD_DAILY_BACKEND_CALLS=500 --set-secrets GROQ_API_KEY=groq-api-key:latest
```

The secret `groq-api-key` must exist in Secret Manager first. Cloud Run's filesystem is in-memory, so the cache
resets on each new instance unless a volume is mounted at `/data`.

## After deploying

- `GET /health` shows the backend and the guard settings that loaded.
- `GET /metrics` is Prometheus text: cache outcomes by stage and reason, latency histograms, tokens and dollars saved, daily budget left.
- Logs are one JSON line per question (outcome, stage, similarity, latencies). The question text itself is never logged, only a short hash.
