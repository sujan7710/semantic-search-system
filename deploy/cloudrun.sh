#!/usr/bin/env bash
# Deploy CacheGuard to Google Cloud Run.
# Run from the repository root in Cloud Shell (or anywhere gcloud is signed in):
#   bash deploy/cloudrun.sh
# Needs, in the selected project: billing enabled, and Secret Manager secrets
# "groq-api-key" and "cacheguard-admin-token" (see docs/DEPLOY.md).
set -euo pipefail

REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-cacheguard}"

PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
if [ -z "$PROJECT_ID" ]; then
  echo "No project selected. Run: gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
echo "Project: $PROJECT_ID   Region: $REGION   Service: $SERVICE"

echo "== Enabling the services the deploy uses"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com

echo "== Checking secrets (existence only; values are never printed)"
for s in groq-api-key cacheguard-admin-token; do
  if ! gcloud secrets describe "$s" >/dev/null 2>&1; then
    echo "Missing secret '$s'. Create it in Secret Manager first (docs/DEPLOY.md)." >&2
    exit 1
  fi
  gcloud secrets add-iam-policy-binding "$s" --member="serviceAccount:$SA" \
    --role=roles/secretmanager.secretAccessor --condition=None --quiet >/dev/null
  echo "ok: the service can read $s"
done

echo "== Allowing Cloud Build to build from source"
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$SA" \
  --role=roles/run.builder --condition=None --quiet >/dev/null \
  || echo "warning: could not grant roles/run.builder; the build may fail with a permission error"

echo "== Deploying (the first build takes several minutes)"
gcloud run deploy "$SERVICE" --source . --region "$REGION" --port 8000 \
  --memory 2Gi --cpu 1 --cpu-boost --concurrency 8 \
  --min-instances 0 --max-instances 2 --timeout 60 \
  --allow-unauthenticated \
  --set-env-vars CACHEGUARD_PROXY_HOPS=1,CACHEGUARD_DAILY_BACKEND_CALLS=300,CACHEGUARD_RATE_LIMIT_PER_MINUTE=30 \
  --set-secrets GROQ_API_KEY=groq-api-key:latest,CACHEGUARD_ADMIN_TOKEN=cacheguard-admin-token:latest \
  --quiet

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo "== Deployed: $URL"

echo "== Checking that a forged X-Forwarded-For cannot dodge the rate limit"
TOKEN="$(gcloud secrets versions access latest --secret=cacheguard-admin-token)"
MY_IP="$(curl -fsS https://api.ipify.org)"
WHO="$(curl -fsS "$URL/admin/whoami" -H "X-Admin-Token: $TOKEN" -H "X-Forwarded-For: 203.0.113.7")"
unset TOKEN
SEEN="$(printf '%s' "$WHO" | python3 -c 'import json,sys; print(json.load(sys.stdin)["client_ip_used_for_rate_limits"])')"
if [ "$SEEN" = "$MY_IP" ]; then
  echo "PASS: the server sees this machine's real IP; the forged one is ignored"
elif [ "$SEEN" = "203.0.113.7" ]; then
  echo "FAIL: the forged IP was used. Set CACHEGUARD_PROXY_HOPS lower and redeploy." >&2
else
  CHAIN="$(printf '%s' "$WHO" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["x_forwarded_for"]))')"
  echo "CHECK: server used $SEEN, this machine is $MY_IP, forwarded-for chain has $CHAIN entries."
fi

echo
echo "Done. Open: $URL"
