# CacheGuard: Groq gateway + guard + demo page.
#   docker build -t cacheguard .
#   docker run -p 8000:8000 -e GROQ_API_KEY=... cacheguard
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only PyTorch: the default wheel bundles CUDA libraries this image never uses (GBs).
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml README.md ./
RUN mkdir cacheguard && touch cacheguard/__init__.py && pip install . && pip uninstall -y cacheguard

# Bake both models into the image: a cold start then needs no ~420 MB download.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/quora-distilroberta-base')"

COPY cacheguard ./cacheguard
COPY configs ./configs
RUN pip install --no-deps .

RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app /data
USER app

# Models are already in the image; never reach out to the Hub at runtime.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CACHEGUARD_BACKEND=groq \
    CACHEGUARD_CACHE_DB_PATH=/data/cache.sqlite3 \
    PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=4)"

# Hosting platforms set PORT; the shell form expands it.
# No --forwarded-allow-ips='*': trusting X-Forwarded-For from anyone would let a
# visitor pick their own IP and dodge rate limits. CacheGuard reads the header
# itself, counting CACHEGUARD_PROXY_HOPS entries from the right.
CMD ["sh", "-c", "exec uvicorn cacheguard.api:create_default_app --factory --host 0.0.0.0 --port ${PORT}"]
