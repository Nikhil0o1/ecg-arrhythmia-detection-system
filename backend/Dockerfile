# ── Stage 1: build wheels ─────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time system deps only once (layer cache)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="ecg-team" \
      description="ECG Arrhythmia Detection – FastAPI inference service"

# Non-root user for security
RUN groupadd -r ecg && useradd -r -g ecg -m ecg

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only what the runtime needs
COPY api/          api/
COPY inference/    inference/
COPY training/     training/


# Ensure Python can find our packages
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Drop privileges
RUN chown -R ecg:ecg /app
USER ecg

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
