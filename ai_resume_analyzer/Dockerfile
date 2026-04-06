# ─────────────────────────────────────────────────────────────────────
#  Dockerfile — AI Resume Analyzer (Python 3.11 slim, uvicorn)
#
#  Multi-stage build:
#    Stage 1 (builder): install all Python deps
#    Stage 2 (runtime): slim image, no build tools
#
#  Build:   docker build -t resume-analyzer .
#  Run:     docker run -p 8000:8000 resume-analyzer
# ─────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════
#  Stage 1: Builder
# ══════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ══════════════════════════════════════
#  Stage 2: Runtime
# ══════════════════════════════════════
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY api/       ./api/
COPY inference/ ./inference/
COPY training/  ./training/
COPY utils/     ./utils/
COPY models/    ./models/
COPY data/      ./data/

# Non-root user for container security
RUN useradd -m -u 1000 appuser \
 && chown -R appuser:appuser /app
USER appuser

# Create writable directories for logs and feedback
RUN mkdir -p /app/logs /app/data

EXPOSE 8000

# Health check — Docker will restart the container if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Uvicorn: 2 workers, no reload in production
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
