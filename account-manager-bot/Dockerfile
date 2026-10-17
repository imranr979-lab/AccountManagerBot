# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ARG BUILD_DATE
ARG GIT_SHA
ARG APP_VERSION="1.0.0"

LABEL org.opencontainers.image.created="${BUILD_DATE}"     \
      org.opencontainers.image.revision="${GIT_SHA}"       \
      org.opencontainers.image.title="account-manager-bot" \
      org.opencontainers.image.version="${APP_VERSION}"

# Security: non-root user
RUN groupadd -g 1001 appgroup && \
    useradd  -u 1001 -g appgroup -s /sbin/nologin -d /app -m appuser

# Minimal runtime packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appgroup . .

# Create writable directories
RUN mkdir -p /app/docs /app/data/vectorstore /app/logs \
    && chown -R appuser:appgroup /app

USER appuser

# Health check via Gradio health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

EXPOSE 7860

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

CMD ["python", "-m", "src.app"]
