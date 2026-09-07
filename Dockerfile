# ============================================================================
# PatentRank — Production Multi-Stage Container Dockerfile
# Optimized for CPU Inference & Fast Startup
# ============================================================================

FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Runtime Stage
# ============================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Install minimal runtime libraries (libpq for Postgres, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged user for security compliance
RUN groupadd -r patentrank && useradd -r -g patentrank -d /app patentrank

# Copy installed python site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY main.py search_service.py tasks.py celery_app.py segment_documents.py ./
COPY models/ models/
COPY data/ data/
COPY results/ results/

# Set ownership
RUN chown -R patentrank:patentrank /app

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    ELASTICSEARCH_URL=http://elasticsearch:9200 \
    POSTGRES_HOST=pgvector \
    POSTGRES_PORT=5432 \
    POSTGRES_DB=patentrank \
    POSTGRES_USER=postgres \
    POSTGRES_PASSWORD=postgres \
    REDIS_URL=redis://redis:6379/0

USER patentrank

EXPOSE 8000

# Container Healthcheck
HEALTHCHECK --interval=20s --timeout=5s --retries=3 --start-period=15s \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch FastAPI ASGI server
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
