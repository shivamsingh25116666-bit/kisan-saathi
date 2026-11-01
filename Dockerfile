# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile: Kisan Saathi - Autonomous Multilingual Agricultural Assistant
# Multi-stage production container with Playwright headless chromium
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

# System dependencies for OpenCV, GeoPandas, and Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libspatialindex-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser binaries (headless chromium only to save image size)
RUN playwright install --with-deps chromium

# Copy application source code
COPY app/ ./app/
COPY data/ ./data/
COPY static/ ./static/
COPY main.py .
COPY pyproject.toml .


# Create non-root system user and runtime directories
RUN useradd -m -u 1000 kisanuser && \
    mkdir -p /tmp/kisan_agent logs/audit && \
    chown -R kisanuser:kisanuser /app /tmp/kisan_agent logs

USER kisanuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    PORT=8000 \
    BROWSER_HEADLESS=true \
    TMP_DIR=/tmp/kisan_agent \
    AUDIT_LOG_DIR=logs/audit

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main.py"]

