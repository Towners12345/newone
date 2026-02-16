# ── Multi-stage Docker build for production ──
# This is how nPlan would deploy to Azure Container Apps.
# Multi-stage keeps the final image small and secure.

# Stage 1: Install dependencies
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Production image
FROM python:3.12-slim

# Security: run as non-root user
RUN useradd --create-home appuser
WORKDIR /home/appuser

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/

# Switch to non-root user
USER appuser

# Health check — Azure/Kubernetes use this to monitor the service
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Run with uvicorn (production ASGI server)
# --workers 4  = 4 parallel processes (tune for your container size)
# --host 0.0.0.0 = listen on all interfaces (required in containers)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
