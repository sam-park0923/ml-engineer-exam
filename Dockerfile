# Multi-stage build for production ML inference service
# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY data/models/ data/models/

# Install dependencies into a virtual environment
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install . uvicorn[standard] fastapi httptools

# Stage 2: Production image
FROM python:3.12-slim AS production

WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --no-create-home appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code and models
COPY src/ src/
COPY data/models/ data/models/

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_DIR="/app/data/models" \
    PORT=8080

# Expose port
EXPOSE ${PORT}

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run the application with uvicorn
CMD ["uvicorn", "ml_engineer_exam.api.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
