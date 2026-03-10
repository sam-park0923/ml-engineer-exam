"""FastAPI application factory for the ML inference service."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from loguru import logger

from ml_engineer_exam.api.model_registry import ModelRegistry
from ml_engineer_exam.api.routes import create_router


def _resolve_models_dir() -> Path:
    """Resolve the models directory path.

    Checks the MODELS_DIR environment variable first, then falls back to
    the default path relative to the package (data/models/).

    :return: Resolved path to the models directory.
    :rtype: Path
    """
    env_dir = os.environ.get("MODELS_DIR")
    if env_dir:
        return Path(env_dir)
    # Default: relative to project root (data/models/)
    return Path(__file__).resolve().parents[3] / "data" / "models"


# Module-level registry reference for access by routes
_registry: ModelRegistry | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — loads models on startup.

    :param app: The FastAPI application instance.
    :type app: FastAPI
    """
    global _registry
    models_dir = _resolve_models_dir()
    logger.info(f"Loading models from: {models_dir}")
    _registry = ModelRegistry(models_dir)
    app.state.registry = _registry

    router = create_router(_registry)
    app.include_router(router)

    logger.info(
        f"ML Inference Service started. "
        f"Available models: {_registry.available_models}"
    )
    yield
    logger.info("ML Inference Service shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    :return: Configured FastAPI application.
    :rtype: FastAPI
    """
    app = FastAPI(
        title="Milliman IntelliScript ML Inference Service",
        description=(
            "REST API for California housing price prediction using trained "
            "scikit-learn models (Linear Regression, Ridge, Random Forest)."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.get("/", tags=["System"])
    def root():
        """Root endpoint returning service information."""
        return {
            "service": "Milliman IntelliScript ML Inference Service",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
