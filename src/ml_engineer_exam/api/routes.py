"""API route definitions for inference endpoints."""

import pandas as pd
from fastapi import APIRouter, HTTPException

from ml_engineer_exam.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    HousingFeatures,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from ml_engineer_exam.api.model_registry import ModelRegistry
from loguru import logger


def create_router(registry: ModelRegistry) -> APIRouter:
    """Create and return the API router with all inference endpoints.

    :param registry: Model registry containing loaded models and scaler.
    :type registry: ModelRegistry
    :return: Configured FastAPI APIRouter.
    :rtype: APIRouter
    """
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse, tags=["System"])
    def health_check() -> HealthResponse:
        """Health check endpoint. Returns service status and loaded models."""
        return HealthResponse(
            status="healthy",
            models_loaded=registry.available_models,
        )

    @router.get("/models", response_model=list[ModelInfoResponse], tags=["Models"])
    def list_models() -> list[ModelInfoResponse]:
        """List all available models and their types."""
        models_info = []
        for name in registry.available_models:
            model = registry.get_model(name)
            models_info.append(
                ModelInfoResponse(
                    name=name,
                    model_type=type(model).__name__,
                    file_path=str(registry.models_dir / f"{name}.joblib"),
                )
            )
        return models_info

    @router.post(
        "/predict",
        response_model=PredictionResponse,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
        tags=["Prediction"],
    )
    def predict(request: PredictionRequest) -> PredictionResponse:
        """Make a single prediction using the specified model.

        :param request: Prediction request with model name and features.
        :type request: PredictionRequest
        :return: Prediction response with the predicted value.
        :rtype: PredictionResponse
        """
        model_name = request.model_name
        if not registry.is_model_available(model_name):
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. "
                       f"Available models: {registry.available_models}",
            )

        try:
            model = registry.get_model(model_name)
            scaler = registry.scaler

            data = pd.DataFrame([request.features.model_dump()])
            scaled_data = scaler.transform(data)
            prediction = model.predict(scaled_data)

            logger.info(
                f"Prediction made with model '{model_name}': {prediction[0]:.4f}"
            )

            return PredictionResponse(
                model_name=model_name,
                prediction=float(prediction[0]),
                input_features=request.features.model_dump(),
            )
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
        tags=["Prediction"],
    )
    def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Make batch predictions using the specified model.

        :param request: Batch prediction request with model name and list of features.
        :type request: BatchPredictionRequest
        :return: Batch prediction response with list of predicted values.
        :rtype: BatchPredictionResponse
        """
        model_name = request.model_name
        if not registry.is_model_available(model_name):
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. "
                       f"Available models: {registry.available_models}",
            )

        try:
            model = registry.get_model(model_name)
            scaler = registry.scaler

            data = pd.DataFrame([f.model_dump() for f in request.features])
            scaled_data = scaler.transform(data)
            predictions = model.predict(scaled_data)

            logger.info(
                f"Batch prediction made with model '{model_name}': "
                f"{len(predictions)} predictions"
            )

            return BatchPredictionResponse(
                model_name=model_name,
                predictions=[float(p) for p in predictions],
                count=len(predictions),
            )
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
