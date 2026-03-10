"""Pydantic schemas for API request and response validation."""

from pydantic import BaseModel, Field
from typing import Optional


class HousingFeatures(BaseModel):
    """Input features for California housing price prediction.

    :param MedInc: Median income in block group.
    :param HouseAge: Median house age in block group.
    :param AveRooms: Average number of rooms per household.
    :param AveBedrms: Average number of bedrooms per household.
    :param Population: Block group population.
    :param AveOccup: Average number of household members.
    :param Latitude: Block group latitude.
    :param Longitude: Block group longitude.
    """

    MedInc: float = Field(..., description="Median income in block group", examples=[1.6812])
    HouseAge: float = Field(..., description="Median house age in block group", examples=[25.0])
    AveRooms: float = Field(..., description="Average number of rooms per household", examples=[4.192])
    AveBedrms: float = Field(..., description="Average number of bedrooms per household", examples=[1.022])
    Population: float = Field(..., description="Block group population", examples=[1392.0])
    AveOccup: float = Field(..., description="Average number of household members", examples=[3.877])
    Latitude: float = Field(..., description="Block group latitude", examples=[36.06])
    Longitude: float = Field(..., description="Block group longitude", examples=[-119.01])

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "MedInc": 1.6812,
                "HouseAge": 25.0,
                "AveRooms": 4.192200557103064,
                "AveBedrms": 1.0222841225626742,
                "Population": 1392.0,
                "AveOccup": 3.877437325905293,
                "Latitude": 36.06,
                "Longitude": -119.01,
            }
        ]
    }}


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint.

    :param model_name: Name of the model to use for prediction.
    :param features: Housing features for prediction.
    """

    model_name: str = Field(
        default="linear",
        description="Name of the model to use (linear, ridge, random_forest)",
        examples=["linear"],
    )
    features: HousingFeatures


class BatchPredictionRequest(BaseModel):
    """Request body for batch prediction endpoint.

    :param model_name: Name of the model to use for prediction.
    :param features: List of housing features for prediction.
    """

    model_name: str = Field(
        default="linear",
        description="Name of the model to use (linear, ridge, random_forest)",
        examples=["linear"],
    )
    features: list[HousingFeatures]


class PredictionResponse(BaseModel):
    """Response body for prediction endpoint.

    :param model_name: Name of the model used for prediction.
    :param prediction: Predicted median house value.
    :param input_features: The input features used.
    """

    model_name: str
    prediction: float
    input_features: dict


class BatchPredictionResponse(BaseModel):
    """Response body for batch prediction endpoint.

    :param model_name: Name of the model used for prediction.
    :param predictions: List of predicted median house values.
    :param count: Number of predictions made.
    """

    model_name: str
    predictions: list[float]
    count: int


class HealthResponse(BaseModel):
    """Response body for health check endpoint.

    :param status: Health status of the service.
    :param models_loaded: List of loaded model names.
    """

    status: str
    models_loaded: list[str]


class ModelInfoResponse(BaseModel):
    """Response body for model info endpoint.

    :param name: Model name.
    :param model_type: Type of the sklearn model.
    :param file_path: Path to the model file.
    """

    name: str
    model_type: str
    file_path: str


class ErrorResponse(BaseModel):
    """Standard error response.

    :param detail: Error detail message.
    """

    detail: str
