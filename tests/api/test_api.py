"""Tests for the FastAPI inference API."""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from ml_engineer_exam.api.app import create_app
from ml_engineer_exam.api.model_registry import ModelRegistry


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def models_dir() -> Path:
    """Resolve the models directory for testing."""
    project_root = Path(__file__).resolve().parents[2]
    models_path = project_root / "data" / "models"
    if not models_path.exists():
        pytest.skip(f"Models directory not found at {models_path}")
    return models_path


@pytest.fixture(scope="module")
def registry(models_dir: Path) -> ModelRegistry:
    """Create a model registry loaded from disk."""
    return ModelRegistry(models_dir)


@pytest.fixture(scope="module")
def client(registry: ModelRegistry) -> TestClient:
    """Create a FastAPI test client with models pre-loaded."""
    app = create_app()
    # Manually wire up registry and routes without lifespan
    from ml_engineer_exam.api.routes import create_router

    app.state.registry = registry
    router = create_router(registry)
    app.include_router(router)
    return TestClient(app)


@pytest.fixture()
def sample_features() -> dict:
    """Sample California housing features for testing."""
    return {
        "MedInc": 1.6812,
        "HouseAge": 25.0,
        "AveRooms": 4.192200557103064,
        "AveBedrms": 1.0222841225626742,
        "Population": 1392.0,
        "AveOccup": 3.877437325905293,
        "Latitude": 36.06,
        "Longitude": -119.01,
    }


# ── Root endpoint ─────────────────────────────────────────────────────────────


class TestRootEndpoint:
    """Tests for the root / endpoint."""

    def test_root_returns_service_info(self, client: TestClient):
        """Root endpoint should return service metadata."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert data["docs"] == "/docs"


# ── Health endpoint ───────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self, client: TestClient):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert isinstance(data["models_loaded"], list)
        assert len(data["models_loaded"]) > 0

    def test_health_lists_models(self, client: TestClient):
        """Health endpoint should list all loaded models."""
        response = client.get("/health")
        data = response.json()
        for model_name in ["linear", "ridge", "random_forest"]:
            assert model_name in data["models_loaded"]


# ── Models endpoint ──────────────────────────────────────────────────────────


class TestModelsEndpoint:
    """Tests for the /models endpoint."""

    def test_list_models(self, client: TestClient):
        """Models endpoint should list available models with metadata."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        for model_info in data:
            assert "name" in model_info
            assert "model_type" in model_info
            assert "file_path" in model_info


# ── Predict endpoint ─────────────────────────────────────────────────────────


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_linear(self, client: TestClient, sample_features: dict):
        """Single prediction with linear model should succeed."""
        response = client.post(
            "/predict",
            json={"model_name": "linear", "features": sample_features},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "linear"
        assert isinstance(data["prediction"], float)
        assert "input_features" in data

    def test_predict_ridge(self, client: TestClient, sample_features: dict):
        """Single prediction with ridge model should succeed."""
        response = client.post(
            "/predict",
            json={"model_name": "ridge", "features": sample_features},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "ridge"
        assert isinstance(data["prediction"], float)

    def test_predict_random_forest(self, client: TestClient, sample_features: dict):
        """Single prediction with random_forest model should succeed."""
        response = client.post(
            "/predict",
            json={"model_name": "random_forest", "features": sample_features},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "random_forest"
        assert isinstance(data["prediction"], float)

    def test_predict_linear_known_value(self, client: TestClient, sample_features: dict):
        """Linear model prediction should match the known expected value."""
        response = client.post(
            "/predict",
            json={"model_name": "linear", "features": sample_features},
        )
        data = response.json()
        # Known expected value from the existing test suite
        assert abs(data["prediction"] - 0.719122841601914) < 1e-6

    def test_predict_invalid_model(self, client: TestClient, sample_features: dict):
        """Requesting a non-existent model should return 404."""
        response = client.post(
            "/predict",
            json={"model_name": "nonexistent_model", "features": sample_features},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_predict_missing_features(self, client: TestClient):
        """Missing required features should return 422."""
        response = client.post(
            "/predict",
            json={"model_name": "linear", "features": {"MedInc": 1.0}},
        )
        assert response.status_code == 422

    def test_predict_default_model(self, client: TestClient, sample_features: dict):
        """Omitting model_name should default to 'linear'."""
        response = client.post(
            "/predict",
            json={"features": sample_features},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "linear"


# ── Batch predict endpoint ───────────────────────────────────────────────────


class TestBatchPredictEndpoint:
    """Tests for the /predict/batch endpoint."""

    def test_batch_predict(self, client: TestClient, sample_features: dict):
        """Batch prediction should return correct number of results."""
        response = client.post(
            "/predict/batch",
            json={
                "model_name": "linear",
                "features": [sample_features, sample_features],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "linear"
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
        assert all(isinstance(p, float) for p in data["predictions"])

    def test_batch_predict_invalid_model(self, client: TestClient, sample_features: dict):
        """Batch prediction with invalid model should return 404."""
        response = client.post(
            "/predict/batch",
            json={
                "model_name": "invalid",
                "features": [sample_features],
            },
        )
        assert response.status_code == 404


# ── Model Registry unit tests ────────────────────────────────────────────────


class TestModelRegistry:
    """Tests for the ModelRegistry class."""

    def test_available_models(self, registry: ModelRegistry):
        """Registry should list all loaded models."""
        available = registry.available_models
        assert "linear" in available
        assert "ridge" in available
        assert "random_forest" in available

    def test_get_model(self, registry: ModelRegistry):
        """Getting a valid model should return an sklearn estimator."""
        model = registry.get_model("linear")
        assert hasattr(model, "predict")

    def test_get_model_not_found(self, registry: ModelRegistry):
        """Getting an invalid model should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            registry.get_model("nonexistent")

    def test_scaler_loaded(self, registry: ModelRegistry):
        """Registry scaler should be loaded and have transform method."""
        scaler = registry.scaler
        assert hasattr(scaler, "transform")

    def test_is_model_available(self, registry: ModelRegistry):
        """is_model_available should return correct boolean."""
        assert registry.is_model_available("linear") is True
        assert registry.is_model_available("nonexistent") is False
