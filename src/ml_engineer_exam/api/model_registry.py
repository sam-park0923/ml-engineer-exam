"""Model registry for loading and caching trained models and scalers."""

import joblib
from pathlib import Path
from loguru import logger


class ModelRegistry:
    """Registry that loads and caches trained models and scalers from disk.

    :param models_dir: Directory containing model .joblib files.
    :type models_dir: Path
    """

    VALID_MODEL_NAMES = {"linear", "ridge", "random_forest"}

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._models: dict = {}
        self._scaler = None
        self._load_all()

    def _load_all(self) -> None:
        """Load all available models and the scaler from the models directory."""
        scaler_path = self.models_dir / "scaler.joblib"
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        else:
            logger.warning(f"Scaler not found at {scaler_path}")

        for model_name in self.VALID_MODEL_NAMES:
            model_path = self.models_dir / f"{model_name}.joblib"
            if model_path.exists():
                self._models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded model '{model_name}' from {model_path}")
            else:
                logger.warning(f"Model '{model_name}' not found at {model_path}")

    def get_model(self, model_name: str):
        """Retrieve a loaded model by name.

        :param model_name: Name of the model to retrieve.
        :type model_name: str
        :return: The loaded sklearn model.
        :raises KeyError: If the model is not loaded.
        """
        if model_name not in self._models:
            raise KeyError(
                f"Model '{model_name}' not found. "
                f"Available models: {list(self._models.keys())}"
            )
        return self._models[model_name]

    @property
    def scaler(self):
        """Return the fitted scaler.

        :return: The fitted StandardScaler.
        :raises RuntimeError: If the scaler is not loaded.
        """
        if self._scaler is None:
            raise RuntimeError("Scaler not loaded. Ensure scaler.joblib exists in the models directory.")
        return self._scaler

    @property
    def available_models(self) -> list[str]:
        """Return list of available (loaded) model names.

        :return: List of model name strings.
        :rtype: list[str]
        """
        return list(self._models.keys())

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available.

        :param model_name: Name of the model to check.
        :type model_name: str
        :return: True if model is loaded.
        :rtype: bool
        """
        return model_name in self._models
