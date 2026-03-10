from pytest import fixture
from pathlib import Path


@fixture(scope="session")
def session_fixture() -> dict:

    """
    Session scoped fixture to provide model information for tests.

    Resolves the models directory relative to the project root so tests work
    regardless of the machine or home directory layout.

    :return: dict with model_path and sample input_data JSON string.
    :rtype: dict
    """

    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "data" / "models"

    model_info = {
        "model_path": models_dir / "linear.joblib",
        "input_data": "{\"MedInc\": 1.6812, \"HouseAge\": 25.0, \"AveRooms\": 4.192200557103064, \"AveBedrms\": 1.0222841225626742, \"Population\": 1392.0, \"AveOccup\": 3.877437325905293, \"Latitude\": 36.06, \"Longitude\": -119.01}"
    }

    return model_info
