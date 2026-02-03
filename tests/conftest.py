from pytest import fixture
from ml_engineer_exam.config import MLConfig

@fixture(scope="session")
def session_fixture() -> dict:

    """
    Session scoped fixture to provide model information for tests
    :return:
    """

    config = MLConfig(model_name="linear")

    model_info = {
        "model_path": config.model_path,
        "input_data": "{\"MedInc\": 1.6812, \"HouseAge\": 25.0, \"AveRooms\": 4.192200557103064, \"AveBedrms\": 1.0222841225626742, \"Population\": 1392.0, \"AveOccup\": 3.877437325905293, \"Latitude\": 36.06, \"Longitude\": -119.01}"
    }

    return model_info
