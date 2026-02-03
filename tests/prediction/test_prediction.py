import joblib
import json
import pandas as pd
from ml_engineer_exam.prediction import run_prediction

def test_prediction(session_fixture):

    """
    Test prediction function
    :param session_fixture:
        The pytest fixture for the session.
    :return:
        None
    """

    model = joblib.load(session_fixture['model_path'])

    scaler = joblib.load(session_fixture['model_path'].with_name('scaler.joblib'))

    input_info = json.loads(session_fixture['input_data'])

    data = pd.DataFrame([input_info])

    preds = run_prediction(
        model=model,
        data=data,
        scaler=scaler,
    )

    assert preds[0] == 0.719122841601914