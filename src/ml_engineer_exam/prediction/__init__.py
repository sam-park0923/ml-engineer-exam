import numpy as np
import pandas as pd
from loguru import logger

def run_prediction(model, data: pd.DataFrame, scaler):
    """
    Run prediction using the trained model and input data.

    :param model:
        The trained model to use for prediction.
    :type data: pd.DataFrame
    :param data:
        The input data for prediction.
    :param scaler:
        The fitted scaler to preprocess the input data.
    :return:

    """

    # Transform using the fitted scaler
    logger.info(f"Scaling input data")
    scaled_data = scaler.transform(data)

    # Make prediction
    logger.info(f"Making predictions")
    predictions = model.predict(scaled_data)

    return predictions