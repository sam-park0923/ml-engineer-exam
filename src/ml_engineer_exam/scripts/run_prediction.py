from argparse import ArgumentParser
import joblib
import json
import pandas as pd
from ml_engineer_exam.prediction import run_prediction
from ml_engineer_exam.config import MLConfig
from loguru import logger


def main():

    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        '-mn', '--model_name',
        type=str,
        default='linear',
        help='Type of model to predict (linear, ridge, random_forest)'
    )
    argument_parser.add_argument(
        '-id', '--input_data',
        type=str,
        default=None,
        help='The input data file path in string JSON format. e.g. '
             '"{\"MedInc\": 1.6812, \"HouseAge\": 25.0, \"AveRooms\": 4.192200557103064, \"AveBedrms\": 1.0222841225626742, '
             '\"Population\": 1392.0, \"AveOccup\": 3.877437325905293, \"Latitude\": 36.06, \"Longitude\": -119.01}"'

    )

    args = argument_parser.parse_args()
    model_name = args.model_name

    config = MLConfig(model_name=model_name)
    input_info = json.loads(args.input_data)
    data = pd.DataFrame([input_info])

    model = joblib.load(config.model_path)

    scaler = joblib.load(config.model_path.with_name('scaler.joblib'))

    logger.add(config.log_dir / f'{config.model_name}_prediction.log')

    preds = run_prediction(
        model=model,
        data=data,
        scaler=scaler,
    )

    logger.info("Predictions Complete!")

    data['PredictedValue'] = preds[0]

    data.to_json(config.prediction_dir / f'predictions_{config.model_name}.json', indent=4)

    logger.info(data.to_dict(orient='records')[0])


if __name__ == '__main__':


    main()