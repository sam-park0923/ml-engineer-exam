import logging
from argparse import ArgumentParser
from ml_engineer_exam.model import run_model
from ml_engineer_exam.config import MLConfig
from ml_engineer_exam.model.utils import HousingModel
from loguru import logger


def main():

    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        '-mn', '--model_name',
        type=str,
        default='linear',
        help='Type of model to train (linear, ridge, random_forest)'
    )

    args = argument_parser.parse_args()
    model_name = args.model_name

    config = MLConfig(model_name=model_name)

    housing_model = HousingModel(model_type=config.model_name)

    logger.add(config.log_dir / f'{config.model_name}_training.log', rotation="10 MB")

    model, metrics = run_model(
        model=housing_model,
        ml_config=config
    )

    logger.info("Model Training Complete!")
    logger.info(f"RMSE: {metrics['rmse']:.2f}")
    logger.info(f"MAE: {metrics['mae']:.2f}")
    logger.info(f"R² Score: {metrics['r2']:.4f}")

    return model, metrics


if __name__ == '__main__':


    main()