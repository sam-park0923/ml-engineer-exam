import joblib
import pandas as pd
from ml_engineer_exam.prepare import load_data, DataPreprocessor, split_features_target
from ml_engineer_exam.config import MLConfig
from ml_engineer_exam.model.utils import HousingModel
from loguru import logger


def run_model(model: HousingModel, ml_config: MLConfig) -> tuple:

    """Train housing price prediction model."""
    # Load data
    df = load_data()
    X, y = split_features_target(df)

    # Preprocess
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    joblib.dump(preprocessor.scaler, ml_config.model_dir / 'scaler.joblib')

    # Save Preprocessor Data
    input_data = [
        ('X_train.csv', X_train),
        ('X_train_scaled.csv', X_train_scaled),
        ('X_test.csv', X_test),
        ('X_test_scaled.csv', X_test_scaled),
        ('y_train.csv', y_train),
        ('y_test.csv', y_test),
    ]
    for name, data in input_data:
        logger.info(f'Saving input data to: {ml_config.input_data_dir / name}')
        cur_data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        cur_data.columns = X.columns.to_list() if name.startswith('X') else ['MedHouseVal']
        cur_data.to_csv(ml_config.input_data_dir / name, index=False, header=True)

    # Train model
    logger.info("Training model...")
    model.train(X_train_scaled, y_train)

    # Evaluate
    logger.info("Getting Model Evaluation metrics...")
    metrics = model.evaluate(X_test_scaled, y_test)

    # Save model
    if ml_config.model_path:
        logger.info(f"Saving model to {ml_config.model_path}")
        model.save(ml_config.model_path)

    return model, metrics