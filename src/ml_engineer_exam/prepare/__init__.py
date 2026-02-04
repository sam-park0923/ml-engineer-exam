import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def load_data(data_path: Path = None) -> pd.DataFrame:
    """Load California housing dataset (replacement for Boston)."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df


def split_features_target(df: pd.DataFrame, target_col: str = 'MedHouseVal'):
    """Split dataframe into features and target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    def split_data(self, X, y):
        """Split data into train and test sets."""
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def fit_transform(self, X_train):
        """Fit scaler and transform training data."""
        return self.scaler.fit_transform(X_train)

    def transform(self, X_test):
        """Transform test data using fitted scaler."""
        return self.scaler.transform(X_test)
