import pandas as pd
from typing import List, Tuple


class FeatureEngineer:
    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        target: str,
    ):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.target = target

        # Final feature list
        self.features = self.numerical_features + self.categorical_features

    def create_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Create features and target."""

        print("Engineering features...")

        # Feature engineering
        df = df.copy()
        df["tip_pct"] = df["tip_amount"] / (df["fare_amount"] + 0.0001)
        df[self.target] = (df["tip_pct"] > 0.15).astype(int)

        X = df[self.features]
        y = df[self.target]

        return X, y
