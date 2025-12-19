import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer

@pytest.fixture
def raw_data():
    """Provides a raw DataFrame mimicking the source data."""
    data = {
        'passenger_count': [1.0, 2.0, np.nan, 1.0],
        'trip_distance': [1.2, 3.4, 100.0, 0.5], # 100.0 is an outlier
        'fare_amount': [10.0, 20.0, 500.0, -5.0], # -5.0 is invalid
        'PULocationID': [100, 200, 150, 100],
        'DOLocationID': [200, 100, 250, 200],
        'tip_amount': [1.0, 2.0, 0.0, 0.5],
        'payment_type': [1.0, 1.0, 2.0, 1.0],
        'trip_type': [1.0, 1.0, 1.0, 1.0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def clean_data(raw_data):
    """Provides cleaned data by running it through the loader logic."""
    # We can reuse logic from our actual classes if they are decoupled enough.
    # For now, let's just manually clean it to simulate the 'cleaned' state
    # or instantiate the real Loader if possible.

    # Simple manual clean for testing purposes matches logic in DataLoader
    df = raw_data.copy()
    df = df.dropna()
    df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]
    df = df[df['fare_amount'] > 0]
    return df

@pytest.fixture
def feature_engineer():
    """Provides a configured FeatureEngineer instance."""
    return FeatureEngineer(
        numerical_features=['passenger_count', 'trip_distance', 'fare_amount'],
        categorical_features=['PULocationID', 'DOLocationID'],
        target='high_tip'
    )
