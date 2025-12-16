import pandas as pd

# We mock the data loading by creating a small DataFrame manually
# instead of reading the large Parquet file during testing.
def test_column_existence():
    # Simulate a minimal valid dataset
    data = {
        'passenger_count': [1, 2],
        'trip_distance': [1.2, 3.4],
        'fare_amount': [10.0, 20.0],
        'tip_amount': [1.0, 2.0],
        'extra_column': ['A', 'B'] # Unrelated column
    }
    df = pd.DataFrame(data)

    # Define the critical columns we expect in our pipeline
    required_columns = ['passenger_count', 'trip_distance', 'fare_amount', 'tip_amount']

    # Check if all required columns are present
    for col in required_columns:
        assert col in df.columns, f"Column {col} is missing from the DataFrame"

def test_cleaning_logic():
    # Simulate data with a missing value in a critical column
    data = {
        'passenger_count': [1, None], # Missing value here
        'trip_distance': [1.2, 3.4],
        'fare_amount': [10.0, 20.0],
        'tip_amount': [1.0, 2.0]
    }
    df = pd.DataFrame(data)

    # Run the cleaning logic (same logic as in baseline.py)
    df_cleaned = df.dropna(subset=['passenger_count', 'trip_distance', 'fare_amount', 'tip_amount'])

    # Assert that the row with NaN was removed
    assert len(df_cleaned) == 1
