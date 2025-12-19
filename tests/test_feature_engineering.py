import pandas as pd

def test_high_tip_creation(clean_data, feature_engineer):
    """Test that the 'high_tip' target is created correctly."""
    # clean_data is injected automatically from conftest.py
    # feature_engineer is injected automatically

    # 1. Manually add expected logic outcome to verify against
    # Row 0: tip=1.0, fare=10.0 -> 10% -> pass (if threshold is 10%?)
    # Wait, let's look at logic: tip > 0.1 * fare

    X, y = feature_engineer.create_features(clean_data)

    # Verify outputs are not empty
    assert not X.empty
    assert len(y) == len(X)

    # Verify target calculation (simple check)
    # Re-calculate expectation manually:
    expected_y = (clean_data['tip_amount'] > 0.1 * clean_data['fare_amount']).astype(int)

    # Check strict equality
    pd.testing.assert_series_equal(y, expected_y, check_names=False)

def test_categorical_encoding(clean_data, feature_engineer):
    """Test that creating features does not crash on categories."""
    X, y = feature_engineer.create_features(clean_data)

    # We expect columns to be numerical now if we used OneHot,
    # but our current FeatureEngineer might just pass them through or use DictVectorizer later.
    # Let's check that the columns exist.
    assert 'PULocationID' in X.columns
    assert 'DOLocationID' in X.columns
