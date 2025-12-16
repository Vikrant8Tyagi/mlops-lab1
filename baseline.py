import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# 1. CONFIGURATION
# We hardcode the seed to ensure that every time we run this script,
# the data splits and model initialization happen exactly the same way.
RANDOM_SEED = 42
DATA_PATH = "data/green_tripdata_2023-01.parquet"
def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_parquet(path)
    print(f"Initial data shape: {df.shape}")

    # Basic cleaning: Drop rows with missing values in critical columns
    # Note: We specify subset because some unrelated columns (like ehail_fee)
    # might be entirely null, which would cause dropna() to remove ALL rows.
    df = df.dropna(subset=['passenger_count', 'trip_distance', 'fare_amount', 'tip_amount'])
    print(f"Shape after cleaning: {df.shape}")
    print("Sample data (first 3 rows):")
    print(df.head(3))
    return df

def feature_engineering(df):
    print("Engineering features...")
    # Create Target: High Tip (1) if tip > 15% of fare, else Low Tip (0)
    # We add a small epsilon (0.0001) to avoid division by zero
    df['tip_pct'] = df['tip_amount'] / (df['fare_amount'] + 0.0001)
    df['high_tip'] = (df['tip_pct'] > 0.15).astype(int)
    # Select Features
    # Note: We MUST drop 'tip_amount' and 'total_amount' to prevent Data Leakage.
    # If the model knows the tip amount, it doesn't need to predict it!
    features = ['passenger_count', 'trip_distance', 'fare_amount', 'PULocationID', 'DOLocationID']
    target = 'high_tip'

    return df[features], df[target]

def train_model(X_train, y_train):
    print("Training Random Forest...")
    # Initialize model with the fixed RANDOM_SEED
    rf = RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test):
    print("Evaluating...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("--- Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    # Execution Pipeline
    df = load_data(DATA_PATH)
    X, y = feature_engineering(df)

    # Split Data (using the fixed seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
