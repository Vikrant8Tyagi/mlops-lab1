import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. CONFIGURATION
RANDOM_SEED = 42
DATA_PATH = "data/green_tripdata_2023-01.parquet"

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_parquet(path)
    df = df.dropna(subset=['passenger_count', 'trip_distance', 'fare_amount', 'tip_amount'])
    return df

def feature_engineering(df):
    print("Engineering features...")
    df['tip_pct'] = df['tip_amount'] / (df['fare_amount'] + 0.0001)
    df['high_tip'] = (df['tip_pct'] > 0.15).astype(int)

    features = ['passenger_count', 'trip_distance', 'fare_amount', 'PULocationID', 'DOLocationID']
    target = 'high_tip'

    return df[features], df[target]

def train_model(X_train, y_train, n_estimators):
    print(f"Training Random Forest with {n_estimators} trees...")
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test):
    print("Evaluating...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return acc, f1

if __name__ == "__main__":
    # 2. PARSE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=10, help="Number of trees in the forest")
    args = parser.parse_args()

    # 3. SET EXPERIMENT
    mlflow.set_experiment("NYC_Taxi_Experiment")

    # 4. START RUN
    with mlflow.start_run():
        # Log Hyperparameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("seed", RANDOM_SEED)

        # Execution Pipeline
        df = load_data(DATA_PATH)
        X, y = feature_engineering(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

        model = train_model(X_train, y_train, args.n_estimators)
        acc, f1 = evaluate_model(model, X_test, y_test)

        # Log Metrics
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log Model
        # Updated to use 'name' instead of 'artifact_path' to resolve deprecation warning
        mlflow.sklearn.log_model(sk_model=model, name="model")

        print("Run complete. Logged to MLflow.")
