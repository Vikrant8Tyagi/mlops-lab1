import argparse
import json
import logging
import os

import mlflow
import yaml

from src.data_loader import DataLoader
from src.data_validation import DataValidator
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
import src.data_contract as dc


# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("great_expectations").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def load_params(params_path: str) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def serialize_gx_results(results) -> dict:
    output = {"success": results.success, "results": []}
    for r in results.results:
        output["results"].append(
            {
                "success": r.success,
                "expectation": r.expectation_config.type,
                "column": r.expectation_config.kwargs.get("column"),
                "unexpected_list": r.result.get("partial_unexpected_list", []),
            }
        )
    return output


def safe_log_artifact(path: str, artifact_path: str | None = None) -> None:
    """Log file to MLflow if it exists (no crash if missing)."""
    if path and os.path.exists(path):
        mlflow.log_artifact(path, artifact_path=artifact_path)


def run_pipeline(params_path: str) -> None:
    params = load_params(params_path)

    # --- Read config ---
    data_path = params["data"]["path"]
    test_size = float(params["data"].get("test_size", 0.2))

    numerical = params["features"]["numerical"]
    categorical = params["features"]["categorical"]
    target = params["features"]["target"]

    n_estimators = int(params["model"]["n_estimators"])
    random_state = int(params["model"]["random_state"])
    min_accuracy = float(params["model"].get("min_accuracy", 0.0))
    min_f1 = float(params["model"].get("min_f1", 0.0))

    exp_name = params["mlflow"]["experiment_name"]

    # --- Components ---
    # Write temp artifacts to a writable place in containers (Airflow)
    artifact_dir = os.environ.get("LOCAL_ARTIFACT_DIR", "/tmp/mlops_artifacts")
    loader = DataLoader(data_path, artifact_dir=artifact_dir)

    engineer = FeatureEngineer(
        numerical_features=numerical,
        categorical_features=categorical,
        target=target,
    )

    trainer = ModelTrainer(
        n_estimators=n_estimators,
        random_state=random_state,
    )

    # --- MLflow ---
    mlflow.set_experiment(exp_name)

    with mlflow.start_run() as run:
        logger.info(f"Starting Run: {run.info.run_id}")

        # Log pipeline configuration
        mlflow.log_param("contract_version", dc.CONTRACT_VERSION)
        mlflow.log_param("data_source", data_path)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("features_numerical", ",".join(numerical))
        mlflow.log_param("features_categorical", ",".join(categorical))
        mlflow.log_param("target", target)

        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "random_state": random_state,
                "min_accuracy": min_accuracy,
                "min_f1": min_f1,
            }
        )

        # 1) Load & Clean
        try:
            df = loader.load_data()

            stats = df.attrs.get("stats", {})
            for k, v in stats.items():
                mlflow.log_metric(f"loader_{k}", float(v) if isinstance(v, (int, float)) else 0.0)

            # Log dropped sample (now from /tmp/..., not project root)
            dropped_sample_path = df.attrs.get("dropped_sample_path")
            safe_log_artifact(dropped_sample_path)

        except Exception as e:
            mlflow.set_tag("status", "load_failed")
            logger.exception(f"Loader failed: {e}")
            raise

        # 2) Validate with Great Expectations
        validator = DataValidator(df)
        try:
            validator.validate()
            mlflow.set_tag("data_quality", "passed")
        except Exception as e:
            mlflow.set_tag("data_quality", "failed")
            logger.exception(f"Validation failed: {e}")

            # Save GX report to writable dir then log
            os.makedirs(artifact_dir, exist_ok=True)
            gx_report_path = os.path.join(artifact_dir, "gx_report.json")
            if validator.validation_results:
                with open(gx_report_path, "w") as f:
                    json.dump(serialize_gx_results(validator.validation_results), f, indent=2)
                safe_log_artifact(gx_report_path)

            raise

        # 3) Feature Engineering
        X, y = engineer.create_features(df)

        # 4) Train + Evaluate (quality gate in ModelTrainer raises on fail)
        X_test, y_test = trainer.train(X, y, test_size=test_size)
        trainer.evaluate(X_test, y_test, min_accuracy=min_accuracy, min_f1=min_f1)

        logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to config file")
    args = parser.parse_args()
    run_pipeline(args.config)
