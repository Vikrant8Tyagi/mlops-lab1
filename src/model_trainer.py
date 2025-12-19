import os
import tempfile
import shutil
from typing import Tuple

import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.onnx

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from skl2onnx import to_onnx


class ModelTrainer:
    def __init__(self, n_estimators: int, random_state: int):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

    def train(self, X, y, test_size=0.2) -> Tuple[object, object]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        print(f"Training Random Forest with {self.n_estimators} trees...")
        self.model.fit(X_train, y_train)
        return X_test, y_test

    def evaluate(self, X_test, y_test, min_accuracy=0.0, min_f1=0.0):
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Metrics -> Accuracy: {acc:.4f} (Threshold: {min_accuracy}) | "
              f"F1: {f1:.4f} (Threshold: {min_f1})")

        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # --- QUALITY GATE ---
        if not (acc >= min_accuracy and f1 >= min_f1):
            mlflow.set_tag("quality_status", "fail")
            raise ValueError(f"❌ Quality Gate Failed! Accuracy: {acc:.4f}, F1: {f1:.4f}")

        mlflow.set_tag("quality_status", "pass")
        print("✅ Quality Gate Passed.")

        # Use /tmp for artifact staging to avoid permission issues inside Airflow container
        tmp_root = tempfile.mkdtemp(prefix="mlops_lab_")

        try:
            # 1) Log SKLearn model properly (most robust way)
            print("Logging sklearn model to MLflow...")
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                registered_model_name=None  # we will register explicitly below
            )

            # 2) Convert to ONNX
            print("Converting model to ONNX...")
            if hasattr(X_test, "to_numpy"):
                x_sample = X_test.iloc[:1].to_numpy()
            else:
                x_sample = np.array(X_test[:1])

            x_sample = x_sample.astype(np.float32)

            onx = to_onnx(self.model, x_sample)

            # 3) Log ONNX model
            onnx_dir = os.path.join(tmp_root, "onnx_model_dir")
            if os.path.exists(onnx_dir):
                shutil.rmtree(onnx_dir)
            mlflow.onnx.save_model(onx, onnx_dir)
            mlflow.log_artifacts(onnx_dir, artifact_path="onnx_model")
            print("✅ ONNX model logged.")

            # 4) Register model in MLflow Registry
            print("Registering model to MLflow Registry...")
            run_id = mlflow.active_run().info.run_id

            # IMPORTANT: this points to the model logged via mlflow.sklearn.log_model above
            model_uri = f"runs:/{run_id}/model"

            mv = mlflow.register_model(model_uri, "NYC_Taxi_Prod")
            print(f"✅ Registered: Name={mv.name}, Version={mv.version}")

            return acc, f1

        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
