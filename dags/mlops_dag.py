from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "mlops_engineer",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="nyc_taxi_training",
    default_args=default_args,
    description="Train + evaluate NYC taxi model and log to MLflow",
    start_date=datetime(2025, 1, 1),
    schedule=None,          # manual trigger for classroom
    catchup=False,
    tags=["mlops", "training"],
) as dag:

    # optional: quick sanity check
    show_project = BashOperator(
        task_id="show_project_files",
        bash_command="ls -la /opt/airflow/project && python --version",
    )

    train_and_evaluate = BashOperator(
        task_id="train_and_evaluate",
        bash_command="cd /opt/airflow/project && python main.py",
        env={
            "MLFLOW_TRACKING_URI": "http://mlflow_server:5000",
            "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
            "AWS_ACCESS_KEY_ID": "minioadmin",
            "AWS_SECRET_ACCESS_KEY": "minioadmin",
            "AWS_DEFAULT_REGION": "us-east-1",
        },
    )

    show_project >> train_and_evaluate
