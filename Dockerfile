FROM apache/airflow:2.10.2-python3.12
# Switch to root to install system dependencies
# libpq-dev is required for building postgres drivers
USER root
RUN apt-get update && \
    apt-get install -y gcc python3-dev libpq-dev && \
    apt-get clean

# Switch back to airflow user
USER airflow

# Copy requirements and install them
COPY requirements-airflow.txt /requirements-airflow.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /requirements-airflow.txt
