pipeline {
    agent any

    // IMPORTANT:
    // Default checkout triggers the Git plugin before stages run.
    // In your setup (repo mounted at /app), that causes:
    // "fatal: detected dubious ownership in repository at '/app/.git'"
    options {
        skipDefaultCheckout(true)
        timestamps()
    }

    environment {
        APP_DIR = "/app"
        VENV_DIR = "venv"
    }

    stages {

        stage('Setup Environment') {
            steps {
                dir("${APP_DIR}") {
                    sh '''
                    set -eux
                    apt-get update
                    apt-get install -y python3 python3-pip python3-venv docker.io git ca-certificates curl
                    git --version
                    docker --version || true

                    # Mark /app as safe so git commands won't error if they run
                    git config --global --add safe.directory /app
                    git config --global --add safe.directory /app/.git || true
                    '''
                }
            }
        }

        stage('Install Dependencies') {
            steps {
                dir("${APP_DIR}") {
                    sh '''
                    set -eux
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install pandas numpy scikit-learn pytest ruff
                    '''
                }
            }
        }

        stage('Linting') {
            steps {
                dir("${APP_DIR}") {
                    sh '''
                    set -eux
                    . venv/bin/activate
                    ruff check .
                    '''
                }
            }
        }

        stage('Test') {
            steps {
                dir("${APP_DIR}") {
                    sh '''
                    set -eux
                    . venv/bin/activate
                    pytest tests/test_feature_engineering.py tests/test_data.py -v --junitxml=test-results.xml
                    '''
                }
            }
        }

        // -------------------------------
        // LAB 11: BUILD & PUBLISH STAGE
        // -------------------------------
        stage('Build & Publish') {
            steps {
                dir("${APP_DIR}") {
                    script {
                        // Verify docker socket access
                        sh 'docker version'

                        // 1. Build Airflow Prod Image
                        sh 'docker build -t localhost:5001/mlops-airflow:v1 -f Dockerfile.airflow.prod .'

                        // 2. Build API Image
                        sh 'docker build -t localhost:5001/mlops-api:v1 -f Dockerfile.api .'

                        // 3. Push to Local Registry
                        sh 'docker push localhost:5001/mlops-airflow:v1'
                        sh 'docker push localhost:5001/mlops-api:v1'
                    }
                }
            }
        }
    }
}
