pipeline {
    agent any

    stages {
        stage('Setup Environment') {
            steps {
                sh '''
                # In a real setup, we would use a Docker agent with Python
installed.
                # For this simple lab, we install Python directly in the
Jenkins container.
                apt-get update && apt-get install -y python3 python3-pip
python3-venv
                '''
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install -r requirements.txt
                '''
            }
        }

        stage('Linting') {
            steps {
                sh '''
                . venv/bin/activate
                # Stop the build if there are linting errors
                ruff check .
                '''
            }
        }

        stage('Test') {
            steps {
                sh '''
                . venv/bin/activate
                pytest tests/
                '''
            }
        }
    }
}
