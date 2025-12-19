from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
import pandas as pd
import numpy as np
import mlflow.pyfunc

app = FastAPI(title="NYC Taxi Prediction Service")

# 1. Instrument the app (Expose /metrics)
Instrumentator().instrument(app).expose(app)

# 2. Define Input Schema
class TaxiFeatures(BaseModel):
    passenger_count: float
    trip_distance: float
    fare_amount: float
    PULocationID: int
    DOLocationID: int
    tip_amount: float
    payment_type: int
    trip_type: float

# 3. Load Model (with Dynamic Retry)
model = None

def load_production_model():
    """Attempts to load the production model from MLflow Registry."""
    global model
    try:
        model_name = "NYC_Taxi_Prod"
        model_uri = f"models:/{model_name}/1"
        print(f"Attempting to load model from {model_uri}...")

        # Load model
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Model loaded successfully from MLflow.")
        return True
    except Exception as e:
        print(f"⚠️ Could not load model from MLflow: {e}")
        return False

@app.on_event("startup")
def startup_event():
    global model
    if not load_production_model():
        print("Using Dummy Fallback Model for now.")
        model = "DUMMY"

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model != "DUMMY",
        "model_type": "production" if model != "DUMMY" else "fallback"
    }

@app.post("/predict")
def predict(features: TaxiFeatures):
    global model

    # Lazy Loading: If we are currently using the dummy model, try to reload.
    # This allows the user to train the model AFTER starting the API without restarting the container.
    if model == "DUMMY":
        print("Currently using Fallback. Attempting to reload production model...")
        load_production_model()

    if model == "DUMMY":
        # Still failing to load? Use fallback logic.
        prediction = features.trip_distance * 2.5 + features.fare_amount * 0.1
        return {"prediction": prediction, "model_version": "dummy-fallback"}

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])

        # Filter features
        required_features = [
            'passenger_count', 'trip_distance', 'fare_amount',
            'PULocationID', 'DOLocationID'
        ]
        model_input = input_data[required_features]

        # Predict
        prediction = model.predict(model_input)

        if isinstance(prediction, (np.ndarray, pd.Series)):
            result = prediction[0]
        else:
            result = prediction

        return {"prediction": float(result), "model_version": "production"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
