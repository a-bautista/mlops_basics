from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI()

# Define a dictionary to hold model paths for dynamic loading
model_paths = {
    "log_smoteenn": "model_log_smoteenn.pkl",
    "log_tomeklinks": "model_log_tomeklinks.pkl",
    "log_randover": "model_log_randover.pkl",
    "log_smote": "model_log_smote.pkl",
    "log_default": "model_log.pkl"
}

# Define input data format for prediction
class DiabetesData(BaseModel):
    features: List[float]
    model_name: Optional[str] = "log_default"  # Default model if not specified

# Prediction endpoint
@app.post("/predict")
def predict(data: DiabetesData):
    # Load the specified model
    model_path = model_paths.get(data.model_name)
    if model_path is None or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    model = joblib.load(model_path)

    # Validate input features length
    expected_features = model.n_features_in_
    if len(data.features) != expected_features:
        raise HTTPException(
            status_code=400,
            detail=f"Input must contain {expected_features} features."
        )

    # Make prediction
    prediction = model.predict([data.features])[0]
    prediction_label = "diabetic" if prediction == 1 else "non-diabetic"

    return {"prediction": int(prediction), "prediction_label": prediction_label}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction Model API"}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
