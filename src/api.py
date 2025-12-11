import os
import uvicorn
import sqlite3
import json
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import numpy as np

# Import the Inference Engine
try:
    from src.inference import RealTimeInference
except ImportError:
    raise ImportError("Could not import 'src.inference'. Make sure you are running from the project root directory.")

# --- 1. API CONFIGURATION ---
app = FastAPI(
    title="Industrial Digital Twin API",
    description="Real-time Anomaly Detection Microservice for Water Pump Sensors",
    version="1.1.0"
)

# Enable CORS for the HTML Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. GLOBAL STATE ---
inference_engine = None

@app.on_event("startup")
def load_model():
    global inference_engine
    base_dir = os.path.abspath(os.getcwd())
    # Adjust paths if necessary
    model_path = os.path.join(base_dir, "models", "best_model.pth")
    scaler_path = os.path.join(base_dir, "models", "scaler.joblib")
    features_path = os.path.join(base_dir, "models", "feature_columns.joblib")
    db_path = os.path.join(base_dir, "data", "database.db")

    print(f"Loading system from: {base_dir}...")
    try:
        inference_engine = RealTimeInference(
            model_path=model_path,
            scaler_path=scaler_path,
            feature_cols_path=features_path,
            db_path=db_path
        )
        print("✅ Inference Engine initialized successfully.")
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        # We don't raise here to allow API to start, but endpoints will fail if called
        
# --- 3. DATA MODELS ---
class PredictionRequest(BaseModel):
    timestamp: Optional[str] = Field(None, example="2025-12-11 12:00:00")
    machine_status: Optional[str] = Field(None, example="NORMAL")
    sensor_readings: Dict[str, float]

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-12-11 12:00:00",
                "machine_status": "NORMAL",
                "sensor_readings": {"sensor_00": 2.45, "sensor_01": 45.2}
            }
        }

class PredictionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    threshold_used: float
    root_cause_analysis: Dict[str, float]
    status: str

# --- 4. ENDPOINTS ---

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": inference_engine is not None}

@app.get("/simulate/sample")
def get_real_db_sample(status: str = "NORMAL"):
    """
    Fetches a REAL random row from the SQLite database.
    This ensures the data distribution matches the training data exactly.
    """
    if not inference_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    db_path = inference_engine.db_path
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database not found")

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row # Access columns by name
        cursor = conn.cursor()
        
        # Fetch one random row matching the requested status
        # Note: machine_status column in DB is text ('NORMAL', 'BROKEN')
        query = "SELECT * FROM sensor_logs WHERE machine_status = ? ORDER BY RANDOM() LIMIT 1"
        cursor.execute(query, (status,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail=f"No data found for status: {status}")

        # Convert Row to Dict
        row_dict = dict(row)
        
        # Separate metadata from sensor readings
        timestamp = row_dict.pop('timestamp', None)
        machine_status = row_dict.pop('machine_status', None)
        
        # All remaining numeric columns are sensor readings
        # Filter to only include numeric sensor columns (sensor_00, etc.)
        sensor_readings = {k: v for k, v in row_dict.items() if k.startswith('sensor_') and isinstance(v, (int, float))}

        return {
            "timestamp": timestamp,
            "machine_status": machine_status,
            "sensor_readings": sensor_readings
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
def predict_anomaly(payload: PredictionRequest, threshold: float = 0.0065):
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # Flatten input
        flat_data = payload.sensor_readings.copy()
        if payload.timestamp: flat_data['timestamp'] = payload.timestamp
        if payload.machine_status: flat_data['machine_status'] = payload.machine_status

        # Inference
        loss, is_anomaly, sensor_losses = inference_engine.predict(flat_data, threshold=threshold)

        # Root Cause Formatting
        feature_names = inference_engine.features
        if np.ndim(sensor_losses) == 0:
            sensor_losses = np.repeat(sensor_losses, len(feature_names))
        
        feature_errors = dict(zip(feature_names, sensor_losses))
        sorted_errors = dict(sorted(feature_errors.items(), key=lambda item: item[1], reverse=True)[:5])

        return {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(loss),
            "threshold_used": threshold,
            "root_cause_analysis": sorted_errors,
            "status": "CRITICAL" if is_anomaly else "NORMAL"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)