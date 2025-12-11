import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import sqlite3
import time
import os

# --- 1. MODEL DEFINITION (Must match training architecture exactly) ---
class IndustrialAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(IndustrialAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8) # Bottleneck
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- 2. INFERENCE ENGINE (OOP) ---
class RealTimeInference:
    def __init__(self, model_path, scaler_path, feature_cols_path, db_path):
        self.db_path = db_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Artifacts
        try:
            self.scaler = joblib.load(scaler_path)
            self.features = joblib.load(feature_cols_path)
            
            # Initialize Model
            input_dim = len(self.features)
            self.model = IndustrialAutoencoder(input_dim).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval() 
        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ Cannot start Inference Engine. Missing file: {e}. Ensure training is complete.")

    def data_stream_generator(self, query: str):
        """
        Simulates a real-time stream by yielding one row at a time from SQLite
        based on a custom SQL query.
        """
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(query)
        cols = [description[0] for description in cursor.description]
        
        for row in cursor:
            row_dict = dict(zip(cols, row))
            yield row_dict
            
        conn.close()

    def preprocess(self, row_dict):
        """
        Selects relevant features, applies scaling, and handles missing values.
        """
        df = pd.DataFrame([row_dict])
        
        # Filter and align features
        for col in self.features:
            if col not in df.columns:
                df[col] = 0 
                
        df_filtered = df[self.features]

        # FIX: Handle NaNs (Critical for inference stability)
        df_filtered = df_filtered.fillna(0.0)
        
        # Scale
        scaled_data = self.scaler.transform(df_filtered)
        
        return torch.from_numpy(scaled_data).float().to(self.device)

    def predict(self, row_dict, threshold=0.0065):
        """
        Performs inference on a single row, returning total loss and per-feature loss.
        """
        input_tensor = self.preprocess(row_dict)
        
        with torch.no_grad():
            reconstruction = self.model(input_tensor)
            
            # Total MSE Loss (The Anomaly Score)
            loss = torch.mean((input_tensor - reconstruction) ** 2).item()
            
            # Per-Sensor MSE Loss (For Root Cause Analysis)
            # This is the 3rd value required by app.py
            # The unsqueeze is necessary because the input tensor is 1D (batch size 1)
            # Calculate loss for each individual feature, not across the whole tensor
            sensor_losses = torch.mean((input_tensor - reconstruction) ** 2, dim=1).squeeze().cpu().numpy()
            
        is_anomaly = loss > threshold
        return loss, is_anomaly, sensor_losses

# --- 3. EXECUTION SIMULATION ---
if __name__ == "__main__":
    # Paths (Adjust relative to where you run the script)
    # Assuming script is run from project root or src/
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
    SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
    FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_columns.joblib")
    DB_PATH = os.path.join(BASE_DIR, "data", "database.db")

    try:
        engine = RealTimeInference(MODEL_PATH, SCALER_PATH, FEATURES_PATH, DB_PATH)
        
        print("\n🚀 Starting Real-Time Anomaly Detection Stream...")
        print("-" * 80)
        print(f"{'TIMESTAMP':<25} | {'STATUS':<10} | {'LOSS':<10} | {'ANOMALY'}")
        print("-" * 80)

        # 1. Normal Operation Demo
        print("\n>> [Scenario 1] Normal Operation Check...")
        normal_stream = list(engine.data_stream_generator(limit=5, query="SELECT * FROM sensor_logs WHERE machine_status='NORMAL' LIMIT 5"))
        for row in normal_stream:
            loss, is_anomaly = engine.predict(row)
            flag = "🔴 ANOMALY DETECTED" if is_anomaly else "🟢 Normal"
            print(f"{row.get('timestamp'):<25} | {row.get('machine_status'):<10} | {loss:.6f}   | {flag}")
            time.sleep(0.05)

        # 2. Failure Simulation Demo (Fetching broken rows to prove detection)
        print("\n>> [Scenario 2] Pure Failure Check (Expecting 100% Anomalies)...")
        broken_stream = list(engine.data_stream_generator(limit=5, query="SELECT * FROM sensor_logs WHERE machine_status='BROKEN' LIMIT 5"))
        for row in broken_stream:
            loss, is_anomaly = engine.predict(row)
            flag = "🔴 ANOMALY DETECTED" if is_anomaly else "🟢 Normal"
            print(f"{row.get('timestamp'):<25} | {row.get('machine_status'):<10} | {loss:.6f}   | {flag}")
            time.sleep(0.05)

        # 3. Mixed Stream Demo (Injecting anomalies into normal flow)
        print("\n>> [Scenario 3] Mixed Stream (Simulating Transient Failure)...")
        # Fetch segments to stitch together
        normal_part_1 = list(engine.data_stream_generator(limit=5, query="SELECT * FROM sensor_logs WHERE machine_status='NORMAL' LIMIT 5 OFFSET 10"))
        broken_injection = list(engine.data_stream_generator(limit=5, query="SELECT * FROM sensor_logs WHERE machine_status='BROKEN' LIMIT 25"))
        normal_part_2 = list(engine.data_stream_generator(limit=5, query="SELECT * FROM sensor_logs WHERE machine_status='NORMAL' LIMIT 5 OFFSET 20"))
        
        # Combine: Normal -> Broken -> Normal
        mixed_stream = normal_part_1 + broken_injection + normal_part_2

        for row in mixed_stream:
            timestamp = row.get('timestamp', 'N/A')
            machine_status = row.get('machine_status', 'Unknown')
            
            # Run Inference
            loss, is_anomaly, _ = engine.predict(row)
            
            # Formatting Output
            anomaly_flag = "🔴 ANOMALY DETECTED" if is_anomaly else "🟢 Normal"
            
            print(f"{timestamp:<25} | {machine_status:<10} | {loss:.6f}   | {anomaly_flag}")
            
            # Simulate Latency (IoT Sensor Interval)
            time.sleep(0.1) 
        print("-" * 80)

    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
        print("Tip: Ensure you have run 'src/etl.py' and 'notebooks/train_model.py' first.")