import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import torch
import joblib
import numpy as np

# Import the Inference Engine we built earlier
# NOTE: Ensure you run src/etl.py and notebooks/train_model.py first!
from src.inference import RealTimeInference

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AWS Scheer: Industrial Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS & STYLING (Dark Theme & Sidebar FIXES) ---
st.markdown("""
<style>
    /* Main Background adjustments */
    .stApp {
        background-color: #0e1117;
        color: #FAFAFA; /* Default text color */
    }
    
    /* Card-like styling for metrics values */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #ffffff; /* Value is white */
    }
    
    /* Metric Label Color (Ensures labels are readable on dark background) */
    div[data-testid="stMetricLabel"] > label {
        color: #B0B0B0; /* Light gray for labels */
    }

    /* Custom Header Style */
    .header-style {
        font-size: 20px;
        font-weight: bold;
        color: #00BFFF; /* Bright blue for section headers */
        margin-bottom: 10px;
        border-bottom: 1px solid #333;
        padding-bottom: 5px;
    }
    
    /* Status Badge Style */
    .status-badge {
        padding: 8px 15px;
        border-radius: 6px;
        font-weight: bold;
        color: white;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.4); /* Added subtle shadow */
    }
    
    /* Ensure main titles/markdown text is white */
    h1, h2, h3, p {
        color: #FAFAFA !important;
    }

    /* Style Streamlit primary button */
    .stButton>button {
        background-color: #00BFFF;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }

    /* --- SIDEBAR FIXES --- */
    /* Force sidebar background to be dark but distinct */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E;
    }
    /* Force all text in sidebar to be white */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR CONFIGURATION (Business Logic) ---
# Placeholder image for the AWS Scheer context
st.sidebar.image("https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?w=600&auto=format&fit=crop&q=60", caption="AWS Scheer | Digital Twin Lab", use_container_width=True)

st.sidebar.header("🔧 Simulation Controls")
simulation_speed = st.sidebar.slider("Refresh Rate (sec)", 0.05, 1.0, 0.1)
threshold = st.sidebar.slider("Anomaly Threshold (MSE)", 0.001, 0.02, 0.0065, format="%.4f")

st.sidebar.divider()

st.sidebar.header("💰 Business Impact")
downtime_cost_per_hr = st.sidebar.number_input("Cost of Downtime (€/hr)", value=10000, step=1000)
# repair_time_savings = st.sidebar.number_input("Hours Saved (Prediction)", value=4, step=1)
potential_savings = downtime_cost_per_hr# * repair_time_savings

st.sidebar.success(f"Potential ROI: **€{potential_savings:,.0f}** / incident")

st.sidebar.divider()
st.sidebar.markdown("### System Health")
st.sidebar.info("Model: **Autoencoder (PyTorch)**\n\nVersion: **v1.2.0**\n\nStatus: **Online**")

# --- 4. INITIALIZE ENGINE (Cached) ---
@st.cache_resource
def get_inference_engine():
    # Define paths relative to app.py
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
    SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
    FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_columns.joblib")
    DB_PATH = os.path.join(BASE_DIR, "data", "database.db")
    
    return RealTimeInference(MODEL_PATH, SCALER_PATH, FEATURES_PATH, DB_PATH)

try:
    engine = get_inference_engine()
except Exception as e:
    st.error(f"Failed to load system: {e}")
    st.stop()

# --- 5. MAIN DASHBOARD LAYOUT ---

# Top Header Row
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.title("🏭 Pump Station 4: Digital Twin")
    st.markdown("Real-time predictive maintenance dashboard powered by **PyTorch**.")

# Dashboard Grid

# Section 1: Executive Summary
st.markdown("<div class='header-style'>1. Executive Summary</div>", unsafe_allow_html=True)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st_status_card = st.empty()
with kpi2:
    st_gauge = st.empty() # Gauge Chart
with kpi3:
    st_savings = st.empty()
with kpi4:
    st_confidence = st.empty() # Placeholder for another metric

# Section 2: Trends & Diagnostics
st.markdown("<br>", unsafe_allow_html=True) # Spacer

# Full width layout for the main chart since 3D chart is removed
st.markdown("<div class='header-style'>2. Historical Health Trend</div>", unsafe_allow_html=True)
st.caption("Tracks the system's reconstruction error over time. Spikes indicate potential failures.")
chart_placeholder = st.empty()

# Section 3: Data Drill Down
st.markdown("<br>", unsafe_allow_html=True) # Spacer
with st.expander("🔍 3. Raw Sensor Data Telemetry (View All 52 Sensors)"):
    st.caption("Live stream log. Top row (Blue) is the latest reading. Red rows indicate detected anomalies.")
    st_dataframe = st.empty()

# --- 6. SIMULATION LOGIC ---
start_btn = st.button("▶️ Start Live Simulation (Normal -> Failure -> Recovery)")

if start_btn:
    # Prepare Data Stream (Mixed Scenario: Normal -> Broken -> Normal)
    stream_normal_1 = list(engine.data_stream_generator(query="SELECT * FROM sensor_logs WHERE machine_status='NORMAL' LIMIT 15 OFFSET 50"))
    stream_broken = list(engine.data_stream_generator(query="SELECT * FROM sensor_logs WHERE machine_status='BROKEN' LIMIT 15"))
    stream_normal_2 = list(engine.data_stream_generator(query="SELECT * FROM sensor_logs WHERE machine_status='NORMAL' LIMIT 15 OFFSET 100"))
    
    full_stream = stream_normal_1 + stream_broken + stream_normal_2
    
    # History buffers
    history_loss = []
    history_time = []
    accumulated_savings = 0
    
    # NEW: Data Table Log
    sensor_history_log = []
    
    for i, row in enumerate(full_stream):
        # A. PREDICTION & BREAKDOWN
        try:
            # We expect 3 values: loss, is_anomaly, sensor_losses
            prediction_result = engine.predict(row, threshold=threshold)
            
            if len(prediction_result) == 3:
                loss, is_anomaly, sensor_losses = prediction_result
            else:
                # Fallback for compatibility if inference.py wasn't updated
                loss, is_anomaly = prediction_result
                sensor_losses = np.zeros(len(engine.features)) 
                if i == 0: st.warning("Warning: src/inference.py is returning 2 values. Please update it to support Root Cause Analysis.")

        except ValueError:
             st.error("Error: Mismatch in prediction return values. Please update src/inference.py")
             st.stop()
        
        # B. UPDATE KPIs
        if is_anomaly:
            status_html = f"<div class='status-badge' style='background-color: #ff4b4b;'>CRITICAL ALERT</div>"
            accumulated_savings += (downtime_cost_per_hr / 60) # Savings per minute
        else:
            status_html = f"<div class='status-badge' style='background-color: #28a745;'>OPERATIONAL</div>"

        st_status_card.markdown(f"**Current Status**<br>{status_html}", unsafe_allow_html=True)
        st_savings.metric(label="Accumulated Savings", value=f"€{accumulated_savings:,.2f}")
        # Confidence score based on inverse loss
        confidence_score = min(100, (1 - (loss / threshold)) * 100) if loss < threshold else 0
        st_confidence.metric(label="AI Confidence", value=f"{max(0, confidence_score):.1f}%")

        # C. GAUGE CHART (Visual Anomaly Score)
        # This acts as a visual "Speedometer" for health.
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = loss,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Current Anomaly Score"},
            gauge = {
                'axis': {'range': [0, max(0.02, loss*1.2)], 'tickwidth': 1},
                'bar': {'color': "#00BFFF"}, # Bright Blue
                'bgcolor': "#222", # Darker gauge background
                'steps': [
                    {'range': [0, threshold], 'color': "rgba(40, 167, 69, 0.3)"}, # Green
                    {'range': [threshold, max(0.02, loss*1.2)], 'color': "rgba(255, 75, 75, 0.3)"}], # Red
                'threshold': {
                    'line': {'color': "#ff4b4b", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold}
            }
        ))
        fig_gauge.update_layout(height=160, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st_gauge.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{i}")

        # D. MAIN LINE CHART
        history_loss.append(loss)
        history_time.append(i)
        if len(history_loss) > 50:
            history_loss.pop(0)
            history_time.pop(0)

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=history_time, y=history_loss, mode='lines+markers', name='Reconstruction Error',
            line=dict(color='#29b5e8', width=2), fill='tozeroy'
        ))
        fig_line.add_hline(y=threshold, line_dash="dash", line_color="#ff4b4b", annotation_text="Threshold")
        fig_line.update_layout(
            xaxis_title="Time Steps", yaxis_title="MSE Loss", height=350,
            margin=dict(l=20, r=20, t=10, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        chart_placeholder.plotly_chart(fig_line, use_container_width=True, key=f"line_{i}")

        # F. RAW DATA TABLE (Log Style)
        # 1. Prepare row data with status
        row_data = row.copy()
        row_data['Status'] = 'CRITICAL' if is_anomaly else 'OK'
        row_data['Score'] = loss
        
        # 2. Add to history (Insert at top)
        sensor_history_log.insert(0, row_data)
        
        # 3. Limit buffer (e.g., last 50 entries)
        if len(sensor_history_log) > 50:
            sensor_history_log.pop()
            
        # 4. Convert to DataFrame
        df_display = pd.DataFrame(sensor_history_log)
        
        # 5. Reorder columns to show Status/Score first
        if not df_display.empty:
            cols = ['Status', 'Score'] + [c for c in df_display.columns if c not in ['Status', 'Score']]
            df_display = df_display[cols]
        
        # 6. Apply Styling
        def row_styler(row):
            # Index 0 is the latest row (we inserted at 0)
            if row.name == 0:
                # Fixed Blue color for the "Stream Head"
                return ['background-color: #0078D7; color: white; font-weight: bold'] * len(row)
            elif row['Status'] == 'CRITICAL':
                # Red for anomalies stored downward
                return ['background-color: #8B0000; color: white'] * len(row)
            return [''] * len(row)

        st_dataframe.dataframe(
            df_display.style.apply(row_styler, axis=1).format("{:.4f}", subset=['Score']), 
            height=250,
            use_container_width=True
        )
        
        time.sleep(simulation_speed)
    
    st.success("Simulation Cycle Complete.")