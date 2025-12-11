🏭 **Industrial Digital Twin: Predictive Maintenance Engine**

AWS Scheer Studentwerk Data Science Interview Project

"Bridging the gap between raw sensor data and actionable business insights."

---

## 📖 Project Overview

This project demonstrates an End-to-End Industrial AI Pipeline designed to predict machine failures before they occur. It simulates a real-world Digital Twin for a water pump station using IoT sensor data.

Unlike standard "notebook-only" projects, this solution is architected as a production-grade software package, featuring:

- **SQL Data Engineering**: Simulating an ERP/Historian extraction process.
- **Unsupervised Deep Learning**: Using a PyTorch Autoencoder to detect anomalies without needing labeled failure data ("Golden Batch" training).
- **Real-Time Inference**: A decoupled Inference Engine that simulates live IoT streaming.
- **Interactive Dashboards**: Both a Python-based (Streamlit) Control Tower and a modern Web-based (FastAPI + HTML) dashboard.

---

## 📸 Dashboard Preview

![Dashboard Preview](images/dashboard.png)

---

## 🏗️ System Architecture

The project follows a modular Object-Oriented (OOP) design pattern to ensure scalability and maintainability.

![System Architecture](images/flowchart.png)

---

## 🚀 Key Features

### 1. Data Engineering (ETL) & SQL

- **Clean Architecture**: Raw CSV data is not used directly in training. It is first cleaned, processed, and loaded into a SQLite database to mimic a real industrial Historian.
- **Robust Handling**: Automated removal of "Ghost Sensors" (flatlines) and imputation of missing timestamps.

### 2. The AI Model (PyTorch Autoencoder)

- **Architecture**: A deep Undercomplete Autoencoder with Batch Normalization and Dropout.
- **Strategy**: Trained only on Normal data. The model learns the "physics" of a healthy machine. When a broken machine's data is fed in, the Reconstruction Error (MSE) spikes, flagging an anomaly.
- **Performance**:
  - **R² Score (Normal Reconstruction)**: 0.76 (Strong understanding of system dynamics).
  - **ROC-AUC Score**: 0.99 (Excellent separation of Normal vs. Failure).

### 3. Business Value Dashboard

- **Financial Impact**: Calculates estimated cost savings in real-time (€150/minute of downtime saved).
- **Root Cause Analysis**: Automatically identifies which sensors are contributing most to the anomaly, helping technicians fix the right part.

---

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Deep Learning**: PyTorch
- **Data Manipulation**: Pandas, NumPy, Scikit-Learn
- **Database**: SQLite3
- **Visualization**: Plotly, Streamlit
- **API/Web**: FastAPI, Uvicorn, HTML5, TailwindCSS

---

## ⚙️ Installation & Usage

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run ETL Pipeline (Database Creation)

This script reads `data/raw/sensor.csv`, cleans it, and populates `data/database.db`.

```bash
python src/etl.py
```

### 3. Train the AI Model

Trains the Autoencoder and saves the artifacts (`best_model.pth`, `scaler.joblib`) to the `models/` folder.

```bash
python notebooks/train_model.py
```

### 4. Option A: Run Streamlit Dashboard (Python Native)

```bash
streamlit run app.py
```

### 5. Option B: Run Web Dashboard (FastAPI + HTML)

#### Step 1: Start the Microservice

```bash
uvicorn src.api:app --reload
```

#### Step 2: Open `index.html` in your web browser.

---

## 📂 Project Structure

```plaintext
├── README.md                 # Project documentation
├── app.py                    # Streamlit Dashboard
├── data/                     # Data directory
│   ├── database.db           # SQLite database
│   └── raw/                  # Raw data files
│       └── sensor.csv        # Raw sensor data
├── images/                   # Images for documentation
├── models/                   # Saved models and scalers
│   ├── best_model.pth        # Trained PyTorch model
│   ├── feature_columns.joblib # Feature columns
│   └── scaler.joblib         # Scaler for preprocessing
├── notebooks/                # Jupyter notebooks
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   └── train.ipynb           # Model training
├── requirements.txt          # Python dependencies
├── src/                      # Source code
│   ├── __init__.py           # Package initialization
│   ├── api.py                # FastAPI backend
│   ├── etl.py                # ETL pipeline
│   ├── inference.py          # Inference engine
├── templates/                # HTML templates
│   └── index.html            # Web dashboard template
```

---

## 📊 Results Summary

The model successfully detects the transition from "Normal" to "Broken" hours before catastrophic failure.

| Metric      | Score | Interpretation                                      |
|-------------|-------|----------------------------------------------------|
| **ROC AUC** | 0.99  | Near perfect distinction between healthy and broken states. |
| **Precision** | Low   | Expected due to high class imbalance (safety-first approach). |
| **Recall**   | High  | The system catches the majority of failures (High Safety). |

---

**Author**: Mubeen Afzal

**For**: AWS Scheer (Studentwerk Interview)