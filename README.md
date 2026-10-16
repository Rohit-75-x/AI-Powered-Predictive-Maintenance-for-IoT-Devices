# AI-Powered Predictive Maintenance System for IoT Devices 🏭⚡

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Flask API](https://img.shields.io/badge/Flask-REST%20API-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Production--Ready](https://img.shields.io/badge/Status-Complete%20%26%20Verified-brightgreen.svg)]()

> An end-to-end Machine Learning and IoT analytics pipeline designed to predict industrial equipment breakdowns before they occur. Employs physics-grounded telemetry simulation, rolling window feature engineering, automated model benchmarking, and a real-time Flask inference microservice with automated actuation trip simulation.

---

## 📌 Executive Summary & Industry Relevance

In modern manufacturing (Industry 4.0), unplanned downtime costs industrial facilities upwards of **$50 billion annually**. Traditional maintenance relies either on **reactive repairs** (fixing after catastrophe strikes) or **rigid scheduled maintenance** (prematurely replacing functioning parts).

Predictive Maintenance (PdM) leverages AI and IoT telemetry to monitor equipment health continuously, forecasting failures days or hours in advance:
* **Cost Reduction:** Cuts maintenance expenses by **10% to 40%**.
* **Downtime Elimination:** Reduces unexpected breakdowns by **up to 50%**.
* **Asset Longevity:** Extends machinery operational lifespan by **20% to 40%**.
* **Industry Standards:** Deployed across global leaders including **Siemens, Bosch, GE, ABB, Schneider Electric, and Tesla**.

---

## 🏗️ System Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                     INDUSTRIAL TELEMETRY                                           |
|  [Sensors: Temperature (°C), Vibration (mm/s), Current (A), Pressure (kPa), Motor Speed (RPM)]     |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                  DATA INGESTION & PIPELINE                                         |
|  - Microcontroller Edge Stream (ESP32 / Raspberry Pi) OR Virtual Device Telemetry Simulator         |
|  - Missing Value Imputation & Anomaly Boundary Sanitization                                        |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                FEATURE ENGINEERING ENGINE                                          |
|  - Temporal Rolling Windows (Moving Mean, Moving Std Deviation over 5-step horizons)               |
|  - Physics Strain Factor: (Current * 100) / RPM                                                    |
|  - Dynamic Vibration Gradient: Delta Acceleration (|V_t - V_{t-1}|)                               |
|  - StandardScaler Online Normalization                                                             |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                               MACHINE LEARNING INFERENCE ENGINE                                    |
|  - Benchmark Matrix: Logistic Regression vs Random Forest vs Gradient Boosting                     |
|  - Best Performer: Balanced Random Forest Classifier (Optimized for 100% Failure Recall)           |
|  - Dual Output: Binary Decision + Calibrated Failure Probability (%)                               |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                ALERT, API & ACTUATION TRIP SYSTEM                                  |
|  - Production Flask REST API (/predict)                                                            |
|  - Tri-State Diagnostics: NORMAL (Green) | WARNING (Yellow) | CRITICAL TRIP (Red)                  |
|  - Automated Actuator Trip Simulation (Disables motor driver relay upon critical anomaly)          |
+----------------------------------------------------------------------------------------------------+
```

---

## 📊 Benchmark & Evaluation Results

The models were evaluated against an industrial test split with imbalanced failure distributions (failures representing ~3.4% of total runtime):

| Model | Accuracy | Recall (Failure Detection) | Precision | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest Classifier (Selected)** | **99.65%** | **100.00%** | **90.67%** | **0.9510** | **1.0000** |
| Gradient Boosting Classifier | 99.80% | 98.53% | 95.71% | 0.9710 | 0.9998 |
| Logistic Regression (Baseline) | 99.35% | 100.00% | 83.95% | 0.9128 | 0.9998 |

> **Key Industrial Takeaway:** In industrial plants, **Recall is the single most critical metric**. Missing a catastrophic failure causes catastrophic losses, whereas a slight false positive merely triggers a routine manual check. The selected Random Forest achieved **100% Recall**, successfully catching every simulated failure event.

---

## 📈 Visual Proofs & Output Artifacts

All charts are auto-generated and stored in `outputs/`:

| Confusion Matrix | Feature Importance | Anomaly Telemetry Trace |
|:---:|:---:|:---:|
| ![Confusion Matrix](outputs/confusion_matrix.png) | ![Feature Importance](outputs/feature_importance.png) | ![Sensor Trace](outputs/sensor_telemetry_trace.png) |

---

## 📁 Repository Structure

```
AI-Predictive-Maintenance-IoT/
│
├── data/
│   ├── raw/
│   │   └── iot_sensor_data.csv          # 10,000 synthetic industrial records with physics failures
│   └── processed/
│       ├── train.csv                    # Scaled training split
│       └── test.csv                     # Scaled validation split
│
├── models/
│   ├── predictive_maintenance_model.pkl # Trained Random Forest artifact
│   └── scaler.pkl                       # Fitted StandardScaler
│
├── outputs/
│   ├── confusion_matrix.png             # Visual validation proof
│   ├── feature_importance.png           # Feature contribution rankings
│   ├── sensor_telemetry_trace.png       # Live anomaly spike demonstration
│   └── evaluation_report.json           # Precision, Recall & F1 metrics
│
├── docs/
│   ├── ARCHITECTURE.md                  # Comprehensive architectural specification
│   └── HARDWARE_GUIDE.md                # Raspberry Pi, ESP32 & sensor GPIO hookup guide
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py                # Synthetic industrial telemetry generator
│   ├── preprocess.py                    # Feature engineering & temporal rolling window logic
│   ├── train.py                         # Multi-model benchmarking & artifact exporter
│   ├── evaluate.py                      # Performance metric reporter & graph plotter
│   ├── app.py                           # Flask REST API for real-time edge inference
│   └── virtual_device_simulator.py      # Simulated IoT machine streaming live telemetry
│
├── main.py                              # End-to-end master pipeline orchestrator
├── requirements.txt                     # Pinned project dependencies
├── .gitignore                           # Standard git exclusion rules
└── README.md                            # Comprehensive project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/AI-Predictive-Maintenance-IoT.git
cd AI-Predictive-Maintenance-IoT

# Create virtual environment
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Entire End-to-End Pipeline in One Command

```bash
python main.py
```
This single command automatically:
1. Generates 10,000 synthetic industrial telemetry samples.
2. Performs rolling-window feature engineering and data scaling.
3. Benchmarks multiple ML algorithms and saves the optimal model.
4. Generates visual proof plots (`outputs/*.png`).
5. Executes the Virtual IoT Asset Simulator verifying live failure detection.

---

## 🌐 Real-Time REST API Usage

### 1. Launch the Inference Server
```bash
python src/app.py
```
*API will start listening at `http://127.0.0.1:5000`*

### 2. Query the Prediction Endpoint

#### Nominal / Healthy Asset Request:
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"temperature": 55.2, "vibration": 2.1, "current": 11.4, "pressure": 101.2, "rpm": 1750}'
```
**Response:**
```json
{
  "machine_status": "NORMAL",
  "failure_predicted": false,
  "failure_probability_percent": 0.0,
  "safe_to_operate": true,
  "action_recommended": "OPTIMAL: All sensor telemetry within nominal tolerances."
}
```

#### Critical Failure / Bearing Seizure Request:
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"temperature": 96.5, "vibration": 11.2, "current": 28.4, "pressure": 112.0, "rpm": 1350}'
```
**Response:**
```json
{
  "machine_status": "CRITICAL_ANOMALY",
  "failure_predicted": true,
  "failure_probability_percent": 98.4,
  "safe_to_operate": false,
  "action_recommended": "EMERGENCY: Immediate shutdown advised. High risk of mechanical bearing or thermal seizure."
}
```

---

## 💡 Key Learnings & Interview Talking Points

1. **Why Random Forest over Deep Learning (LSTM) for Tabular Sensor Feeds?**
   While LSTMs excel on raw unengineered sequential signals, in industrial SCADA settings, tabular tree ensembles (Random Forest / XGBoost) trained on rolling temporal statistics yield superior inference speeds (<2ms latency), complete interpretability via feature importance, and zero GPU hardware requirements at the edge.
2. **Handling Extreme Class Imbalance:**
   Machine failures account for less than 5% of factory operations. Using standard accuracy is deceptive. By utilizing stratified sampling, class-weight balancing (`class_weight='balanced'`), and optimizing for **Recall & F1-score**, the system eliminates fatal false negatives.
3. **Hardware-to-Cloud Bridge:**
   The codebase is decoupled: a student can execute 100% of the project virtually on any laptop, or deploy the identical API to an AWS EC2 instance and connect physical hardware (Raspberry Pi + DHT22 + SW-420 + MCP3008 + L298N) using the provided guide in `docs/HARDWARE_GUIDE.md`.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.
