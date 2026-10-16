"""
AI-Powered Predictive Maintenance for IoT Devices
Master Pipeline Orchestrator (End-to-End Execution)

Executes all 5 stages in sequence:
1. IoT Telemetry Generation
2. Preprocessing & Feature Engineering
3. Multi-Model Training & Benchmarking
4. Evaluation & Visual Asset Export
5. Virtual Device Stream Simulation
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import save_dataset
from src.preprocess import preprocess_and_split
from src.train import train_and_benchmark
from src.evaluate import evaluate_and_plot
from src.virtual_device_simulator import simulate_device

def run_pipeline():
    print("="*75)
    print("  AI-POWERED PREDICTIVE MAINTENANCE SYSTEM FOR IoT DEVICES")
    print("  End-to-End Pipeline Execution")
    print("="*75)
    
    # Step 1: Telemetry Dataset Generation
    print("\n[PHASE 1/5] Generating Synthetic Industrial IoT Telemetry...")
    save_dataset("data/raw/iot_sensor_data.csv")
    
    # Step 2: Feature Engineering & Preprocessing
    print("\n[PHASE 2/5] Engineering Temporal Rolling Features & Scaling...")
    preprocess_and_split(
        input_csv="data/raw/iot_sensor_data.csv",
        output_dir="data/processed",
        models_dir="models"
    )
    
    # Step 3: Model Training & Benchmarking
    print("\n[PHASE 3/5] Benchmarking Machine Learning Classifiers...")
    train_and_benchmark(data_dir="data/processed", models_dir="models")
    
    # Step 4: Model Evaluation & Visual Artifacts
    print("\n[PHASE 4/5] Evaluating Best Model & Generating Proof Graphs...")
    evaluate_and_plot(
        data_dir="data/processed",
        models_dir="models",
        raw_data_path="data/raw/iot_sensor_data.csv",
        outputs_dir="outputs"
    )
    
    # Step 5: Virtual IoT Simulation Run
    print("\n[PHASE 5/5] Executing Virtual IoT Device Health Diagnostics...")
    simulate_device(mode="offline", iterations=10)
    
    print("\n" + "="*75)
    print("  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("  Artifacts generated:")
    print("   - Raw Dataset: data/raw/iot_sensor_data.csv")
    print("   - Model Checkpoint: models/predictive_maintenance_model.pkl")
    print("   - Feature Scaler: models/scaler.pkl")
    print("   - Visual Proofs: outputs/confusion_matrix.png, feature_importance.png, sensor_telemetry_trace.png")
    print("   - Performance Report: outputs/evaluation_report.json")
    print("="*75)

if __name__ == "__main__":
    run_pipeline()
