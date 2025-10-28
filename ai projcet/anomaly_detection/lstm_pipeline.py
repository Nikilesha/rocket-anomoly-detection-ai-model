# anomaly_detection/lstm_pipeline.py
import os
import numpy as np
import pandas as pd
from anomaly_detection.data_preprocessing import TelemetryPreprocessor
from anomaly_detection.lstm_model import LSTMAnomalyDetector, train_lstm_autoencoder, detect_anomalies

# Path to telemetry CSV
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_tel", "telemetry_sample.csv")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Telemetry CSV not found: {CSV_PATH}")

# Load telemetry
df = pd.read_csv(CSV_PATH)
telemetry_columns = [
    "engine_temp", "fuel_flow_rate", "chamber_pressure", "oxidizer_flow_rate",
    "vibration_engine", "thrust_output", "cpu_temp", "power_draw",
    "data_bus_errors", "hull_stress", "hull_vibration",
    "external_temp", "altitude", "acceleration", "burn_duration", "mission_time"
]
data = df[telemetry_columns].values

# Preprocess
preprocessor = TelemetryPreprocessor()
normalized_data = preprocessor.fit_transform(data)
seq_length = 10
sequences = preprocessor.create_sequences(normalized_data, seq_length=seq_length)

# Train LSTM autoencoder
input_dim = sequences.shape[2]
model = train_lstm_autoencoder(sequences, input_dim=input_dim, num_epochs=20, batch_size=32, lr=1e-3)

# Detect anomalies
anomaly_mask, reconstruction_error = detect_anomalies(model, sequences, threshold=0.02)

print(f"[✅] Detected {np.sum(anomaly_mask)} anomalies out of {len(anomaly_mask)} sequences")
