# anomaly_detection/detect_anomalies.py
import torch
import numpy as np
import os
import pandas as pd
from anomaly_detection.lstm_model import LSTMAnomalyDetector
from anomaly_detection.transformer_model import TransformerAnomalyDetector
from anomaly_detection.data_preprocessing import TelemetryPreprocessor

# Path to telemetry CSV
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_tel", "telemetry_sample.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Telemetry CSV not found: {CSV_PATH}")

# Load telemetry
df = pd.read_csv(CSV_PATH)

# Select numeric telemetry columns
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
sequences = preprocessor.create_sequences(normalized_data, seq_length=10)

# Convert to tensor
x = torch.tensor(sequences, dtype=torch.float32)

# ---------------- LSTM Model ----------------
lstm_model = LSTMAnomalyDetector(input_dim=len(telemetry_columns))
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# Train simple LSTM autoencoder
print("[🛰️] Training LSTM anomaly detector...")
for epoch in range(10):
    optimizer.zero_grad()
    out = lstm_model(x)
    loss = criterion(out, x)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/10, Loss: {loss.item():.6f}")

# Save LSTM model
torch.save(lstm_model.state_dict(), "lstm_model.pth")
print("✅ LSTM model saved as lstm_model.pth")

# ---------------- Transformer Model ----------------
transformer_model = TransformerAnomalyDetector(input_dim=len(telemetry_columns))
optimizer_t = torch.optim.Adam(transformer_model.parameters(), lr=1e-3)

print("[🛰️] Training Transformer anomaly detector...")
for epoch in range(10):
    optimizer_t.zero_grad()
    out_t = transformer_model(x)
    loss_t = criterion(out_t, x)
    loss_t.backward()
    optimizer_t.step()
    print(f"Epoch {epoch+1}/10, Loss: {loss_t.item():.6f}")

torch.save(transformer_model.state_dict(), "transformer_model.pth")
print("✅ Transformer model saved as transformer_model.pth")
