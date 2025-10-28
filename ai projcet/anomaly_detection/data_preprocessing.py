# anomaly_detection/data_preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

class TelemetryPreprocessor:
    """
    Handles normalization and sequence creation for rocket telemetry data.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, data):
        """
        Normalize the data to [0,1].
        Args:
            data: numpy array (num_samples, num_features)
        Returns:
            normalized data
        """
        return self.scaler.fit_transform(data)

    def transform(self, data):
        """Normalize using existing fitted scaler."""
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        """Reverse normalization (optional for debugging)."""
        return self.scaler.inverse_transform(data)

    def create_sequences(self, data, seq_length=10):
        """
        Converts continuous telemetry into overlapping sequences.
        Args:
            data: np.array (num_samples, num_features)
            seq_length: number of timesteps per sequence
        Returns:
            np.array (num_sequences, seq_length, num_features)
        """
        sequences = []
        for i in range(len(data) - seq_length):
            seq = data[i : i + seq_length]
            sequences.append(seq)
        return np.array(sequences)

    def denoise(self, data, alpha=0.1):
        """
        Simple exponential moving average filter to smooth noisy signals.
        Args:
            data: numpy array (num_samples, num_features)
            alpha: smoothing factor [0,1]
        Returns:
            smoothed array
        """
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for t in range(1, len(data)):
            smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
        return smoothed


# ------------------ Example usage ------------------
if __name__ == "__main__":
    # Path to the CSV outside the module
    CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_tel", "telemetry_sample.csv")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Telemetry CSV not found: {CSV_PATH}")

    # Load CSV
    df = pd.read_csv(CSV_PATH)

    # Drop non-numeric columns if any (like timestamp)
    telemetry_columns = [
        "engine_temp", "fuel_flow_rate", "chamber_pressure", "oxidizer_flow_rate",
        "vibration_engine", "thrust_output", "cpu_temp", "power_draw",
        "data_bus_errors", "hull_stress", "hull_vibration",
        "external_temp", "altitude", "acceleration", "burn_duration", "mission_time"
    ]
    data = df[telemetry_columns].values

    pre = TelemetryPreprocessor()
    normalized = pre.fit_transform(data)
    seqs = pre.create_sequences(normalized, seq_length=10)

    print(f"[✅] Loaded telemetry from CSV: {CSV_PATH}")
    print(f"[✅] Created {len(seqs)} sequences of shape {seqs.shape}")
