import pandas as pd
import joblib
import time
from .feature_extraction import extract_time_series_features, normalize_features
import os

# ---- Load trained model ----
model_path = "predictive_maintenance/model.pkl"
model = joblib.load(model_path)
print("[🛰️] Loaded predictive maintenance model successfully.")

# ---- Function to perform prediction ----
def predict_health(sample_reading: dict):
    df = pd.DataFrame([sample_reading])

    try:
        features = extract_time_series_features(df)
        if isinstance(features, tuple):
            features = features[0]  # unpack if returned as tuple
        features = normalize_features(features)
        if isinstance(features, tuple):
            features = features[0]
    except Exception as e:
        print(f"[⚠️] Feature extraction failed: {e}")
        features = df.fillna(0)

    try:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
    except Exception as e:
        print(f"[❌] Model prediction failed: {e}")
        return None

    status = "⚠️ AT RISK of failure" if prediction == 1 else "✅ HEALTHY"
    print(f"\n[🧩] Prediction: {status}")
    print(f"[📈] Failure Probability: {probability:.2f}")
    return {"status": status, "probability": float(probability)}

# ---- Live loop using telemetry CSV ----
def live_monitor(interval: float = 2.0, csv_path: str = "sample_tel/telemetry_sample.csv"):
    """
    Continuously monitors health using telemetry data from CSV.
    Always uses the latest row from the CSV.
    """
    print(f"[🚀] Starting live health monitor using CSV ({csv_path}) with interval {interval}s...")

    if not os.path.exists(csv_path):
        print(f"[❌] CSV file not found: {csv_path}")
        return

    while True:
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                print("[⚠️] CSV is empty. Skipping this cycle.")
                time.sleep(interval)
                continue

            latest_snapshot = df.tail(1).to_dict(orient="records")[0]
            result = predict_health(latest_snapshot)
            if result:
                print(f"[🕒] Status: {result['status']} | Probability: {result['probability']:.2f}")

        except Exception as e:
            print(f"[❌] Error in live monitoring loop: {e}")

        time.sleep(interval)


# ---- Run the live loop ----
if __name__ == "__main__":
    live_monitor(interval=3)
