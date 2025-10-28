# predictive_maintenance/health_monitor.py

import pandas as pd
import joblib
import numpy as np
import time
from typing import Union, List, Dict
from .feature_extraction import extract_time_series_features, normalize_features
import os


class HealthMonitor:
    def __init__(
        self,
        model_path: str = "predictive_maintenance/model.pkl",
        probability_threshold: float = 0.5,
        tolerance: float = 0.0,
        severity_display_cutoff: float = 0.01,
    ):
        self.model = joblib.load(model_path)
        self.probability_threshold = float(probability_threshold)
        self.tolerance = float(tolerance)
        self.severity_display_cutoff = float(severity_display_cutoff)
        self.model_features = getattr(self.model, "feature_names_in_", None)
        print("[🛰️] Health Monitor initialized, model loaded.")
        if self.model_features is not None:
            print(f"[ℹ️] Model expects {len(self.model_features)} features.")

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.model_features is None:
            return features
        for col in self.model_features:
            if col not in features.columns:
                features[col] = 0.0
        return features[self.model_features]

    def _safe_severity(self, outside_amount: float, safe_range: float, weight: float) -> float:
        if safe_range == 0:
            return (outside_amount / (abs(outside_amount) + 1.0)) * weight
        return (outside_amount / safe_range) * weight

    def check_health(self, telemetry: Union[Dict, List[Dict], pd.DataFrame]):
        if isinstance(telemetry, dict):
            df = pd.DataFrame([telemetry])
            telemetry_for_rules = telemetry
        elif isinstance(telemetry, list):
            df = pd.DataFrame(telemetry)
            telemetry_for_rules = telemetry[-1] if telemetry else {}
        elif isinstance(telemetry, pd.DataFrame):
            df = telemetry
            telemetry_for_rules = telemetry.iloc[-1].to_dict()
        else:
            raise TypeError("Telemetry must be a dict, list of dicts, or DataFrame")

        # ---- Feature extraction ----
        try:
            features = extract_time_series_features(df)
            if isinstance(features, tuple):
                features = features[0]  # Unpack if tuple
            features = normalize_features(features)
            if isinstance(features, tuple):
                features = features[0]  # Unpack if tuple
        except Exception as e:
            print(f"[⚠️] Feature extraction failed, using raw telemetry: {e}")
            features = df.fillna(0)

        # ---- Feature alignment ----
        try:
            features = self._align_features(features)
        except Exception as e:
            print(f"[⚠️] Feature alignment failed: {e}")

        # Use last row only
        features_snapshot = features.tail(1)

        # ---- ML prediction ----
        try:
            prediction = int(self.model.predict(features_snapshot)[0])
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")

        try:
            probability = float(self.model.predict_proba(features_snapshot)[0][1])
        except Exception:
            probability = float(prediction)

        status = "⚠️ AT RISK" if prediction == 1 else "✅ HEALTHY"
        anomalies = []
        total_severity = 0.0

        # ---- Rule-based checks ----
        rules = {
            "vibration_engine": {"min": 0.0, "max": 0.06, "weight": 2},
            "thrust_output": {"min": 920, "max": 1000, "weight": 3},
            "chamber_pressure": {"min": 0, "max": 1e7, "weight": 3},
            "engine_temp": {"min": 3000, "max": 3500, "weight": 2},
            "cpu_temp": {"min": 0, "max": 85, "weight": 1},
        }

        for key, rule in rules.items():
            if key not in telemetry_for_rules:
                continue
            val = telemetry_for_rules[key]
            min_val, max_val, weight = rule["min"], rule["max"], rule["weight"]
            adj_min = min_val * (1.0 - self.tolerance) if min_val > 0 else min_val
            adj_max = max_val * (1.0 + self.tolerance)
            severity = 0.0
            reason = None

            if val < adj_min:
                outside_amount = float(adj_min - val)
                safe_range = float(adj_max - adj_min) if adj_max > adj_min else 0.0
                severity = self._safe_severity(outside_amount, safe_range, weight)
                reason = "below safe range"
            elif val > adj_max:
                outside_amount = float(val - adj_max)
                safe_range = float(adj_max - adj_min) if adj_max > adj_min else 0.0
                severity = self._safe_severity(outside_amount, safe_range, weight)
                reason = "above safe range"

            if severity and abs(severity) >= self.severity_display_cutoff:
                sev_rounded = round(float(severity), 2)
                anomalies.append(
                    {"sensor": key, "value": float(val), "severity": sev_rounded, "reason": reason}
                )
                total_severity += sev_rounded

        # ---- ML-based severity ----
        if probability >= self.probability_threshold:
            ml_severity = (probability - self.probability_threshold) / (
                1.0 - self.probability_threshold
            ) * 5.0
            ml_severity = round(float(ml_severity), 2)
            if ml_severity >= self.severity_display_cutoff:
                anomalies.append(
                    {
                        "sensor": "ML model",
                        "value": float(probability),
                        "severity": ml_severity,
                        "reason": "high failure probability",
                    }
                )
            total_severity += ml_severity

        return {
            "status": status,
            "failure_probability": float(probability),
            "anomalies": anomalies,
            "total_severity_score": round(float(total_severity), 2),
        }

    def monitor_loop(self, telemetry_stream, interval: float = 1.0):
        for telemetry in telemetry_stream:
            try:
                health = self.check_health(telemetry)
            except Exception as e:
                print(f"[❌] Error while checking health: {e}")
                time.sleep(interval)
                continue

            print(
                f"Health Status: {health['status']}, "
                f"Failure Probability: {health['failure_probability']:.2f}, "
                f"Total Severity: {health['total_severity_score']}"
            )

            if health["anomalies"]:
                for a in health["anomalies"]:
                    print(
                        f"   ⚠️ {a['sensor']}: {a['reason']} "
                        f"(value={a['value']}, severity={a['severity']})"
                    )

            time.sleep(interval)


# ---- Standalone run: read CSV and predict ----
if __name__ == "__main__":
    csv_path = os.path.join("sample_tel", "telemetry_sample.csv")
    if not os.path.exists(csv_path):
        print("[❌] CSV not found. Make sure 'sample_tel/telemetry_sample.csv' exists.")
    else:
        df = pd.read_csv(csv_path)
        print(f"[📊] Loaded {len(df)} telemetry records from CSV.")

        monitor = HealthMonitor(
            probability_threshold=0.5,
            tolerance=0.02,
            severity_display_cutoff=0.05
        )

        result = monitor.check_health(df)
        print("\nHealth Check Result:")
        print(result)
