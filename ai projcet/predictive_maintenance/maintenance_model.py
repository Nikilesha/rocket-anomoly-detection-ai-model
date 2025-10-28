import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

from .feature_extraction import extract_time_series_features, normalize_features
from sample_tel import sample_Tel  # ✅ real telemetry source

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class PredictiveMaintenanceModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
        )

    def prepare_data(self, telemetry_df, label_column="failure_flag"):
        if label_column not in telemetry_df.columns:
            raise KeyError(
                f"[❌] '{label_column}' missing! Columns: {telemetry_df.columns.tolist()}"
            )

        X = telemetry_df.drop(columns=[label_column])
        y = telemetry_df[label_column]

        if y.value_counts().min() < 2:
            print("[⚠️] One class too small (<2 samples). Using unstratified split.")
            return train_test_split(X, y, test_size=0.3, random_state=42)
        else:
            return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    def train(self, X_train, y_train):
        print("[🧠] Training model...")
        self.model.fit(X_train, y_train)
        print("[✅] Training complete.")

    def evaluate(self, X_test, y_test):
        print("[📊] Evaluating model...")
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else None

        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
        print("\nClassification Report:\n", classification_report(y_test, preds))

        if probs is not None and len(set(y_test)) > 1:
            print("ROC-AUC Score:", round(roc_auc_score(y_test, probs), 3))
        else:
            print("ROC-AUC Score: Not applicable (only one class present).")

    def save_model(self, path="predictive_maintenance/model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[💾] Model saved at: {path}")


# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    print("[🚀] Starting Predictive Maintenance Model using REAL telemetry data...")

    # ---- Step 1: Convert sample_Tel to a DataFrame ----
    telemetry = pd.DataFrame([sample_Tel])

    # ---- Step 2: Assign realistic failure flag ----
    telemetry["failure_flag"] = (
        (telemetry["vibration_engine"] > 0.04)
        | (telemetry["thrust_output"] < 920)
        | (telemetry["cpu_temp"] > 85)
        | (telemetry["hull_stress"] > 0.004)
    ).astype(int)

    # ---- Step 3: Feature extraction & normalization ----
    telemetry_features = extract_time_series_features(telemetry)
    if isinstance(telemetry_features, tuple):
        telemetry_features = telemetry_features[0]  # unpack if tuple
    telemetry_features = normalize_features(telemetry_features)
    if isinstance(telemetry_features, tuple):
        telemetry_features = telemetry_features[0]

    # ---- Step 4: Train/test split ----
    model_trainer = PredictiveMaintenanceModel()
    X_train, X_test, y_train, y_test = model_trainer.prepare_data(
        telemetry_features, label_column="failure_flag"
    )

    # ---- Step 5: Train, evaluate, save ----
    model_trainer.train(X_train, y_train)
    model_trainer.evaluate(X_test, y_test)
    model_trainer.save_model()

    print("[🏁] Predictive Maintenance training complete using live telemetry data.")
