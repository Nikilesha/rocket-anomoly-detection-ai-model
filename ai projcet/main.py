import os
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from rocket_env.main_env import RocketLaunchEnv
import numpy as np

# --- Paths ---
MODEL_PATH = "./rocket_ppo_model.zip"        # Match your saved model
TELEMETRY_PATH = "sample_tel/telemetry_Sample.csv"

# --- Load trained model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Train it first using train_model().")

env = RocketLaunchEnv()
model = PPO.load(MODEL_PATH, env=env)

# --- Run learned model simulation ---
print("\n🧠 Running trained PPO model for evaluation...\n")

obs, _ = env.reset()
learnt_data = []

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    learnt_data.append({
        "step": step,
        "altitude": info["Alt"],
        "velocity": info["Vel"],
        "fuel": info["Fuel"],
        "reward": info["Reward"],
    })

    env.render()

    if terminated or truncated:
        print("\n🎯 Simulation ended.")
        break

learnt_df = pd.DataFrame(learnt_data)

print(
    f"\nFinal Altitude: {info['Alt']:.2f} m | "
    f"Velocity: {info['Vel']:.2f} m/s | "
    f"Remaining Fuel: {info['Fuel']:.2f} kg | "
    f"Total Reward: {sum(learnt_df['reward']):.2f}"
)

# --- Visualization (if telemetry file exists) ---
# --- Visualization (if telemetry file exists) ---
if os.path.exists(TELEMETRY_PATH):
    telemetry_data = pd.read_csv(TELEMETRY_PATH)
    telemetry_data.columns = telemetry_data.columns.str.strip().str.lower()

    # Try to find matching column names (case-insensitive)
    alt_col = next((c for c in telemetry_data.columns if "alt" in c), None)
    vel_col = next((c for c in telemetry_data.columns if "vel" in c), None)
    fuel_col = next((c for c in telemetry_data.columns if "fuel" in c), None)

    if not all([alt_col, vel_col, fuel_col]):
        print("\n⚠️ Telemetry file columns not recognized. Expected columns containing: 'alt', 'vel', 'fuel'.")
        print(f"Found columns: {list(telemetry_data.columns)}")
    else:
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(telemetry_data[alt_col], label="Sample Telemetry", color="orange")
        plt.plot(learnt_df["altitude"], label="Learned Trajectory", color="blue")
        plt.ylabel("Altitude (m)")
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(telemetry_data[vel_col], label="Sample Telemetry", color="orange")
        plt.plot(learnt_df["velocity"], label="Learned Trajectory", color="blue")
        plt.ylabel("Velocity (m/s)")
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(telemetry_data[fuel_col], label="Sample Telemetry", color="orange")
        plt.plot(learnt_df["fuel"], label="Learned Trajectory", color="blue")
        plt.xlabel("Steps")
        plt.ylabel("Fuel (kg)")
        plt.legend()

        plt.tight_layout()
        plt.show()
else:
    print(f"\nTelemetry file not found at {TELEMETRY_PATH}. Skipping comparison plot.")
