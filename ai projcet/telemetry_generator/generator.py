# rocket_telemetry_generator/generator.py
import os
import time
import random
from .db_utils import save_telemetry, fetch_all_telemetry
import pandas as pd

# CSV output folder outside the generator module
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_tel")
os.makedirs(SAMPLE_DIR, exist_ok=True)  # Create folder if not exists
CSV_PATH = os.path.join(SAMPLE_DIR, "telemetry_sample.csv")


def generate_telemetry(mission_time, fuel_start=1000):
    """
    Generate telemetry values based on mission_time.
    """
    engine_temp = 3200 + 400 * (mission_time / 150) + random.uniform(-10, 10)
    fuel_level = max(fuel_start - 5 * mission_time + random.uniform(-2, 2), 0)
    chamber_pressure = 9e6 + 1e6 * (mission_time / 150) + random.uniform(-1e5, 1e5)
    oxidizer_flow_rate = 250 + 50 * (mission_time / 150) + random.uniform(-5, 5)
    vibration_engine = round(random.uniform(0.0, 0.05), 3)
    thrust_output = 900 + 100 * (mission_time / 150) + random.uniform(-10, 10)
    cpu_temp = 70 + 10 * (mission_time / 150) + random.uniform(-2, 2)
    power_draw = 400 + 50 * (mission_time / 150) + random.uniform(-5, 5)
    data_bus_errors = random.randint(0, 2)
    hull_stress = 0.001 + 0.003 * (mission_time / 150) + random.uniform(-0.0005, 0.0005)
    hull_vibration = 0.005 + 0.01 * (mission_time / 150) + random.uniform(-0.001, 0.001)
    external_temp = -50 + 0.2 * mission_time + random.uniform(-2, 2)
    altitude = 10000 + 8000 * (mission_time / 150) + random.uniform(-50, 50)
    acceleration = 20 + 10 * (mission_time / 150) + random.uniform(-1, 1)
    burn_duration = 20 + random.uniform(-1, 1)

    telemetry = {
        "engine_temp": round(engine_temp, 1),
        "fuel_flow_rate": round(5 + random.uniform(-0.5, 0.5), 1),
        "chamber_pressure": round(chamber_pressure, 1),
        "oxidizer_flow_rate": round(oxidizer_flow_rate, 1),
        "vibration_engine": vibration_engine,
        "thrust_output": round(thrust_output, 1),
        "cpu_temp": round(cpu_temp, 1),
        "power_draw": round(power_draw, 1),
        "data_bus_errors": data_bus_errors,
        "hull_stress": round(hull_stress, 4),
        "hull_vibration": round(hull_vibration, 4),
        "external_temp": round(external_temp, 1),
        "altitude": round(altitude, 1),
        "acceleration": round(acceleration, 1),
        "burn_duration": round(burn_duration, 1),
        "mission_time": round(mission_time, 1)
    }
    return telemetry


def run_generator(interval_sec=2, total_time=150):
    """
    Generate telemetry, store it in MySQL, and update CSV continuously outside the generator folder.
    """
    print("[🛰️] Rocket Telemetry Generator Started...")
    mission_time = 0
    all_data = []

    try:
        while mission_time <= total_time:
            telemetry = generate_telemetry(mission_time)
            save_telemetry(telemetry)
            all_data.append(telemetry)

            # Update CSV continuously
            df = pd.DataFrame(all_data)
            df.to_csv(CSV_PATH, index=False)

            time.sleep(interval_sec)
            mission_time += interval_sec

        print(f"[✅] All telemetry saved to CSV: {CSV_PATH}")

    except KeyboardInterrupt:
        print("\n[⏹️] Generator stopped manually. CSV saved up to last data point.")


if __name__ == "__main__":
    run_generator(interval_sec=2, total_time=150)
