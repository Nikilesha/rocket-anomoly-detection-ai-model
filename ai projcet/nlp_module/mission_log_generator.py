# nlp_module/mission_log_generator.py

import time

def generate_mission_log(telemetry_stream, health_monitor, output_file="mission_logs.txt"):
    """
    Generates mission logs based on telemetry and health monitor.
    Each log entry contains timestamp, health status, failure probability, and summary.
    """
    print(f"[🧾] Generating mission logs in {output_file}...")

    # Open file in UTF-8 to handle emojis
    with open(output_file, "w", encoding="utf-8") as f:
        for telemetry in telemetry_stream:
            health = health_monitor.check_health(telemetry)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Simple log entry
            if health["status"] == "✅ HEALTHY":
                summary = "All systems nominal."
            else:
                anomaly_sensors = [a['sensor'] for a in health['anomalies']]
                summary = "Anomalies detected: " + ", ".join(anomaly_sensors)

            log_entry = f"[{timestamp}] STATUS: {health['status']}, Failure Probability: {health['failure_probability']:.2f} | {summary}"
            f.write(log_entry + "\n")

    print("[✅] Mission logs generation complete.")
