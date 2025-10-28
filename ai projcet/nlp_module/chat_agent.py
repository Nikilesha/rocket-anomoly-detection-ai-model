import os
import numpy as np
import pandas as pd
from predictive_maintenance.health_monitor import HealthMonitor
from .mission_log_generator import generate_mission_log
from .report_generator import generate_summary_report


class RocketChatAgent:
    def __init__(self):
        self.monitor = HealthMonitor()
        print("[🛰️] Health Monitor initialized, model loaded.")
        print("[🤖] Rocket Chat Agent initialized.\n")

        # Base directory (nlp_module)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(self.base_dir, exist_ok=True)

        # Paths
        self.log_file = os.path.join(self.base_dir, "mission_logs.txt")
        self.report_file = os.path.join(self.base_dir, "mission_report.txt")

        # Path to sample telemetry data
        self.sample_csv = os.path.join(os.path.dirname(self.base_dir), "sample_tel", "telemetry_sample.csv")

        # Auto-generate logs and report
        self._auto_generate_files()

    # -------------------------
    # Auto generation logic
    # -------------------------
    def _auto_generate_files(self):
        try:
            print("[⚙️] Attempting to use sample telemetry data...")
            telemetry_stream = self.load_sample_data() if os.path.exists(self.sample_csv) else self.mock_stream()

            generate_mission_log(telemetry_stream, self.monitor, output_file=self.log_file)
            print(f"[✅] Mission logs created: {self.log_file}")

            if os.path.exists(self.log_file):
                generate_summary_report(self.log_file, output_file=self.report_file)
                print(f"[✅] Report generated: {self.report_file}")
        except Exception as e:
            print(f"[❌] Auto-generation failed: {e}")

    # -------------------------
    # Chat Loop
    # -------------------------
    def run(self):
        print("Available commands:\n"
              "  health      - Check health of a single telemetry snapshot\n"
              "  generate    - Generate mission logs from telemetry stream\n"
              "  report      - Generate human-readable report from logs\n"
              "  view_logs   - Show the last 10 lines of mission logs\n"
              "  view_report - Show the generated report\n"
              "  exit        - Exit the chat agent\n")

        while True:
            cmd = input(">> ").strip().lower()
            if cmd == "exit":
                print("Exiting Rocket Chat Agent.")
                break
            elif cmd == "health":
                self._handle_health()
            elif cmd == "generate":
                self._handle_generate()
            elif cmd == "report":
                self._handle_report()
            elif cmd == "view_logs":
                self._view_file(self.log_file)
            elif cmd == "view_report":
                self._view_file(self.report_file)
            else:
                print("[❓] Unknown command.\n")

    # -------------------------
    # Handlers
    # -------------------------
    def _handle_health(self):
        print("Enter telemetry as key=value pairs separated by spaces "
              "(e.g., engine_temp=3400 vibration_engine=0.02)")
        user_input = input("Telemetry >> ")
        telemetry = {}
        try:
            for pair in user_input.split():
                if "=" not in pair:
                    continue
                key, val = pair.split("=", 1)
                key = key.strip()
                val = val.strip()
                try:
                    val = float(val)
                except ValueError:
                    pass
                telemetry[key] = val

            if not telemetry:
                raise ValueError("No valid telemetry entered.")

            health = self.monitor.check_health(telemetry)
            print(f"[🛰️] Health Status: {health['status']}, "
                  f"Failure Probability: {health['failure_probability']:.2f}, "
                  f"Total Severity: {health['total_severity_score']}")
            if health["anomalies"]:
                for a in health["anomalies"]:
                    print(f"   ⚠️ {a['sensor']}: {a['reason']} "
                          f"(value={a['value']}, severity={a['severity']})")
        except Exception as e:
            print(f"[❌] Error parsing telemetry or checking health: {e}")

    def _handle_generate(self):
        try:
            if os.path.exists(self.sample_csv):
                print(f"[📁] Using sample telemetry file: {self.sample_csv}")
                stream = self.load_sample_data()
            else:
                print("[⚙️] No sample data found, using mock telemetry.")
                stream = self.mock_stream()

            generate_mission_log(stream, self.monitor, output_file=self.log_file)
            print(f"[✅] Mission logs generated: {self.log_file}")
        except Exception as e:
            print(f"[❌] Error generating mission logs: {e}")

    def _handle_report(self):
        try:
            if os.path.exists(self.log_file):
                generate_summary_report(self.log_file, output_file=self.report_file)
                print(f"[✅] Report generated: {self.report_file}")
            else:
                print("[❌] Log file missing. Cannot generate report.")
        except Exception as e:
            print(f"[❌] Error generating report: {e}")

    # -------------------------
    # File Viewer
    # -------------------------
    def _view_file(self, filepath):
        if not os.path.exists(filepath):
            print(f"[❌] File '{filepath}' does not exist.")
            return
        print(f"--- Showing last 10 lines of {filepath} ---")
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()[-10:]
            for line in lines:
                print(line.strip())
        print("--- End ---\n")

    # -------------------------
    # Sample Telemetry Loader
    # -------------------------
    def load_sample_data(self):
        df = pd.read_csv(self.sample_csv)
        for _, row in df.iterrows():
            yield row.to_dict()

    # -------------------------
    # Mock Telemetry Stream
    # -------------------------
    def mock_stream(self):
        for _ in range(50):
            yield {
                "engine_temp": np.random.uniform(3300, 3500),
                "fuel_flow_rate": np.random.uniform(240, 260),
                "chamber_pressure": np.random.uniform(9.5e6, 10e6),
                "oxidizer_flow_rate": np.random.uniform(250, 310),
                "vibration_engine": np.random.uniform(0, 0.05),
                "thrust_output": np.random.uniform(900, 1000),
                "cpu_temp": np.random.uniform(60, 85),
                "power_draw": np.random.uniform(400, 500),
                "data_bus_errors": np.random.randint(0, 5),
                "hull_stress": np.random.uniform(0, 0.005),
                "hull_vibration": np.random.uniform(0, 0.02),
                "external_temp": np.random.uniform(-100, 50),
                "altitude": np.random.uniform(0, 100000),
                "acceleration": np.random.uniform(0, 50),
                "burn_duration": np.random.uniform(0, 60),
                "mission_time": np.random.uniform(0, 500),
            }


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    agent = RocketChatAgent()
    agent.run()
