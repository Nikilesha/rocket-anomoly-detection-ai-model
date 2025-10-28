# nlp_module/report_generator.py

import re
from statistics import mean

def generate_summary_report(log_file="mission_logs.txt", output_file=None):
    """
    Parses mission log file and generates a concise summary of mission health.
    Returns the summary as a string. Optionally writes to output_file.
    """
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            logs = f.readlines()
    except FileNotFoundError:
        print(f"[⚠️] Log file '{log_file}' not found. Run mission logging first.")
        return ""

    total_logs = len(logs)
    if total_logs == 0:
        print("[⚠️] Log file is empty — no data to summarize.")
        return ""

    # Detect lines with anomalies
    anomaly_entries = [l for l in logs if "Detected anomalies" in l]

    # Extract failure probabilities
    fail_probs = []
    for l in logs:
        match = re.search(r"Failure Probability: ([0-9.]+)", l)
        if match:
            fail_probs.append(float(match.group(1)))

    avg_fail_prob = mean(fail_probs) if fail_probs else 0.0
    anomaly_rate = len(anomaly_entries) / total_logs * 100 if total_logs else 0.0

    # Build summary
    summary = f"""
🛰️ MISSION SUMMARY REPORT
----------------------------
Total Log Entries Analyzed: {total_logs}
Average Failure Probability: {avg_fail_prob:.2f}
Anomaly Rate: {anomaly_rate:.1f}%

Top Observations:
- {'Frequent anomalies detected — possible performance degradation.' if anomaly_rate > 30 else 'System remained stable with minimal anomalies.'}
- {'Average failure probability suggests moderate operational risk.' if avg_fail_prob > 0.5 else 'Failure probability stayed within safe range.'}

Recommendations:
- {"Perform deeper analysis on anomaly patterns." if anomaly_rate > 20 else "Routine check sufficient."}
- {"Inspect propulsion and vibration systems closely." if 'vibration_engine' in ''.join(anomaly_entries) else "No critical subsystems affected."}

---------------------------------
Generated automatically by Rocket NLP Report Module
""".strip()

    # Write to file if requested
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"[📄] Mission summary report saved to {output_file}\n")

    return summary
