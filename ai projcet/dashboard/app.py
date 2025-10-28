import streamlit as st
import pandas as pd
from plots import line_plot, bar_plot, summary_table

# ----------------------------------
# Load real telemetry data
# ----------------------------------
@st.cache_data
def load_data(dir_path="../sample_tel", file_name="telemetry_sample.csv"):
    import os

    # Get absolute path of this file (dashboard/app.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Build full path to telemetry CSV (go up one level from dashboard/)
    csv_path = os.path.join(base_dir, dir_path, file_name)
    csv_path = os.path.normpath(csv_path)

    if not os.path.exists(csv_path):
        st.error(f"❌ Telemetry file not found at: {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path)
    return df



df = load_data()

# ----------------------------------
# Streamlit App Layout
# ----------------------------------
st.set_page_config(page_title="Rocket Telemetry Dashboard", layout="wide")
st.title("🚀 Rocket Telemetry Dashboard")

# Sidebar controls
st.sidebar.header("Telemetry Controls")

# Select available columns for dynamic plotting
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
time_col = "mission_time" if "mission_time" in df.columns else numeric_cols[0]

x_col = st.sidebar.selectbox("X-axis", [time_col] + numeric_cols)
y_col = st.sidebar.selectbox("Y-axis", numeric_cols, index=numeric_cols.index("engine_temp") if "engine_temp" in numeric_cols else 0)
chart_type = st.sidebar.radio("Chart Type", ["Line Plot", "Bar Plot"])

st.sidebar.markdown("---")
st.sidebar.write("Use the dropdowns above to explore telemetry metrics like:")
st.sidebar.markdown("🧱 `engine_temp`, `thrust_output`, `chamber_pressure`, `fuel_flow_rate`, etc.")

# ----------------------------------
# Main Visualization Area
# ----------------------------------
st.subheader(f"{chart_type}: {y_col} vs {x_col}")

if chart_type == "Line Plot":
    fig = line_plot(x_col=x_col, y_col=y_col, title=f"{y_col} vs {x_col}")
else:
    fig = bar_plot(x_col=x_col, y_col=y_col, title=f"{y_col} vs {x_col}")

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------
# Summary Table
# ----------------------------------
st.subheader("📊 Telemetry Summary (Top 10 Records)")
st.dataframe(summary_table(cols=[x_col, y_col]))
