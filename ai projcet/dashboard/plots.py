import plotly.graph_objects as go
import pandas as pd
import os

# Function to load telemetry data from directory
def load_telemetry_data(dir_path="../sample_tel", file_name="telemetry_sample.csv"):
    csv_path = os.path.join(dir_path, file_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Telemetry file not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def line_plot(x_col, y_col, title="Telemetry Line Plot", dir_path="sample_Tel", file_name="telemetry_sample.csv"):
    df = load_telemetry_data(dir_path, file_name)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='lines+markers',
        name=y_col
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_dark'
    )
    return fig


def bar_plot(x_col, y_col, title="Telemetry Bar Plot", dir_path="sample_Tel", file_name="telemetry_sample.csv"):
    df = load_telemetry_data(dir_path, file_name)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[x_col],
        y=df[y_col],
        name=y_col
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_dark'
    )
    return fig


def summary_table(cols=None, dir_path="sample_Tel", file_name="telemetry_sample.csv"):
    df = load_telemetry_data(dir_path, file_name)
    if cols:
        df = df[cols]
    return df.head(10)
