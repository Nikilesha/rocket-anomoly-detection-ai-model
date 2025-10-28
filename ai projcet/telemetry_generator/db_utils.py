# rocket_telemetry_generator/db_utils.py
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import pandas as pd
import os
from .config import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB

def save_telemetry(data):
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = connection.cursor()

        query = """
        INSERT INTO telemetry (
            timestamp, engine_temp, fuel_flow_rate, chamber_pressure, oxidizer_flow_rate,
            vibration_engine, thrust_output, cpu_temp, power_draw, data_bus_errors,
            hull_stress, hull_vibration, external_temp, altitude, acceleration,
            burn_duration, mission_time
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            datetime.now(),
            data.get("engine_temp"),
            data.get("fuel_flow_rate"),
            data.get("chamber_pressure"),
            data.get("oxidizer_flow_rate"),
            data.get("vibration_engine"),
            data.get("thrust_output"),
            data.get("cpu_temp"),
            data.get("power_draw"),
            data.get("data_bus_errors"),
            data.get("hull_stress"),
            data.get("hull_vibration"),
            data.get("external_temp"),
            data.get("altitude"),
            data.get("acceleration"),
            data.get("burn_duration"),
            data.get("mission_time")
        )

        cursor.execute(query, values)
        connection.commit()
        print(f"[✅] Telemetry saved: {values}")

    except Error as e:
        print(f"[❌] Error saving telemetry: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def fetch_all_telemetry():
    """
    Fetch all telemetry records from MySQL as a DataFrame
    """
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        df = pd.read_sql("SELECT * FROM telemetry ORDER BY timestamp ASC", connection)
        return df
    except Error as e:
        print(f"[❌] Error fetching telemetry: {e}")
        return pd.DataFrame()
    finally:
        if connection.is_connected():
            connection.close()
