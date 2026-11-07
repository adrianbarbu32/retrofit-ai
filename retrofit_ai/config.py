"""Paths and shared constants."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "pack")
DATA_FULL = os.path.join(PACK, "data", "full")
BUILD = os.path.join(ROOT, "build")
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")

# make the organizer's simulator package importable without installation
_SRC = os.path.join(PACK, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

MACHINE_TYPES = ["BEARING_WEAR", "SUCTION_STRAINER_CLOGGING", "CAVITATION", "DISCHARGE_RESTRICTION", "MISALIGNMENT",
                 "IMBALANCE", "IMPELLER_WEAR", "AIR_ENTRAINMENT"]
SENSOR_TYPES = ["SENSOR_DRIFT", "SENSOR_STUCK", "SENSOR_CALIBRATION_SHIFT", "SENSOR_NOISE_SPIKES", "COMMUNICATION_LOSS"]
ALL_TYPES = MACHINE_TYPES + SENSOR_TYPES
TYPE_CLASS = {**{t: "MACHINE" for t in MACHINE_TYPES}, **{t: "SENSOR" for t in SENSOR_TYPES}}

CHANNELS = ["motor_speed_rpm", "motor_current_a", "motor_power_kw", "suction_pressure_kpa", "discharge_pressure_kpa",
            "flow_m3_h", "fluid_temp_c", "bearing_temp_c", "motor_winding_temp_c", "vibration_rms_mm_s",
            "vibration_accel_rms_g", "vibration_crest_factor", "ambient_temp_c", "valve_position_pct"]
SHORT = {"motor_speed_rpm": "n", "motor_current_a": "i", "motor_power_kw": "pw", "suction_pressure_kpa": "ps",
         "discharge_pressure_kpa": "pd", "flow_m3_h": "q", "fluid_temp_c": "tf", "bearing_temp_c": "bt",
         "motor_winding_temp_c": "wt", "vibration_rms_mm_s": "v", "vibration_accel_rms_g": "a",
         "vibration_crest_factor": "c", "ambient_temp_c": "amb", "valve_position_pct": "vp"}
LONG = {v: k for k, v in SHORT.items()}
NICE = {"n": "speed", "i": "motor current", "pw": "motor power", "ps": "suction pressure", "pd": "discharge pressure",
        "q": "flow", "tf": "fluid temperature", "bt": "bearing temperature", "wt": "winding temperature",
        "v": "vibration velocity", "a": "vibration acceleration", "c": "crest factor", "vp": "valve position",
        "dh": "pump head (discharge - suction)", "amb": "room temperature"}
UNIT = {"n": "rpm", "i": "A", "pw": "kW", "ps": "kPa", "pd": "kPa", "q": "m3/h", "tf": "degC", "bt": "degC", "wt": "degC",
        "v": "mm/s", "a": "g", "c": "", "vp": "%", "dh": "kPa", "amb": "degC"}
