"""Physics-based severity estimate: which latent condition level best explains the measured deviations right now?

For a diagnosed machine fault, a 1-D search over the latent variable (e.g. strainer clog factor) re-runs the healthy twin
with that condition under the current operating context and picks the level whose predicted readings match the
measured 30-min medians best (noise-normalised least squares over the fault's evidence channels). The estimate feeds
the sandbox what-if ("would cleaning the strainer now restore flow?") instead of a guessed hypothesis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .twin import asset_params, load_profiles, simulate_healthy

LATENT = {  # fault type -> (latent key, grid, evidence channels, sandbox hypothesis key)
    "SUCTION_STRAINER_CLOGGING": ("clog", np.r_[0.05, np.linspace(0.25, 8.0, 32)], ["suction_pressure_kpa", "motor_speed_rpm", "motor_power_kw",
                                                                                    "flow_m3_h", "valve_position_pct", "discharge_pressure_kpa", "motor_current_a"], "clog"),
    "DISCHARGE_RESTRICTION": ("restriction", np.linspace(1.0, 4.0, 31), ["discharge_pressure_kpa", "flow_m3_h", "motor_current_a", "motor_power_kw",
                                                                         "motor_speed_rpm", "valve_position_pct"], "restriction"),
    "IMPELLER_WEAR": ("impeller_wear", np.linspace(0.0, 0.15, 31), ["discharge_pressure_kpa", "motor_current_a", "motor_power_kw", "motor_speed_rpm",
                                                                    "flow_m3_h", "valve_position_pct"], "impeller_wear"),
    "IMBALANCE": ("unbalance", np.linspace(0.0, 2.0, 41), ["vibration_rms_mm_s"], "unbalance"),
    "MISALIGNMENT": ("misalignment", np.linspace(0.02, 1.0, 50), ["vibration_rms_mm_s", "bearing_temp_c"], "misalignment"),
    "BEARING_WEAR": ("bearing_wear", np.linspace(0.05, 0.95, 46), ["vibration_accel_rms_g", "vibration_crest_factor", "vibration_rms_mm_s"], "bearing_wear"),
    "AIR_ENTRAINMENT": ("air", np.linspace(0.0, 0.05, 26), ["discharge_pressure_kpa", "flow_m3_h", "motor_power_kw"], "air_entrainment"),
}
NOISE = {"suction_pressure_kpa": 1.5, "discharge_pressure_kpa": 2.5, "motor_speed_rpm": 3.0, "motor_power_kw": 0.3, "motor_current_a": 0.5,
         "flow_m3_h": 3.0, "valve_position_pct": 0.8, "vibration_rms_mm_s": 0.15, "vibration_accel_rms_g": 0.02, "vibration_crest_factor": 0.15,
         "bearing_temp_c": 0.8}
_PROF = {}


def estimate(ftype: str, frame: pd.DataFrame, j: int, row: pd.Series) -> dict | None:
    """frame: complete 1-min asset frame (Dataset.asset_frame); j: minute index of the alert. Uses minutes <= j only."""
    if ftype not in LATENT:
        return None
    key, grid, chans, hyp_key = LATENT[ftype]
    if "p" not in _PROF:
        _PROF["p"] = load_profiles()
    sp = frame["controlled_setpoint"].dropna().iloc[0] if frame["controlled_setpoint"].notna().any() else None
    A = asset_params(row, _PROF["p"], sp)
    lo = max(0, j - 29)
    w = frame.iloc[lo:j + 1]
    run = (w["running_fraction"] >= 0.99).to_numpy()
    if run.sum() < 5:
        return None
    w = w[run]
    meas = {c: float(np.nanmedian(w[f"{c}__mean"])) for c in chans if f"{c}__mean" in w and w[f"{c}__mean"].notna().any()}
    if not meas:
        return None
    dem = float(np.nanmedian(w["duty_demand_pct"])) / 100.0
    tin, amb = float(np.nanmedian(w["supply_water_temp_c"])), float(np.nanmedian(w["ambient_temp_c"]))
    z = float(np.nanmedian(w["suction_static_head_m"])) if w["suction_static_head_m"].notna().any() else A["_suction_static_default"]
    G = len(grid)
    errs = np.zeros(G)
    preds = []
    # steady-state evaluation (long settled horizon so thermal lags converge): run 240 identical minutes per grid value
    for g, val in enumerate(grid):
        n = 240
        out = simulate_healthy(A, np.full(n, dem), np.full(n, tin), np.full(n, amb), np.full(n, z), np.ones(n, bool),
                               latent={key: float(val)})
        p = {c: float(out[c][-1]) for c in meas}
        preds.append(p)
        errs[g] = sum(((meas[c] - p[c]) / NOISE.get(c, 1.0)) ** 2 for c in meas)
    # calibration offsets make absolute matching imperfect: compare against the healthy prediction of the same channel
    gi = int(np.argmin(errs))
    healthy = preds[0]
    return {"latent": key, "hypothesis_key": hyp_key, "value": float(grid[gi]), "fit_error": float(errs[gi]),
            "healthy_error": float(errs[0]), "measured": meas, "predicted": preds[gi], "healthy_pred": healthy,
            "context": {"demand": dem, "inlet_temp_c": tin, "ambient_c": amb, "suction_static_m": z}}
