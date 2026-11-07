"""Healthy digital twin at 1-minute resolution (deterministic, causal).

For every minute, what would a *healthy* pump with this public datasheet (assets.csv) show under the observed operating
context (duty demand, inlet temperature, room temperature, suction level, run state)? The equations are the documented
reduced-order model (TECHNICAL_MODEL.md sections 2-7, using `retrofit_sim.physics`), evaluated with a healthy latent
state and no random terms. Unknown per-asset details (exact motor efficiency, vibration base levels) use documented
profile defaults, so absolute residuals carry an asset-specific bias that the diagnosis model learns to handle.

Causality: each minute uses only that minute's context plus first-order lags driven by earlier minutes.
"""
from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd

from . import config  # noqa: F401  (adds the simulator to sys.path)
from retrofit_sim import physics as ph

EFF_TABLE = ([5.5, 11, 15, 30, 45, 55, 75, 90], [0.89, 0.91, 0.92, 0.936, 0.945, 0.95, 0.955, 0.957])
HEALTHY = {"bearing_wear": 0.05, "clog": 0.05, "restriction": 1.0, "misalignment": 0.02, "unbalance": 0.0,
           "impeller_wear": 0.0, "air": 0.0}


def load_profiles():
    import os

    import yaml
    with open(os.path.join(config.PACK, "configs", "fleet_public.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)["profiles"]


def asset_params(row: pd.Series, profiles: dict, setpoint_text: str | None = None) -> dict:
    """Simulator parameters rebuilt from the public asset sheet + documented profile defaults."""
    a = copy.deepcopy(profiles[row["profile"]])
    a.update({
        "asset_id": row["asset_id"], "profile": row["profile"], "control": row["control"],
        "q_bep_m3h": float(row["pump_rated_flow_m3_h"]), "h_bep_m": float(row["pump_rated_head_m"]),
        "h0_shutoff_ratio": float(row["pump_shutoff_head_m"]) / float(row["pump_rated_head_m"]),
        "eta_bep": float(row["pump_bep_efficiency"]), "npshr_bep_m": float(row["npsh3_at_bep_m"]),
        "q_design_m3h": float(row["design_flow_m3_h"]), "h_design_m": float(row["design_head_m"]),
        "system_static_m": float(row["static_head_m"]), "motor_kw": float(row["motor_rated_kw"]),
        "rated_rpm": float(row["motor_rated_rpm"]), "voltage_v": float(row["supply_voltage_v"]),
        "rated_current_a": float(row["motor_rated_current_a"]),
    })
    kw = a["motor_kw"]
    eff = float(np.interp(kw, *EFF_TABLE))
    pf = kw * 1000.0 / (eff * np.sqrt(3) * a["voltage_v"] * a["rated_current_a"])
    a["motor_eff"], a["motor_pf"] = eff, float(np.clip(pf, 0.78, 0.92))
    a.setdefault("design_supply_temp_c", 25.0 if row["profile"] == "SHIP_SW_COOLING" else 32.0)
    if pd.notna(row.get("min_speed_ratio")):
        a["r_min"] = float(row["min_speed_ratio"])
    if a["control"] == "VFD_PRESSURE":
        m = re.search(r"([-\d.]+)\s*kPa", str(setpoint_text or ""))
        a["dp_setpoint_m"] = float(m.group(1)) / 9.81 if m else 0.9 * a["h_design_m"]
    ctx = a.get("context", {})
    a["_suction_static_default"] = float(ctx.get("suction_static_m", ctx.get("sea_chest_submergence_m", 3.0)))
    return a


def _lag(x, dt_s, tau_s, x0=None):
    from scipy.signal import lfilter
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    a = np.exp(-dt_s / tau_s)
    zi = np.array([(x[0] if x0 is None else x0) * a])
    y, _ = lfilter([1 - a], [1, -a], x, zi=zi)
    return y


def simulate_healthy(A: dict, demand, supply_temp, ambient, suction_static, running, dt_s: float = 60.0,
                     latent: dict | None = None) -> dict:
    """Vectorised healthy (or hypothesised) plant state per time step. Arrays of equal length."""
    L = dict(HEALTHY)
    if latent:
        L.update(latent)
    dem = np.asarray(demand, dtype=float)
    t_sup = np.asarray(supply_temp, dtype=float)
    amb = np.asarray(ambient, dtype=float)
    z_s = np.asarray(suction_static, dtype=float)
    run = np.asarray(running, dtype=bool)
    n = len(dem)
    Qb = A["q_bep_m3h"] / 3600.0
    Hb, h0, eta_b = A["h_bep_m"], A["h0_shutoff_ratio"], A["eta_bep"]
    rho = ph.water_density(t_sup, A.get("salinity_psu", 0.0))
    H_static = np.full(n, A["system_static_m"])
    Qd = A["q_design_m3h"] / 3600.0
    K_design = (A["h_design_m"] - A["system_static_m"]) / Qd ** 2
    v_des = A.get("valve_design_position", 0.8)
    K_valve_full = K_design * A.get("valve_share_of_K", 0.25) * (50.0 ** (v_des - 1.0)) ** 2
    K_s0 = A.get("suction_loss_m_at_design", 0.8) / Qd ** 2
    K_suction = K_s0 * (1.0 + A.get("strainer_share", 0.6) * ((1.0 + L["clog"]) ** 2 - 1.0))
    K_pipe = max(K_design * (1.0 - A.get("valve_share_of_K", 0.25)) - K_s0, 0.1 * K_design) * L["restriction"]
    hfw = 1.0 - L["impeller_wear"]
    ctrl = A["control"]
    valve = np.full(n, v_des)
    q_req = np.full(n, np.nan)
    if ctrl == "VFD_TEMPERATURE":
        t_set = A.get("controlled_temp_setpoint_c", 36.0)
        t_ref = A.get("design_supply_temp_c", 25.0)
        need = dem * (t_set - t_ref) / np.maximum(t_set - t_sup, 3.0)
        q_req = np.clip(need, A.get("min_flow_frac", 0.4), 1.15) * Qd
        K_nom = K_pipe + ph.valve_K(valve, K_valve_full) + K_suction
        r_need = ph.speed_for_flow(q_req, Hb, Qb, h0, H_static, K_nom, hfw)
        r_t = np.clip(r_need, A.get("r_min", 0.55), A.get("r_max", 1.0))
    elif ctrl == "VFD_PRESSURE":
        valve = np.clip(0.35 + 0.65 * dem, 0.2, 1.0)
        dp = A.get("dp_setpoint_m", 0.9 * A["h_design_m"])
        K_nom = K_pipe + ph.valve_K(valve, K_valve_full) + K_suction
        Q2 = np.maximum(dp - H_static, 0.1) / K_nom
        r_need = np.sqrt((dp + Hb * hfw * (h0 - 1) / Qb ** 2 * Q2) / (Hb * h0 * hfw))
        r_t = np.clip(r_need, A.get("r_min", 0.5), A.get("r_max", 1.0))
    elif ctrl == "FLOW_VALVE":
        q_sp = np.clip(dem, 0.2, 1.1) * Qd
        q_req = q_sp
        r_t = np.ones(n)
        K_needed = (Hb * h0 * hfw - H_static) / np.maximum(q_sp, 1e-4) ** 2 - Hb * (h0 - 1) * hfw / Qb ** 2 - K_pipe - K_suction
        rel = np.sqrt(K_valve_full / np.maximum(K_needed, 1e-6))
        valve = np.clip(1.0 + np.log(np.clip(rel, 1e-3, 1.0)) / np.log(50.0), 0.05, 1.0)
    else:
        r_t = np.ones(n)
    if A.get("speed_override") is not None:
        r_t = np.full(n, float(A["speed_override"]))
    if A.get("valve_override") is not None:
        valve = np.full(n, float(A["valve_override"]))
    r = np.where(run, r_t, 0.0)
    K_total = K_pipe + ph.valve_K(valve, K_valve_full) + K_suction
    p_loop = A.get("loop_pressurisation_kpa", 0.0) * 1000.0
    pv = ph.vapour_pressure_pa(t_sup)
    hf = hfw * (1.0 - 2.5 * L["air"])
    A_suc = Qb / A.get("suction_velocity_bep_m_s", 2.5)

    def npsh(Qx):
        p_tot = ph.P_ATM + p_loop + rho * ph.G * (z_s - K_suction * Qx ** 2)
        return p_tot - 0.5 * rho * (Qx / A_suc) ** 2, (p_tot - pv) / (rho * ph.G), ph.npsh_required(Qx, r, Qb, A["npshr_bep_m"])

    Q_hi = ph.solve_operating_point(r, Hb, Qb, h0, H_static, K_total, hf)
    lo, hi = np.zeros(n), Q_hi.copy()
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        _, a_m, r_m = npsh(mid)
        g = ph.solve_operating_point(r, Hb, Qb, h0, H_static, K_total, hf * ph.cavitation_head_factor(a_m, r_m)[0])
        up = g > mid
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    Q = 0.5 * (lo + hi)
    p_s_abs, npsha, npshr = npsh(Q)
    cav, sigma = ph.cavitation_head_factor(npsha, npshr)
    Hh = ph.pump_head(Q, r, Hb, Qb, h0, hf * cav)
    p_s_abs = np.maximum(p_s_abs, pv + 50.0)
    p_d_abs = p_s_abs + rho * ph.G * Hh
    q_rel = np.where(r > 0, Q / (np.maximum(r, 1e-6) * Qb), 0.0)
    P_sh = ph.shaft_power(Q, Hh, r, Qb, Hb, eta_b, h0, rho, hf * cav) * (1 + 0.04 * L["bearing_wear"] ** 2 + 0.02 * L["misalignment"])
    P_r = A["motor_kw"] * 1000.0
    vfd = A.get("vfd_efficiency") if ctrl.startswith("VFD") else None
    P_in, _, _, I, lam = ph.motor_electrical(P_sh, P_r, A["motor_eff"], A["motor_pf"], A["voltage_v"], A.get("const_loss_share", 0.35), vfd)
    I = np.where(run, I, 0.0)
    P_in = np.where(run, P_in, 0.0)
    slip = A.get("rated_slip", 0.015) * np.clip(lam, 0, 1.5)
    rpm = r * A["rated_rpm"] / (1 - A.get("rated_slip", 0.015)) * (1 - slip) * (r > 0)
    P_hyd = rho * ph.G * Q * Hh
    dT = np.minimum(np.where(Q > 1e-4, (P_sh - P_hyd) * 0.9 / (rho * 4180.0 * np.maximum(Q, 1e-4)), 0.0), 25.0)
    t_fluid = t_sup + _lag(dT, dt_s, 60.0)
    bw, mis = L["bearing_wear"], L["misalignment"]
    rise = A.get("bearing_rise_k", 28.0) * r ** 1.3 * (0.75 + 0.25 * np.clip(lam, 0, 1.2))
    rise = rise + 40.0 * max(bw - 0.4, 0) ** 1.6 * (0.4 + 0.6 * r) + 6.0 * mis * r
    brg_ss = amb * 0.6 + t_fluid * 0.4 + np.where(run, rise, 0.0)
    t_brg = _lag(brg_ss, dt_s, A.get("bearing_tau_s", 1500.0))
    w_rise = A.get("winding_rise_k", 75.0) * (0.25 + 0.75 * np.clip(lam, 0, 1.4) ** 2)
    t_wnd = _lag(amb + np.where(run, w_rise, 0.0), dt_s, A.get("winding_tau_s", 2400.0))
    v_struct = A.get("vib_base_mm_s", 1.2) * (0.4 + 0.6 * r ** 1.5)
    v_imb = (A.get("unbalance_base_mm_s", 0.6) + 4.0 * L["unbalance"]) * r ** 2
    v_mis = 4.5 * mis * (0.5 + 0.5 * np.clip(lam, 0, 1.2)) * r
    v_hyd = A.get("hydraulic_vib_mm_s", 1.0) * r ** 2 * np.abs(q_rel - 1.0) ** 1.5 * (1.0 + 2.0 * (q_rel < 0.5))
    cav_int = np.clip((1.25 - sigma) / 0.45, 0, 1) * (r > 0.2)
    v_cav = A.get("cavitation_vib_mm_s", 3.0) * cav_int * r ** 2 + 3.0 * L["air"] * r
    v_brg = 14.0 * max(bw - 0.4, 0) ** 1.5 * (0.3 + 0.7 * r)
    vib = np.sqrt(v_struct ** 2 + v_imb ** 2 + v_mis ** 2 + v_hyd ** 2 + v_cav ** 2 + v_brg ** 2) * (r > 0.02)
    a_base = A.get("accel_base_g", 0.15) * (0.3 + 0.7 * r ** 2)
    a_brg = 2.2 * max(bw - 0.12, 0) ** 1.3 * r
    a_cav = 1.5 * cav_int * r ** 2 + 0.8 * L["air"]
    acc = np.sqrt(a_base ** 2 + a_brg ** 2 + a_cav ** 2 + (0.02 * vib) ** 2) * (r > 0.02)
    xb = max(bw - 0.12, 0)
    crest = np.where(r > 0.02, 3.2 + 50.0 * xb * np.exp(-6.0 * xb) + 1.0 * cav_int, np.nan)
    return {
        "speed_ratio": r, "motor_speed_rpm": rpm, "motor_current_a": I, "motor_power_kw": P_in / 1000.0,
        "suction_pressure_kpa": (p_s_abs - ph.P_ATM) / 1000.0, "discharge_pressure_kpa": (p_d_abs - ph.P_ATM) / 1000.0,
        "flow_m3_h": Q * 3600.0, "fluid_temp_c": t_fluid, "bearing_temp_c": t_brg, "motor_winding_temp_c": t_wnd,
        "vibration_rms_mm_s": vib, "vibration_accel_rms_g": acc, "vibration_crest_factor": crest,
        "valve_position_pct": valve * 100.0, "head_m": Hh, "q_rel": q_rel, "npsha_m": npsha, "npshr_m": npshr,
        "npsh_margin": sigma, "flow_required_m3h": np.where(run, q_req * 3600.0, np.nan), "motor_load": lam,
        "rho": rho, "vapour_kpa": pv / 1000.0, "suction_area_m2": np.full(n, A_suc), "cavitation_intensity": cav_int,
    }


def twin_for_asset(row: pd.Series, ctx: pd.DataFrame, running: np.ndarray, profiles: dict) -> pd.DataFrame:
    """Healthy expectations for one asset on the 1-min grid of `ctx` (operating_context rows of that asset)."""
    A = asset_params(row, profiles, ctx["controlled_setpoint"].dropna().iloc[0] if "controlled_setpoint" in ctx and ctx["controlled_setpoint"].notna().any() else None)
    z = ctx["suction_static_head_m"].to_numpy(dtype=float) if "suction_static_head_m" in ctx else np.full(len(ctx), np.nan)
    z = np.where(np.isnan(z), A["_suction_static_default"], z)
    dem = ctx["duty_demand_pct"].to_numpy(dtype=float) / 100.0
    out = simulate_healthy(A, dem, ctx["supply_water_temp_c"].to_numpy(dtype=float), ctx["ambient_temp_c"].to_numpy(dtype=float),
                           z, running)
    df = pd.DataFrame(out)
    df.insert(0, "window_start_utc", ctx["window_start_utc"].values)
    return df
