"""Causal per-minute features for one asset: healthy-twin residuals, trends, noise, stuck/gap detectors, context.

All operations look backwards only (rolling windows ending at the current minute, forward fills, expanding
statistics), so the feature row of minute t depends on data with timestamps <= t + 1 min (the end of the minute).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SHORT
from .snapshots import asset_snapshot_features
from .twin import load_profiles, twin_for_asset

PROFILES = ["SHIP_SW_COOLING", "SHIP_LT_FRESHWATER", "INDUSTRIAL_CW", "PROCESS_TRANSFER"]
CONTROLS = ["VFD_TEMPERATURE", "VFD_PRESSURE", "FLOW_VALVE", "FIXED_SPEED"]
TIERS = ["T0_LEGACY", "T1_RETROFIT", "T2_FULL"]
TRANSIENT_STATES = {"arrival manoeuvring", "departure manoeuvring", "shift start", "shift end", "set-point change",
                    "start-up", "shutdown"}
REF_MIN = 12 * 60          # reference period: first 12 h of steady running
WIRELESS = {"vibration_rms_mm_s", "vibration_accel_rms_g", "vibration_crest_factor", "bearing_temp_c"}
_PROFILES_CACHE = {}


def _profiles():
    if "p" not in _PROFILES_CACHE:
        _PROFILES_CACHE["p"] = load_profiles()
    return _PROFILES_CACHE["p"]


def _run_length(mask: np.ndarray) -> np.ndarray:
    """Consecutive True count ending at each index."""
    out = np.zeros(len(mask), dtype=np.int32)
    c = 0
    for j, x in enumerate(mask):
        c = c + 1 if x else 0
        out[j] = c
    return out


def _since_last(mask: np.ndarray, cap: int = 100000) -> np.ndarray:
    """Minutes since the last True (cap if never)."""
    idx = np.where(mask, np.arange(len(mask)), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, np.arange(len(mask)) - last, cap).astype(np.int32)


def _reference(e: pd.Series, steady_cum: np.ndarray) -> pd.Series:
    """Median of e over the first REF_MIN steady minutes (expanding until then, frozen afterwards)."""
    in_ref = steady_cum <= REF_MIN
    ref = e.where(in_ref).expanding(min_periods=1).median()
    frozen = ref[in_ref].iloc[-1] if in_ref.any() and ref[in_ref].notna().any() else np.nan
    ref = ref.where(in_ref, frozen)
    return ref.ffill()


def _scales(row) -> dict:
    hd = 9.81 * float(row["design_head_m"])     # design head in kPa (water)
    return {"n": float(row["motor_rated_rpm"]) / 100.0, "i": float(row["motor_rated_current_a"]) / 100.0,
            "pw": float(row["motor_rated_kw"]) / 100.0, "ps": hd / 100.0, "pd": hd / 100.0, "dh": hd / 100.0,
            "q": float(row["design_flow_m3_h"]) / 100.0, "vp": 1.0, "tf": 1.0, "bt": 1.0, "wt": 1.0, "c": 1.0}


def build_features(ds, asset_id: str, with_snapshots: bool = True) -> pd.DataFrame:
    fr = ds.asset_frame(asset_id)
    row = ds.assets[ds.assets.asset_id == asset_id].iloc[0]
    snaps = ds.snapshots(asset_id) if (with_snapshots and ds.snapshots_path is not None) else None
    return features_from_frame(fr, row, snaps)


def features_from_frame(fr: pd.DataFrame, row: pd.Series, snaps: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features from a complete 1-min grid frame (1-min aggregates + operating context), e.g. an edge-box buffer."""
    n = len(fr)
    F = {"asset_id": fr["asset_id"].values, "window_start_utc": fr["window_start_utc"].values}
    mode = fr["operating_mode"].astype(str).values
    rows = fr["rows"].to_numpy(dtype=float)
    gap = rows < 30
    # pump state (during a data gap, assume the previous state persists)
    known = ~gap
    run_raw = (fr["running_fraction"].to_numpy(dtype=float) >= 0.99) & (mode == "RUNNING")
    ener_raw = fr["running_fraction"].to_numpy(dtype=float) > 0.5
    ener = pd.Series(np.where(known, ener_raw, np.nan)).ffill().fillna(0).astype(bool).values
    run = run_raw & known
    since_start = _run_length(run)
    steady = run & (since_start >= 10)
    steady_cum = np.cumsum(steady)
    locked = mode == "LOCKED_OUT"
    F["running"] = run.astype(np.int8)
    F["steady"] = steady.astype(np.int8)
    F["since_start"] = np.minimum(since_start, 720)
    F["since_lockout"] = np.minimum(_since_last(locked), 20000)
    F["locked"] = locked.astype(np.int8)
    F["gap"] = gap.astype(np.int8)
    F["gap_run"] = _run_length(gap)
    F["rows_frac"] = rows / 60.0
    F["gap_10"] = pd.Series(gap.astype(float)).rolling(10, min_periods=1).sum().values
    F["t_since_window"] = np.minimum(np.arange(n), 2880)
    F["profile"] = PROFILES.index(row["profile"])
    F["control"] = CONTROLS.index(row["control"])
    F["tier"] = TIERS.index(row["instrumentation_tier"])
    # context
    dem = fr["duty_demand_pct"].to_numpy(dtype=float)
    F["demand"] = dem
    F["d_demand10"] = np.abs(dem - pd.Series(dem).shift(10).fillna(pd.Series(dem)).values)
    sched = fr["schedule_state"].astype(str).values
    F["transient_sched"] = np.isin(sched, list(TRANSIENT_STATES)).astype(np.int8)
    F["sched_maint"] = (sched == "maintenance (locked out)").astype(np.int8)
    hs = fr["sea_state_hs_m"].to_numpy(dtype=float)
    F["sea_state"] = np.nan_to_num(hs, nan=0.0)
    F["sea_max60"] = pd.Series(F["sea_state"]).rolling(60, min_periods=1).max().values
    z = fr["suction_static_head_m"].to_numpy(dtype=float)
    zs = pd.Series(z)
    F["suction_level"] = z
    F["suction_level_drop"] = (zs.rolling(24 * 60, min_periods=30).median() - zs.rolling(15, min_periods=3).median()).values
    F["supply_temp"] = fr["supply_water_temp_c"].to_numpy(dtype=float)
    # healthy twin
    tw = twin_for_asset(row, fr, ener, _profiles())
    F["r_twin"] = tw["speed_ratio"].values
    F["qrel_twin"] = tw["q_rel"].values
    F["npsh_twin"] = tw["npsh_margin"].values
    sc = _scales(row)
    avail = [c for c in str(row["channels_available"]).split("|") if c]
    E = {}
    for col in avail:
        s = SHORT.get(col)
        if s is None or s == "amb":
            continue
        m = fr[f"{col}__mean"]
        cnt = fr[f"{col}__count"].fillna(0).to_numpy(dtype=float)
        exp = tw[col].to_numpy(dtype=float) if col in tw else np.full(n, np.nan)
        mv = m.to_numpy(dtype=float)
        if s in ("v", "a"):
            e = 100.0 * np.log(np.maximum(mv, 1e-4) / np.maximum(exp, 1e-4))
        else:
            e = (mv - exp) / sc[s]
        # wireless retrofit sensors (T1) report every 300 s: static property of the installation, not learned from data
        sparse = row["instrumentation_tier"] == "T1_RETROFIT" and col in WIRELESS
        e = pd.Series(np.where(steady & (cnt > 0), e, np.nan))
        E[s] = (e, sparse, mv, cnt)
    if "ps" in E and "pd" in E:
        dh = (E["pd"][2] - E["ps"][2] - (tw["discharge_pressure_kpa"].values - tw["suction_pressure_kpa"].values)) / sc["dh"]
        ok = steady & (E["pd"][3] > 0) & (E["ps"][3] > 0)
        E["dh"] = (pd.Series(np.where(ok, dh, np.nan)), False, None, None)
    # electrical consistency: current implied by measured power at the twin's operating point (independent of hydraulics)
    if "i" in E and "pw" in E:
        ratio = tw["motor_current_a"].values / np.maximum(tw["motor_power_kw"].values, 1e-3)
        ivp = (E["i"][2] - E["pw"][2] * ratio) / sc["i"]
        E["ivp"] = (pd.Series(np.where(steady, ivp, np.nan)), False, None, None)
    # pump-curve consistency (flow meter + pressures): head at the measured flow vs the healthy pump curve
    if "q" in E and "pd" in E:
        from retrofit_sim import physics as ph
        A_r = tw["speed_ratio"].values
        if "n" in E:
            A_r = np.where(np.isfinite(E["n"][2]), E["n"][2] / max(float(row["motor_rated_rpm"]), 1) * 1.0, A_r)
        qb = float(row["pump_rated_flow_m3_h"]) / 3600.0
        h0 = float(row["pump_shutoff_head_m"]) / float(row["pump_rated_head_m"])
        H_curve = ph.pump_head(E["q"][2] / 3600.0, np.maximum(A_r, 0.05), float(row["pump_rated_head_m"]), qb, h0)
        ps_m = E["ps"][2] if "ps" in E else tw["suction_pressure_kpa"].values
        h_meas = (E["pd"][2] - ps_m) / (9.81 * tw["rho"].values / 1000.0)
        cr = 100.0 * (h_meas - H_curve) / float(row["design_head_m"])
        E["curve"] = (pd.Series(np.where(steady, cr, np.nan)), False, None, None)
    for s, (e, sparse, mv, cnt) in E.items():
        win, mp = (30, 2) if sparse else (15, 5)
        lvl = e.rolling(win, min_periods=mp).median()
        ref = _reference(e, steady_cum)
        lff = lvl.ffill(limit=240)
        F[f"{s}_lvl"] = lvl.values
        F[f"{s}_dref"] = (lvl - ref).values
        F[f"{s}_d1h"] = (lvl - lff.shift(60)).values
        F[f"{s}_d3h"] = (lvl - lff.shift(180)).values
        F[f"{s}_d6h"] = (lvl - lff.shift(360)).values
        F[f"{s}_d12h"] = (lvl - lff.shift(720)).values
        F[f"{s}_lvl180"] = e.rolling(180, min_periods=10 if not sparse else 4).median().values
        # shape of the change: largest 30-min jump within the last 6 h (step) vs steady slope (ramp)
        j30 = (lvl - lff.shift(30)).abs()
        F[f"{s}_jump6h"] = j30.rolling(360, min_periods=30).max().values
        F[f"{s}_slope6h"] = ((lvl - lff.shift(120)) - (lff.shift(240) - lff.shift(360))).values
        if mv is None:
            continue
        col = [c for c, v in SHORT.items() if v == s][0]
        valid = cnt > 0
        # stuck detector: minutes since the reading last changed (ignoring zero readings of a stopped motor)
        vv = pd.Series(np.where(valid, mv, np.nan)).ffill()
        changed = valid & (np.r_[True, np.abs(np.diff(vv.values)) > 0])
        flat = np.arange(n) - pd.Series(np.where(changed, np.arange(n), np.nan)).ffill().fillna(0).values
        if s in ("n", "i", "pw"):
            flat = np.where(ener, flat, 0)
        F[f"{s}_flat"] = np.minimum(flat, 3000)
        if not sparse:
            sd = fr[f"{col}__std"].to_numpy(dtype=float)
            rg = (fr[f"{col}__max"] - fr[f"{col}__min"]).to_numpy(dtype=float)
            okm = steady & (cnt >= 30)
            sd_s = pd.Series(np.where(okm, sd, np.nan))
            rg_s = pd.Series(np.where(okm, rg, np.nan))
            ref_sd = _reference(sd_s, steady_cum)
            ref_rg = _reference(rg_s, steady_cum)
            F[f"{s}_noise"] = np.log((sd_s + 1e-3) / (ref_sd + 1e-3)).rolling(10, min_periods=3).median().values
            F[f"{s}_range"] = np.log((rg_s + 1e-3) / (ref_rg + 1e-3)).rolling(10, min_periods=3).max().values
            F[f"{s}_noise_abs"] = (sd_s / sc.get(s, 1.0)).rolling(10, min_periods=3).median().values if s in sc else np.nan
        else:
            raw = pd.Series(np.where(valid & steady, mv, np.nan))
            resid = raw - raw.rolling(60, min_periods=3).median()
            sdr = resid.abs().rolling(90, min_periods=4).median()
            ref_sdr = _reference(sdr, steady_cum)
            F[f"{s}_noise"] = np.log((sdr + 1e-3) / (ref_sdr + 1e-3)).values
            F[f"{s}_range"] = np.log((resid.abs().rolling(90, min_periods=4).max() + 1e-3) / (ref_sdr * 3 + 1e-3)).values
    # measured NPSH margin from suction pressure + fluid temperature (vapour pressure) vs the pump's NPSH3
    if "ps" in E:
        from retrofit_sim import physics as ph
        ps_m = E["ps"][2]
        q = E["q"][2] if "q" in E else tw["flow_m3_h"].values
        tf = E["tf"][2] if "tf" in E else F["supply_temp"]
        rho = tw["rho"].values
        v_s = (q / 3600.0) / tw["suction_area_m2"].values
        npsha = ((ps_m * 1000.0 + ph.P_ATM) + 0.5 * rho * v_s ** 2 - ph.vapour_pressure_pa(tf)) / (rho * ph.G)
        sig = npsha / np.maximum(tw["npshr_m"].values, 1e-3)
        sig = pd.Series(np.where(steady & np.isfinite(sig), sig, np.nan))
        F["npsh_meas"] = sig.rolling(5, min_periods=1).median().values
        F["npsh_meas_min30"] = sig.rolling(30, min_periods=1).min().values
    # absolute (twin-independent) vibration levels normalised by the speed dependence
    r = F["r_twin"]
    for col, s, shape in (("vibration_accel_rms_g", "a_abs", 0.3 + 0.7 * r ** 2), ("vibration_rms_mm_s", "v_abs", 0.4 + 0.6 * r ** 1.5),
                          ("vibration_crest_factor", "c_abs", np.ones(n))):
        if f"{col}__mean" in fr and col in avail:
            x = fr[f"{col}__mean"].to_numpy(dtype=float) / np.maximum(shape, 0.05)
            F[s] = pd.Series(np.where(steady, x, np.nan)).rolling(30, min_periods=1).median().values
    out = pd.DataFrame(F)
    if snaps is not None and len(snaps) and "vibration_accel_rms_g" in avail:
        sf = asset_snapshot_features(snaps, row)
        if len(sf):
            cols = [c for c in sf.columns if c not in ("snap_time",)]
            first = sf.iloc[0]
            for c in ("v1x", "v2x", "v3x", "bpfo_env", "kurt_res", "hf_rms", "vvp"):
                if c in sf:
                    sf[c + "_rel"] = np.log((sf[c] + 1e-4) / (first[c] + 1e-4))
            sf["v21"] = np.log((sf.get("v2x", 0) + 1e-3) / (sf.get("v1x", 0) + 1e-3))
            # available once the snapshot has been captured (+1 min processing)
            sf["window_start_utc"] = sf["snap_time"].dt.floor("min") + pd.Timedelta(minutes=1)
            out["window_start_utc"] = pd.to_datetime(out["window_start_utc"], utc=True)
            out = pd.merge_asof(out.sort_values("window_start_utc"), sf.drop(columns=["snap_time"]).sort_values("window_start_utc"),
                                on="window_start_utc", direction="backward", tolerance=pd.Timedelta(hours=8))
            out["snap_age_h"] = np.nan
    out["window_start_utc"] = pd.to_datetime(out["window_start_utc"], utc=True)
    for c in out.columns:
        if out[c].dtype == np.float64:
            out[c] = out[c].astype(np.float32)
    return out
