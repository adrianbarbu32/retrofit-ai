"""Minute-level training labels from event tables (same window conventions as the official scorer)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ALL_TYPES

SOFT_END = {"CAVITATION", "AIR_ENTRAINMENT"}


def _ts(s):
    return pd.to_datetime(s, utc=True, format="ISO8601")


def event_windows(events: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    """Scorer windows: start = max(onset, detectable - 6 h, window start); end per BENCHMARK_RULES section 4."""
    ev = events.copy()
    for c in ("onset_utc", "detectable_utc", "action_deadline_utc", "end_utc"):
        ev[c] = _ts(ev[c])
    ev["observable"] = ev["observable"].astype(str).str.lower().isin(["true", "1"])
    ws = dict(zip(assets.asset_id, _ts(assets.window_start_utc)))
    we = dict(zip(assets.asset_id, _ts(assets.window_end_utc)))
    st, en = [], []
    for _, e in ev.iterrows():
        s = max(e["onset_utc"], ws[e["asset_id"]])
        if pd.notna(e["detectable_utc"]):
            s = max(s, e["detectable_utc"] - pd.Timedelta(hours=6))
        st.append(s)
        if pd.isna(e["end_utc"]):
            en.append(we[e["asset_id"]])
        else:
            soft = e["fault_type"] in SOFT_END or e["event_class"] == "SENSOR"
            en.append(min(e["end_utc"] + (pd.Timedelta(minutes=30) if soft else pd.Timedelta(0)), we[e["asset_id"]]))
    ev["w_start"], ev["w_end"] = st, en
    return ev


def minute_labels(times: pd.Series, ev_asset: pd.DataFrame) -> pd.DataFrame:
    """y_<TYPE>: 1 inside [detectable, w_end] of an observable event, NaN (ignored) in the pre-detectable part of the
    window and over unobservable events, 0 elsewhere. Also y_ANY and the matching event id of the earliest active event."""
    t = pd.to_datetime(times, utc=True).values
    out = {f"y_{k}": np.zeros(len(t), dtype=np.float32) for k in ALL_TYPES}
    any_pos = np.zeros(len(t), dtype=np.float32)
    for _, e in ev_asset.iterrows():
        k = e["fault_type"]
        if k not in ALL_TYPES:
            continue
        y = out[f"y_{k}"]
        a = np.datetime64(e["w_start"].tz_convert(None)) if hasattr(e["w_start"], "tz_convert") else e["w_start"]
        b = np.datetime64(e["w_end"].tz_convert(None))
        tt = t.astype("datetime64[ns]")
        win = (tt >= np.datetime64(a, "ns")) & (tt <= np.datetime64(b, "ns"))
        if not e["observable"] or pd.isna(e["detectable_utc"]):
            y[win & (y == 0)] = np.nan
            continue
        d = np.datetime64(e["detectable_utc"].tz_convert(None), "ns")
        pos = win & (tt >= d)
        pre = win & (tt < d)
        y[pre & (y != 1)] = np.nan
        y[pos] = 1.0
        any_pos[pos] = 1.0
    out["y_ANY"] = any_pos
    return pd.DataFrame(out)
