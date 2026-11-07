"""Edge-box engine: consume read-only MQTT-style telemetry messages, aggregate to 1-min windows, diagnose, alert.

    python -m retrofit_sim replay --data pack/data/full --asset PUMP-08 | python -m retrofit_ai.edge --asset PUMP-08

Input: JSON Lines {"topic": ..., "payload": {timestamp_utc, operating_mode, <channels>...}} on stdin (the organizer's
replay adapter). Output: one JSON line per advisory alert on stdout (topic retrofit/<asset>/alerts), plus a footprint
report (latency per minute, memory) on stderr at the end. The engine only reads; it has no command or write path.

Operating context (duty demand, inlet temperature, ...) is read from the plant/vessel automation; in this demo it is
taken from operating_context.csv up to the current minute.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from retrofit_ai import config  # noqa: E402
from retrofit_ai.config import CHANNELS  # noqa: E402
from retrofit_ai.data import Dataset  # noqa: E402
from retrofit_ai.explain import explain  # noqa: E402
from retrofit_ai.features import features_from_frame  # noqa: E402
from retrofit_ai.model import load_models, predict_asset  # noqa: E402
from retrofit_ai.policy import alerts_for_asset  # noqa: E402


class MinuteAggregator:
    """1 Hz rows -> 1-min aggregates in the features_1min schema (mean/std/min/max/count per channel)."""

    def __init__(self):
        self.minute = None
        self.rows = []
        self.done = []

    def add(self, p: dict):
        ts = pd.Timestamp(p["timestamp_utc"])
        m = ts.floor("min")
        if self.minute is not None and m != self.minute:
            self._close()
        self.minute = m
        self.rows.append(p)

    def _close(self):
        df = pd.DataFrame(self.rows)
        rec = {"window_start_utc": self.minute}
        for c in CHANNELS:
            v = pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(dtype=float)
            v = v.dropna()
            rec[f"{c}__mean"] = v.mean() if len(v) else np.nan
            rec[f"{c}__std"] = v.std() if len(v) > 1 else np.nan
            rec[f"{c}__min"] = v.min() if len(v) else np.nan
            rec[f"{c}__max"] = v.max() if len(v) else np.nan
            rec[f"{c}__count"] = len(v)
        modes = df["operating_mode"].astype(str)
        rec["operating_mode"] = modes.mode().iloc[0]
        rec["running_fraction"] = float((modes == "RUNNING").mean())
        rec["rows"] = len(df)
        rec["quality_any"] = int(np.bitwise_or.reduce(df.get("quality_flags", pd.Series([0])).fillna(0).astype(int).values))
        self.done.append(rec)
        self.rows = []

    def flush(self):
        if self.rows:
            self._close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--data", default=config.DATA_FULL)
    ap.add_argument("--every", type=int, default=1, help="run the diagnosis every N completed minutes")
    ap.add_argument("--footprint", default=os.path.join(config.RESULTS, "edge_footprint.json"))
    a = ap.parse_args(argv)
    ds = Dataset(a.data, "edge")
    row = ds.assets[ds.assets.asset_id == a.asset].iloc[0]
    ctx = ds.context[ds.context.asset_id == a.asset].drop(columns=["asset_id"])
    models, feats = load_models()
    from scripts.predict import load_policy
    params = load_policy()
    t_start = None                       # monitoring starts with the first received data
    agg = MinuteAggregator()
    sent = set()
    lat = []
    n_msg = 0
    last_done = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if t_start is None:
            t_start = pd.Timestamp(msg["payload"]["timestamp_utc"]).floor("min")
        agg.add(msg["payload"])
        n_msg += 1
        if len(agg.done) - last_done < a.every:
            continue
        last_done = len(agg.done)
        t0 = time.perf_counter()
        mins = pd.DataFrame(agg.done)
        end = mins["window_start_utc"].max()
        grid = pd.DataFrame({"window_start_utc": pd.date_range(t_start, end, freq="1min")})
        fr = grid.merge(mins, on="window_start_utc", how="left").merge(ctx, on="window_start_utc", how="left")
        fr["rows"] = fr["rows"].fillna(0)
        fr["running_fraction"] = fr["running_fraction"].fillna(0.0)
        fr["operating_mode"] = fr["operating_mode"].astype(object).where(fr["operating_mode"].notna(), "NO_DATA")
        fr.insert(0, "asset_id", a.asset)
        f = features_from_frame(fr, row, None)
        P = predict_asset(f, models, feats)
        for al in alerts_for_asset(P, params):
            key = (al["fault_type"], al["alert_time_utc"])
            if key in sent:
                continue
            sent.add(key)
            ev, act = explain(al["fault_type"], f.iloc[al["_row"]], row)
            out = {"topic": f"retrofit/{a.asset}/alerts", "payload": {k: al[k] for k in ("asset_id", "alert_time_utc", "event_class",
                                                                                          "fault_type", "confidence")} | {"evidence": ev, "recommended_action": act}}
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
        lat.append(time.perf_counter() - t0)
    agg.flush()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    rep = {"asset": a.asset, "messages": n_msg, "minutes_processed": len(agg.done), "diagnosis_runs": len(lat),
           "latency_ms_median": round(1000 * float(np.median(lat)), 1) if lat else None,
           "latency_ms_p95": round(1000 * float(np.percentile(lat, 95)), 1) if lat else None,
           "peak_memory_mb": round(rss, 1), "alerts": len(sent),
           "note": "full-history recomputation each minute (simple, deterministic); an incremental version would be cheaper"}
    sys.stderr.write(json.dumps(rep) + "\n")
    if a.footprint:
        os.makedirs(os.path.dirname(a.footprint), exist_ok=True)
        with open(a.footprint, "w") as fh:
            json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()
