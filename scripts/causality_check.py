"""Causality check: alerts must not depend on future data.

For each pump, the engine is re-run on telemetry, context and waveform snapshots truncated at random cut points. Every
alert raised before the cut in the full run must be identical (time, class, type) in the truncated run, and vice versa.
Mirrors the organizers' audit (BENCHMARK_RULES.md section 2).
Usage: python scripts/causality_check.py [--assets PUMP-08 PUMP-09 PUMP-10 PUMP-11 PUMP-12] [--cuts 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from retrofit_ai import config  # noqa: E402
from retrofit_ai.data import Dataset  # noqa: E402
from retrofit_ai.features import features_from_frame  # noqa: E402
from retrofit_ai.model import load_models, predict_asset  # noqa: E402
from retrofit_ai.policy import alerts_for_asset  # noqa: E402
from scripts.predict import load_policy  # noqa: E402


def run(fr, row, snaps, models, feats, params):
    f = features_from_frame(fr, row, snaps)
    P = predict_asset(f, models, feats)
    return [(a["alert_time_utc"], a["event_class"], a["fault_type"]) for a in alerts_for_asset(P, params)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="*", default=["PUMP-08", "PUMP-09", "PUMP-10", "PUMP-11", "PUMP-12"])
    ap.add_argument("--cuts", type=int, default=3)
    a = ap.parse_args()
    ds = Dataset(config.DATA_FULL, "official")
    models, feats = load_models()
    params = load_policy()
    rng = np.random.default_rng(2026)
    report, ok_all = [], True
    for aid in a.assets:
        fr = ds.asset_frame(aid)
        row = ds.assets[ds.assets.asset_id == aid].iloc[0]
        snaps = ds.snapshots(aid) if ds.snapshots_path else None
        full = run(fr, row, snaps, models, feats, params)
        for cut_min in sorted(rng.integers(600, len(fr) - 60, a.cuts)):
            cut = fr["window_start_utc"].iloc[cut_min]
            fr_t = fr.iloc[:cut_min].copy()
            sn_t = None
            if snaps is not None and len(snaps):
                ts = pd.to_datetime(snaps["timestamp_utc"], utc=True)
                sn_t = snaps[ts < cut]
            trunc = run(fr_t, row, sn_t, models, feats, params)
            before = [x for x in full if pd.Timestamp(x[0]) <= cut]
            same = before == trunc
            ok_all &= same
            report.append({"asset": aid, "cut": str(cut), "alerts_before_cut_full": len(before), "alerts_truncated": len(trunc),
                           "identical": bool(same)})
            print(report[-1], flush=True)
            if not same:
                print("   full :", before[-5:])
                print("   trunc:", trunc[-5:])
    os.makedirs(config.RESULTS, exist_ok=True)
    with open(os.path.join(config.RESULTS, "causality_check.json"), "w") as fh:
        json.dump({"all_identical": bool(ok_all), "runs": report}, fh, indent=1)
    print("CAUSALITY CHECK:", "PASSED" if ok_all else "FAILED")
