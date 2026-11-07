"""Tune the alert policy (per-type threshold and persistence) by maximising the official composite score.

Tuning data: probability frames of synthetic hold-out batches (never used for model fitting). The official validation
pumps are only reported, never used for tuning, so their score stays an honest check.
Usage: python scripts/tune_policy.py --holdout b1013-b1015 [--passes 2]
Output: models/policy.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from retrofit_ai import config  # noqa: E402
from retrofit_ai.config import ALL_TYPES  # noqa: E402
from retrofit_ai.data import Dataset  # noqa: E402
from retrofit_ai.evaluate import per_type_table, score_alerts, summary  # noqa: E402
from retrofit_ai.policy import PolicyParams, run_policy, smooth  # noqa: E402
from scripts.train_eval import PROB_DIR, expand  # noqa: E402

THR_GRID = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
PERSIST_GRID = [3, 8, 20]


def load_probs(names, ids=None):
    frames = []
    for name in names:
        for p in sorted(glob.glob(os.path.join(PROB_DIR, name, "*.parquet"))):
            if ids is not None and os.path.basename(p)[:-8] not in ids:
                continue
            frames.append(pd.read_parquet(p))
    return frames


def make_scorer(frames, dss):
    ev = pd.concat([d.events for d in dss], ignore_index=True)
    assets = pd.concat([d.assets for d in dss], ignore_index=True)
    bn = pd.concat([d.benign for d in dss], ignore_index=True)
    ids = sorted({P["asset_id"].iloc[0] for P in frames})

    def f(params, cache):
        al = run_policy(frames, params, cache)
        return score_alerts(al, ev, assets, bn, ids)
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--win", type=int, default=10)
    ap.add_argument("--init", default=None, help="start from this policy json")
    a = ap.parse_args()
    names = expand(a.holdout)
    dss = [Dataset(os.path.join(config.BUILD, "synth", n), n) for n in names]
    frames = load_probs(names)
    off = Dataset(config.DATA_FULL, "official")
    val_ids = set(off.assets[off.assets.split == "validation"].asset_id)
    val_frames = load_probs(["official"], val_ids)
    print(f"tuning on {len(frames)} synthetic hold-out asset-weeks", flush=True)
    sc = make_scorer(frames, dss)
    sc_val = make_scorer(val_frames, [off])
    best_params = PolicyParams(win=a.win)
    if a.init and os.path.exists(a.init):
        best_params = PolicyParams.from_dict(json.load(open(a.init)))
        print(f"starting from {a.init}")
    cache = [dict() for _ in frames]
    cache_val = [dict() for _ in val_frames]
    t0 = time.time()
    r = sc(best_params, cache)
    best = r["composite_score"]
    print(f"start: {summary(r)} ({time.time() - t0:.1f}s/eval)", flush=True)
    types = [k for k in ALL_TYPES if k != "COMMUNICATION_LOSS"]
    for ps in range(a.passes):
        for k in types:
            for thr in THR_GRID:
                for pe in PERSIST_GRID:
                    p = PolicyParams.from_dict(json.loads(json.dumps(best_params.to_dict())))
                    p.thr[k], p.persist[k] = thr, pe
                    s = sc(p, cache)["composite_score"]
                    if s > best + 1e-6:
                        best, best_params = s, p
            # also test disabling the type entirely (if it only adds false alerts)
            p = PolicyParams.from_dict(json.loads(json.dumps(best_params.to_dict())))
            p.enabled[k] = False
            s = sc(p, cache)["composite_score"]
            if s > best + 1e-6:
                best, best_params = s, p
            print(f"  pass {ps + 1} {k:26s} thr={best_params.thr[k]:.2f} persist={best_params.persist[k]:2d} "
                  f"enabled={best_params.enabled[k]} -> {best:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    # re-arm horizon per type (how long a type must stay quiet before a new alert of that type is allowed)
    for k in types:
        for rearm in (60, 180, 360, 720, 1440):
            p = PolicyParams.from_dict(json.loads(json.dumps(best_params.to_dict())))
            p.rearm[k] = rearm
            s = sc(p, cache)["composite_score"]
            if s > best + 1e-6:
                best, best_params = s, p
    print(f"  re-arm tuned -> {best:.2f}", flush=True)
    # episodic faults: a pump stop ends the episode (process pumps run in batches)
    for k in ("CAVITATION", "AIR_ENTRAINMENT", "SENSOR_NOISE_SPIKES"):
        for stop in (None, 3, 10, 30):
            for pe in (2, 3, 5):
                p = PolicyParams.from_dict(json.loads(json.dumps(best_params.to_dict())))
                if stop is None:
                    p.rearm_on_stop.pop(k, None)
                else:
                    p.rearm_on_stop[k] = stop
                p.persist[k] = pe
                s = sc(p, cache)["composite_score"]
                if s > best + 1e-6:
                    best, best_params = s, p
        print(f"  {k} re-arm on stop={best_params.rearm_on_stop.get(k)} persist={best_params.persist[k]} -> {best:.2f}", flush=True)
    # competing hypotheses that should not both be announced for one episode
    pairs = [("SENSOR_DRIFT", "SENSOR_CALIBRATION_SHIFT"), ("SUCTION_STRAINER_CLOGGING", "AIR_ENTRAINMENT"),
             ("DISCHARGE_RESTRICTION", "SENSOR_CALIBRATION_SHIFT"), ("DISCHARGE_RESTRICTION", "SENSOR_DRIFT"),
             ("IMPELLER_WEAR", "SUCTION_STRAINER_CLOGGING"), ("MISALIGNMENT", "IMBALANCE"), ("IMPELLER_WEAR", "DISCHARGE_RESTRICTION")]
    for a1, a2 in pairs:
        p = PolicyParams.from_dict(json.loads(json.dumps(best_params.to_dict())))
        p.suppress.setdefault(a1, [])
        p.suppress.setdefault(a2, [])
        p.suppress[a1] = sorted(set(p.suppress[a1]) | {a2})
        p.suppress[a2] = sorted(set(p.suppress[a2]) | {a1})
        s = sc(p, cache)["composite_score"]
        print(f"  suppress pair {a1}/{a2}: {s:.2f} (best {best:.2f})", flush=True)
        if s > best + 1e-6:
            best, best_params = s, p
    r = sc(best_params, cache)
    print(f"tuned hold-out: {summary(r)}")
    print(per_type_table(r))
    rv = sc_val(best_params, cache_val)
    print(f"official validation (not used for tuning): {summary(rv)}")
    print(per_type_table(rv))
    os.makedirs(config.MODELS, exist_ok=True)
    with open(os.path.join(config.MODELS, "policy.json"), "w") as fh:
        json.dump(best_params.to_dict(), fh, indent=1)
    print("saved models/policy.json")
