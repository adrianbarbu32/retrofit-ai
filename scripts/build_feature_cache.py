"""Compute per-asset feature tables (+ minute labels where events exist) and cache them as parquet.

Usage:
  python scripts/build_feature_cache.py --official            # pack/data/full (train, validation, test assets)
  python scripts/build_feature_cache.py --synthetic           # every finished batch under build/synth
Output: build/features/<dataset>/<asset_id>.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from retrofit_ai import config  # noqa: E402
from retrofit_ai.data import Dataset, synthetic_batches  # noqa: E402

OUT = os.path.join(config.BUILD, "features")


def _one(args):
    path, name, aid = args
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd

    from retrofit_ai.data import Dataset
    from retrofit_ai.features import build_features
    from retrofit_ai.labels import event_windows, minute_labels
    out = os.path.join(OUT, name, f"{aid}.parquet")
    if os.path.exists(out):
        return aid, 0.0, "cached"
    t0 = time.time()
    ds = Dataset(path, name)
    f = build_features(ds, aid)
    ev = ds.events[ds.events.asset_id == aid]
    if len(ev):
        ew = event_windows(ev, ds.assets)
        lab = minute_labels(f["window_start_utc"], ew)
        f = pd.concat([f.reset_index(drop=True), lab.reset_index(drop=True)], axis=1)
    f["split"] = ds.assets.set_index("asset_id").loc[aid, "split"]
    f["dataset"] = name
    os.makedirs(os.path.dirname(out), exist_ok=True)
    f.to_parquet(out, index=False)
    return aid, time.time() - t0, "ok"


def run(datasets, workers):
    jobs = []
    for path, name in datasets:
        ds = Dataset(path, name)
        jobs += [(path, name, a) for a in ds.assets.asset_id]
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for aid, dt, st in ex.map(_one, jobs, chunksize=1):
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} assets ({time.time() - t0:.0f}s)", flush=True)
    print(f"features done: {len(jobs)} assets in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dir", action="append", default=[], help="extra dataset dir (organizer layout), name = basename")
    a = ap.parse_args()
    ds = []
    if a.official:
        ds.append((config.DATA_FULL, "official"))
    if a.synthetic:
        ds += [(p, os.path.basename(p)) for p in synthetic_batches(os.path.join(config.BUILD, "synth"))]
    for d in a.dir:  # "path" or "path:name"
        path, _, name = d.partition(":")
        ds.append((path, name or os.path.basename(os.path.normpath(path))))
    run(ds, a.workers)
