"""Run the organizer baselines B0 (plant alarms) and B1 (mode-aware z, parameters chosen on the official train split:
k=6, persist=15) on any dataset in the organizer layout, scored with the official scorer.
Usage: python scripts/baselines_on.py <dir> [<dir> ...]
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from retrofit_ai import config  # noqa: E402,F401
from retrofit_ai.data import Dataset  # noqa: E402
from retrofit_ai.evaluate import score_alerts, summary  # noqa: E402
from retrofit_sim import baselines as B  # noqa: E402


def run(dirs, k=6, persist=15):
    ds = [Dataset(d) for d in dirs]
    assets = pd.concat([d.assets for d in ds], ignore_index=True)
    feat = pd.concat([pd.read_parquet(os.path.join(d.pub, "features_1min.parquet")) for d in ds], ignore_index=True)
    al = pd.concat([pd.read_csv(os.path.join(d.pub, "alarms.csv")) for d in ds], ignore_index=True)
    ev = pd.concat([d.events for d in ds], ignore_index=True)
    bn = pd.concat([d.benign for d in ds], ignore_index=True)
    zf = B.zscores(B.regime_frame(feat, assets))
    b1 = B.b1_threshold(zf, k, persist)
    b0 = B.b0_scada(al, set(assets.asset_id))
    return {"B0_scada_alarms": score_alerts(b0, ev, assets, bn), "B1_mode_aware_z": score_alerts(b1, ev, assets, bn)}


if __name__ == "__main__":
    res = run(sys.argv[1:])
    for k, r in res.items():
        print(f"[{k}] {summary(r)}")
