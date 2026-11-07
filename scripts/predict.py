"""Produce alert files (official template format) with evidence and recommended actions.

Usage:
  python scripts/predict.py --split validation      # writes results/predictions_validation.csv and prints the score
  python scripts/predict.py --split test            # writes submission/predictions_test.csv
  python scripts/predict.py --split train
Optional: --data <dir in organizer layout> to run on another data directory (features computed on the fly).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from retrofit_ai import config  # noqa: E402
from retrofit_ai.data import Dataset  # noqa: E402
from retrofit_ai.evaluate import per_type_table, score_alerts, summary  # noqa: E402
from retrofit_ai.explain import explain  # noqa: E402
from retrofit_ai.features import build_features  # noqa: E402
from retrofit_ai.model import load_models, predict_asset  # noqa: E402
from retrofit_ai.policy import PolicyParams, run_policy  # noqa: E402

TEMPLATE_COLS = ["asset_id", "alert_time_utc", "event_class", "fault_type", "confidence", "evidence", "recommended_action"]


def load_policy(path=os.path.join(config.MODELS, "policy.json")) -> PolicyParams:
    if os.path.exists(path):
        with open(path) as fh:
            return PolicyParams.from_dict(json.load(fh))
    return PolicyParams()


def alerts_with_explanations(ds: Dataset, asset_ids: list, models, feats, params, use_cache=True):
    rows = []
    for aid in asset_ids:
        cp = os.path.join(config.BUILD, "features", ds.name, f"{aid}.parquet")
        f = pd.read_parquet(cp) if (use_cache and os.path.exists(cp)) else build_features(ds, aid)
        P = predict_asset(f, models, feats)
        al = run_policy([P], params)
        row = ds.assets[ds.assets.asset_id == aid].iloc[0]
        for _, a in al.iterrows():
            j = int(a["_row"])
            fr = f.iloc[j]
            probs = {k: float(P[k].iloc[j]) for k in models if k in P and pd.notna(P[k].iloc[j])}
            ev, act = explain(a["fault_type"], fr, row, probs)
            rows.append({**{c: a[c] for c in ("asset_id", "alert_time_utc", "event_class", "fault_type", "confidence")},
                         "evidence": ev, "recommended_action": act})
    return pd.DataFrame(rows, columns=TEMPLATE_COLS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ds = Dataset(a.data or config.DATA_FULL, "official" if a.data is None else "custom")
    ids = list(ds.assets[ds.assets.split == a.split].asset_id)
    models, feats = load_models()
    params = load_policy()
    out = alerts_with_explanations(ds, ids, models, feats, params, use_cache=a.data is None)
    path = a.out or (os.path.join(config.ROOT, "submission", "predictions_test.csv") if a.split == "test"
                     else os.path.join(config.RESULTS, f"predictions_{a.split}.csv"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    print(f"wrote {len(out)} alerts for {len(ids)} assets -> {path}")
    if a.split != "test" and len(ds.events):
        r = score_alerts(out, ds.events, ds.assets, ds.benign, ids)
        print(summary(r))
        print(per_type_table(r))
        pe = pd.DataFrame(r["per_event"])[["event_id", "fault_type", "detected", "predicted_fault_type", "timeliness",
                                           "delay_after_detectable_h"]]
        print(pe.to_string(index=False))
