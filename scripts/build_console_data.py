"""Precompute alerts (with evidence) and per-minute probabilities for all 12 official pumps for the operator console.

Everything is causal, so showing the precomputed results up to replay time t is equivalent to running the engine live.
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from retrofit_ai import config  # noqa: E402
from retrofit_ai.data import Dataset  # noqa: E402
from retrofit_ai.model import load_models, predict_asset  # noqa: E402
from scripts.predict import alerts_with_explanations, load_policy  # noqa: E402

if __name__ == "__main__":
    ds = Dataset(config.DATA_FULL, "official")
    models, feats = load_models()
    params = load_policy()
    out_p = os.path.join(config.BUILD, "probs", "official")
    os.makedirs(out_p, exist_ok=True)
    for aid in ds.assets.asset_id:
        f = pd.read_parquet(os.path.join(config.BUILD, "features", "official", f"{aid}.parquet"))
        predict_asset(f, models, feats).to_parquet(os.path.join(out_p, f"{aid}.parquet"), index=False)
    al = alerts_with_explanations(ds, list(ds.assets.asset_id), models, feats, params)
    os.makedirs(os.path.join(config.RESULTS, "console"), exist_ok=True)
    al.to_csv(os.path.join(config.RESULTS, "console", "alerts_all.csv"), index=False)
    print(f"console data: {len(al)} alerts on {al.asset_id.nunique()} pumps")
