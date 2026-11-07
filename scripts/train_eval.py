"""Train per-type models and evaluate the full alert pipeline with the official scorer.

Examples
  python scripts/train_eval.py --synthetic-train b1000-b1012 --synthetic-holdout b1013-b1015
  python scripts/train_eval.py --official-only          # smoke test: train on PUMP-01..07, evaluate on PUMP-08/09
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from retrofit_ai import config  # noqa: E402
from retrofit_ai.data import Dataset  # noqa: E402
from retrofit_ai.evaluate import per_type_table, score_alerts, summary  # noqa: E402
from retrofit_ai.model import (build_training_table, feature_columns, load_cached, load_models, predict_asset,  # noqa: E402
                               train_models)
from retrofit_ai.policy import PolicyParams, run_policy  # noqa: E402

PROB_DIR = os.path.join(config.BUILD, "probs")


def expand(spec: str) -> list:
    if not spec:
        return []
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += [f"b{i}" for i in range(int(a.lstrip("b")), int(b.lstrip("b")) + 1)]
        else:
            out.append(part)
    return out


def predict_set(name: str, assets: list | None, models, feats) -> list:
    frames = []
    for p in load_cached([name], assets=set(assets) if assets else None):
        f = pd.read_parquet(p)
        P = predict_asset(f, models, feats)
        os.makedirs(os.path.join(PROB_DIR, name), exist_ok=True)
        P.to_parquet(os.path.join(PROB_DIR, name, os.path.basename(p)), index=False)
        frames.append(P)
    return frames


def eval_set(label, frames, ds_list, params, verbose=True):
    alerts = run_policy(frames, params)
    ev = pd.concat([d.events for d in ds_list], ignore_index=True)
    assets = pd.concat([d.assets for d in ds_list], ignore_index=True)
    bn = pd.concat([d.benign for d in ds_list], ignore_index=True)
    ids = sorted({P["asset_id"].iloc[0] for P in frames})
    r = score_alerts(alerts, ev, assets, bn, ids)
    if verbose:
        print(f"[{label}] {summary(r)}")
    return r, alerts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-train", default="")
    ap.add_argument("--synthetic-holdout", default="")
    ap.add_argument("--official-only", action="store_true")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--neg-p", type=float, default=0.12)
    a = ap.parse_args()
    off = Dataset(config.DATA_FULL, "official")
    train_ids = list(off.assets[off.assets.split == "train"].asset_id)
    val_ids = list(off.assets[off.assets.split == "validation"].asset_id)
    t0 = time.time()
    if not a.no_train:
        paths = load_cached(["official"], assets=set(train_ids)) + load_cached(expand(a.synthetic_train))
        print(f"training on {len(paths)} asset-weeks", flush=True)
        tab = build_training_table(paths, neg_p=a.neg_p)
        feats = feature_columns(tab)
        print(f"training table {tab.shape}, {len(feats)} features ({time.time() - t0:.0f}s)", flush=True)
        models = train_models(tab, feats, max_iter=a.max_iter)
        del tab
        print(f"trained in {time.time() - t0:.0f}s", flush=True)
    models, feats = load_models()
    params = PolicyParams()
    fr_tr = predict_set("official", train_ids, models, feats)
    fr_va = predict_set("official", val_ids, models, feats)
    eval_set("official-train", fr_tr, [off], params)
    r, al = eval_set("official-val", fr_va, [off], params)
    print(per_type_table(r))
    hold = expand(a.synthetic_holdout)
    if hold:
        frames, dss = [], []
        for name in hold:
            dss.append(Dataset(os.path.join(config.BUILD, "synth", name), name))
            frames += predict_set(name, None, models, feats)
        r, _ = eval_set("synthetic-holdout", frames, dss, params)
        print(per_type_table(r))
    print(f"total {time.time() - t0:.0f}s")
