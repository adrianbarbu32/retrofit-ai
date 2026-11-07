"""Per-fault-type probability models (gradient-boosted trees on twin-residual features)."""
from __future__ import annotations

import glob
import json
import os

import joblib
import numpy as np
import pandas as pd

from .config import ALL_TYPES, BUILD, MACHINE_TYPES, MODELS

NON_FEATURES = {"asset_id", "window_start_utc", "split", "dataset"}
FEAT_DIR = os.path.join(BUILD, "features")


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURES and not c.startswith("y_") and df[c].dtype != object]


def load_cached(names: list, assets: set | None = None, exclude_assets: set | None = None) -> list:
    out = []
    for name in names:
        for p in sorted(glob.glob(os.path.join(FEAT_DIR, name, "*.parquet"))):
            aid = os.path.basename(p)[:-8]
            if assets is not None and aid not in assets:
                continue
            if exclude_assets and aid in exclude_assets:
                continue
            out.append(p)
    return out


def _sample_rows(f: pd.DataFrame, rng, neg_p=0.12, early_min=240, rest_cap=400) -> np.ndarray:
    keep = rng.random(len(f)) < neg_p
    for k in ALL_TYPES + ["ANY"]:
        c = f"y_{k}"
        if c not in f:
            continue
        y = f[c].to_numpy()
        pos = y == 1
        if not pos.any():
            continue
        start = pos & ~np.r_[False, pos[:-1]]
        run_id = np.cumsum(start)
        # minutes since the start of this positive run
        idx = np.arange(len(y))
        first = pd.Series(np.where(start, idx, np.nan)).ffill().to_numpy()
        since = idx - first
        early = pos & (since < early_min)
        rest = pos & ~early
        p_rest = min(1.0, rest_cap / max(rest.sum(), 1))
        keep |= early | (rest & (rng.random(len(y)) < p_rest))
    return keep


def build_training_table(paths: list, seed=0, **kw) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for p in paths:
        f = pd.read_parquet(p)
        if "y_ANY" not in f:
            continue
        parts.append(f[_sample_rows(f, rng, **kw)])
    return pd.concat(parts, ignore_index=True, sort=False)


def train_models(train: pd.DataFrame, feats: list, out_dir: str = MODELS, max_iter=300, log=print) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    os.makedirs(out_dir, exist_ok=True)
    # one float32 matrix; per-model row subsets are taken (and freed) one at a time to bound memory
    X = np.empty((len(train), len(feats)), dtype=np.float32)
    for j, c in enumerate(feats):
        X[:, j] = train[c].to_numpy(dtype=np.float32, na_value=np.nan)
    steady = train["steady"].to_numpy() == 1
    Y = {k: train[f"y_{k}"].to_numpy(dtype=np.float32) for k in ALL_TYPES if f"y_{k}" in train}
    rng = np.random.default_rng(0)
    models = {}
    for k in ALL_TYPES:
        y = Y.get(k)
        if y is None:
            continue
        ok = ~np.isnan(y)
        if k in MACHINE_TYPES:
            ok &= steady                     # machine diagnoses only on steady running minutes
        idx = np.nonzero(ok)[0]
        yk = y[idx].astype(int)
        if yk.sum() < 50:
            log(f"  {k}: too few positives ({yk.sum()}), skipped")
            continue
        probe = X[rng.choice(idx, min(len(idx), 80000), replace=False)]
        cnt = np.isfinite(probe).sum(axis=0)
        var = np.nanstd(np.where(np.isfinite(probe), probe, np.nan), axis=0)
        cols = [j for j in range(len(feats)) if cnt[j] >= 20 and np.nan_to_num(var[j]) > 0]
        fk = [feats[j] for j in cols]
        Xk = X[np.ix_(idx, cols)]
        clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=60,
                                             l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                             n_iter_no_change=25, random_state=0)
        clf.fit(Xk, yk)
        del Xk
        clf.feature_list_ = fk
        models[k] = clf
        log(f"  {k}: n={len(idx)} pos={yk.sum()} iters={clf.n_iter_} feats={len(fk)}")
    joblib.dump({"models": models, "features": feats}, os.path.join(out_dir, "type_models.joblib"))
    return models


def load_models(path: str = os.path.join(MODELS, "type_models.joblib")):
    d = joblib.load(path)
    return d["models"], d["features"]


def predict_asset(f: pd.DataFrame, models: dict, feats: list) -> pd.DataFrame:
    X = f.reindex(columns=feats).astype(np.float32)
    out = pd.DataFrame({"asset_id": f["asset_id"].values, "window_start_utc": pd.to_datetime(f["window_start_utc"], utc=True).values})
    steady = f["steady"].to_numpy() == 1
    for k, m in models.items():
        p = m.predict_proba(X.reindex(columns=m.feature_list_).to_numpy())[:, 1]
        if k in MACHINE_TYPES:
            p = np.where(steady, p, np.nan)
        out[k] = p.astype(np.float32)
    for c in ("steady", "running", "locked", "gap", "gap_run", "since_lockout", "since_start", "transient_sched"):
        if c in f:
            out[c] = f[c].values
    return out
