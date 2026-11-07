"""Loaders for the organizer data layout (official pack and synthetic batches built with the same writer)."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd


def _ts(s):
    return pd.to_datetime(s, utc=True, format="ISO8601")


class Dataset:
    """One data directory in the organizer layout: <dir>/assets.csv, features_1min.parquet, operating_context.csv, ...

    For synthetic batches pass the batch root (containing public/ and private/)."""

    def __init__(self, path: str, name: str | None = None):
        pub = os.path.join(path, "public")
        self.root = path
        self.pub = pub if os.path.isdir(pub) else path
        self.name = name or os.path.basename(os.path.normpath(path))
        self.assets = pd.read_csv(os.path.join(self.pub, "assets.csv"))
        self._feat = None
        self._ctx = None
        self._snap = None
        self.events = self._load_events()
        bn = [p for p in ("benign_windows_train_validation.csv", "benign_windows_stress.csv") if os.path.exists(os.path.join(self.pub, p))]
        self.benign = pd.concat([pd.read_csv(os.path.join(self.pub, p)) for p in bn], ignore_index=True) if bn else pd.DataFrame(
            columns=["asset_id", "benign_type", "note", "start_utc", "end_utc"])
        mp = os.path.join(self.pub, "maintenance.csv")
        self.maintenance = pd.read_csv(mp) if os.path.exists(mp) else pd.DataFrame()

    def _load_events(self):
        parts = []
        for p in ("events_train.csv", "events_validation.csv"):
            f = os.path.join(self.pub, p)
            if os.path.exists(f):
                parts.append(pd.read_csv(f))
        f = os.path.join(self.root, "private", "events_stress_all.csv")
        if os.path.exists(f):
            parts.append(pd.read_csv(f))
        if not parts:
            return pd.DataFrame(columns=["event_id", "asset_id", "event_class", "fault_type", "onset_utc", "detectable_utc",
                                         "action_deadline_utc", "end_utc", "observable"])
        ev = pd.concat(parts, ignore_index=True).drop_duplicates("event_id")
        return ev

    @property
    def features(self) -> pd.DataFrame:
        if self._feat is None:
            f = pd.read_parquet(os.path.join(self.pub, "features_1min.parquet"))
            f["window_start_utc"] = pd.to_datetime(f["window_start_utc"], utc=True)
            self._feat = f
        return self._feat

    @property
    def context(self) -> pd.DataFrame:
        if self._ctx is None:
            c = pd.read_csv(os.path.join(self.pub, "operating_context.csv"))
            c["window_start_utc"] = _ts(c["timestamp_utc"])
            self._ctx = c.drop(columns=["timestamp_utc"])
        return self._ctx

    @property
    def snapshots_path(self) -> str | None:
        p = os.path.join(self.pub, "vibration_snapshots.parquet")
        return p if os.path.exists(p) else None

    def snapshots(self, asset_id: str) -> pd.DataFrame:
        p = self.snapshots_path
        if p is None:
            return pd.DataFrame()
        import pyarrow.parquet as pq
        t = pq.read_table(p, filters=[("asset_id", "=", asset_id)])
        return t.to_pandas()

    def asset_row(self, asset_id: str) -> pd.Series:
        return self.assets.set_index("asset_id").loc[asset_id].rename(asset_id).to_frame().T.reset_index(names="asset_id").iloc[0]

    def asset_frame(self, asset_id: str) -> pd.DataFrame:
        """Complete 1-min grid for one asset: 1-min features (NaN / rows=0 where absent) joined with operating context."""
        row = self.assets[self.assets.asset_id == asset_id].iloc[0]
        t0, t1 = _ts(row["window_start_utc"]), _ts(row["window_end_utc"])
        grid = pd.DataFrame({"window_start_utc": pd.date_range(t0, t1, freq="1min", inclusive="left")})
        f = self.features[self.features.asset_id == asset_id].drop(columns=["asset_id"])
        c = self.context[self.context.asset_id == asset_id].drop(columns=["asset_id"])
        df = grid.merge(f, on="window_start_utc", how="left").merge(c, on="window_start_utc", how="left")
        df["rows"] = df["rows"].fillna(0)
        df["running_fraction"] = df["running_fraction"].fillna(0.0)
        df["operating_mode"] = df["operating_mode"].astype(object).where(df["operating_mode"].notna(), "NO_DATA")
        df.insert(0, "asset_id", asset_id)
        return df


def synthetic_batches(root: str) -> list[str]:
    return sorted(p for p in glob.glob(os.path.join(root, "b*")) if os.path.isdir(os.path.join(p, "public")) and
                  os.path.exists(os.path.join(p, "public", "features_1min.parquet")))
