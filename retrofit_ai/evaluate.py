"""Scoring harness around the official scorer (pack/src/retrofit_sim/scoring.py)."""
from __future__ import annotations

import io
import contextlib

import pandas as pd

from . import config  # noqa: F401
from retrofit_sim.scoring import score


def score_alerts(alerts: pd.DataFrame, events: pd.DataFrame, assets: pd.DataFrame, benign: pd.DataFrame | None = None,
                 asset_ids=None) -> dict:
    a = assets if asset_ids is None else assets[assets.asset_id.isin(asset_ids)]
    a = a.copy()
    a["split"] = "eval"
    ev = events[events.asset_id.isin(a.asset_id)]
    pred = alerts.drop(columns=[c for c in alerts.columns if c.startswith("_")], errors="ignore")
    if len(pred) == 0:
        pred = pd.DataFrame(columns=["asset_id", "alert_time_utc", "event_class", "fault_type", "confidence"])
    with contextlib.redirect_stdout(io.StringIO()):
        return score(pred, ev, a, benign, split="eval")


def summary(r: dict) -> str:
    return (f"composite={r['composite_score']:.1f} F1={r['event_f1']:.2f} rec={r['event_recall']:.2f} prec={r['alert_precision']:.2f} "
            f"tl={r['timeliness']:.2f} type={r['fault_type_recall']:.2f} cls={r['class_recall']:.2f} "
            f"fa/day={r['false_alerts_per_asset_day']:.2f} (events={r['events_scored']}, FA ep={r['false_alert_episodes']}, "
            f"excess={r['excess_repeat_episodes']})")


def per_type_table(r: dict) -> pd.DataFrame:
    pe = pd.DataFrame(r["per_event"])
    if not len(pe):
        return pe
    g = pe.groupby("fault_type").agg(n=("event_id", "size"), detected=("detected", "mean"), type_ok=("type_correct", "mean"),
                                     cls_ok=("class_correct", "mean"), tl=("timeliness", "mean"))
    return g.round(2)
