"""Build the measured-performance report: our engine vs. the organizer baselines, per-type breakdown, bootstrap CIs,
value of each instrumentation tier, and the business case with our measured recall / false-alert rate.

Usage: python scripts/report.py --holdout b1013-b1015 [--tiers build/tiers build/tiers_b1013 build/tiers_b1014]
Outputs (results/): metrics_table.md/.csv, per_type_holdout.md, sensor_value.md, business_case.md, report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from retrofit_ai import config  # noqa: E402
from retrofit_ai.data import Dataset  # noqa: E402
from retrofit_ai.evaluate import score_alerts  # noqa: E402
from retrofit_ai.model import load_models, predict_asset  # noqa: E402
from retrofit_ai.policy import run_policy  # noqa: E402
from scripts.baselines_on import run as run_baselines  # noqa: E402
from scripts.predict import alerts_with_explanations, load_policy  # noqa: E402
from scripts.train_eval import expand  # noqa: E402

KEYS = [("composite_score", "Composite", 1), ("event_f1", "Event F1", 2), ("event_recall", "Recall", 2),
        ("alert_precision", "Precision", 2), ("false_alerts_per_asset_day", "False+excess /asset-day", 2),
        ("timeliness", "Timeliness", 2), ("fault_type_recall", "Fault-type recall", 2), ("class_recall", "Class recall", 2),
        ("median_lead_time_h", "Median lead to deadline [h]", 1)]


def fmt(v, nd):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def ours_on(dirs_names, models, feats, params):
    dss, frames = [], []
    for d, name in dirs_names:
        ds = Dataset(d, name)
        dss.append(ds)
        for p in sorted(glob.glob(os.path.join(config.BUILD, "features", name, "*.parquet"))):
            frames.append(predict_asset(pd.read_parquet(p), models, feats))
    al = run_policy(frames, params)
    ev = pd.concat([d.events for d in dss], ignore_index=True)
    assets = pd.concat([d.assets for d in dss], ignore_index=True)
    bn = pd.concat([d.benign for d in dss], ignore_index=True)
    return score_alerts(al, ev, assets, bn), al, ev, assets, bn


def bootstrap(al, ev, assets, bn, n=200, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(assets.asset_id))
    out = []
    for _ in range(n):
        pick = rng.choice(ids, len(ids), replace=True)
        # duplicate-safe resampling: rename repeated assets
        a_parts, e_parts, l_parts, b_parts = [], [], [], []
        for k, aid in enumerate(pick):
            new = f"{aid}#{k}"
            a_parts.append(assets[assets.asset_id == aid].assign(asset_id=new))
            e_parts.append(ev[ev.asset_id == aid].assign(asset_id=new, event_id=lambda d: d.event_id + f"#{k}"))
            l_parts.append(al[al.asset_id == aid].assign(asset_id=new))
            b_parts.append(bn[bn.asset_id == aid].assign(asset_id=new))
        r = score_alerts(pd.concat(l_parts), pd.concat(e_parts), pd.concat(a_parts), pd.concat(b_parts))
        out.append(r["composite_score"])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def per_type(r_ours, r_b1):
    a = pd.DataFrame(r_ours["per_event"]).groupby("fault_type").agg(events=("event_id", "size"), ours_detected=("detected", "mean"),
                                                                     ours_type_ok=("type_correct", "mean"))
    b = pd.DataFrame(r_b1["per_event"]).groupby("fault_type").agg(b1_detected=("detected", "mean"), b1_type_ok=("type_correct", "mean"))
    return a.join(b).round(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="b1013-b1015")
    ap.add_argument("--tiers", nargs="*", default=["build/tiers", "build/tiers_b1013", "build/tiers_b1014"])
    ap.add_argument("--boot", type=int, default=100)
    a = ap.parse_args()
    models, feats = load_models()
    params = load_policy()
    off = Dataset(config.DATA_FULL, "official")
    rep = {}
    # official splits: ours (validation via the full explained pipeline)
    for split in ("train", "validation"):
        ids = list(off.assets[off.assets.split == split].asset_id)
        al = alerts_with_explanations(off, ids, models, feats, params)
        rep[("Ours", split)] = score_alerts(al, off.events, off.assets, off.benign, ids)
    base = json.load(open(os.path.join(config.RESULTS, "baselines", "baseline_report.json")))["results"]
    for b in ("B0_scada_alarms", "B1_mode_aware_z", "B2_cusum", "B3_isolation_forest"):
        for split in ("train", "validation"):
            rep[(b, split)] = base[f"{b}|{split}"]
    # synthetic hold-out (never used for fitting; used to tune the policy thresholds)
    hold = [(os.path.join(config.BUILD, "synth", n), n) for n in expand(a.holdout)]
    r, al, ev, assets, bn = ours_on(hold, models, feats, params)
    rep[("Ours", "synthetic hold-out")] = r
    lo, hi = bootstrap(al, ev, assets, bn, n=a.boot) if a.boot else (np.nan, np.nan)
    bres = run_baselines([d for d, _ in hold])
    for k, v in bres.items():
        rep[(k, "synthetic hold-out")] = v
    rows = []
    for (m, split), v in rep.items():
        rows.append({"method": m, "split": split, **{lab: fmt(v.get(k), nd) for k, lab, nd in KEYS},
                     "events": v.get("events_scored")})
    tab = pd.DataFrame(rows)
    order = {"train": 0, "validation": 1, "synthetic hold-out": 2}
    tab = tab.sort_values(["split", "method"], key=lambda s: s.map(order) if s.name == "split" else s).reset_index(drop=True)
    os.makedirs(config.RESULTS, exist_ok=True)
    tab.to_csv(os.path.join(config.RESULTS, "metrics_table.csv"), index=False)
    with open(os.path.join(config.RESULTS, "metrics_table.md"), "w") as fh:
        fh.write("Official scorer (`retrofit_sim.scoring`). *Ours* on the official train split is in-sample (the 7 train pumps are part "
                 "of the training data, together with 500 synthetic pump-weeks). Validation pumps were never used for fitting or tuning. "
                 "The synthetic hold-out (75 pump-weeks, unseen seeds and pumps) was used only to tune alert thresholds. "
                 f"Bootstrap 95 % interval of our hold-out composite (resampling pumps): {lo:.1f} - {hi:.1f}.\n\n")
        fh.write(tab.to_markdown(index=False))
    pt = per_type(rep[("Ours", "synthetic hold-out")], rep[("B1_mode_aware_z", "synthetic hold-out")])
    pt.reset_index().to_csv(os.path.join(config.RESULTS, "per_type_holdout.csv"), index=False)
    with open(os.path.join(config.RESULTS, "per_type_holdout.md"), "w") as fh:
        fh.write("Per fault type on the synthetic hold-out (share of events detected / detected with the correct fault type).\n\n")
        fh.write(pt.to_markdown())
    print(tab.to_string(index=False))
    print(pt)
    # value of instrumentation: same pumps and faults at three sensor levels
    tier_rows, tier_type = [], []
    for tier in ("T0_LEGACY", "T1_RETROFIT", "T2_FULL"):
        dn = [(os.path.join(t, tier), f"{tier}_{os.path.basename(os.path.normpath(t))}") for t in a.tiers
              if os.path.exists(os.path.join(t, tier, "public", "features_1min.parquet"))]
        dn = [(d, n) for d, n in dn if glob.glob(os.path.join(config.BUILD, "features", n, "*.parquet"))]
        if not dn:
            continue
        rt, *_ = ours_on(dn, models, feats, params)
        rb = run_baselines([d for d, _ in dn])["B1_mode_aware_z"]
        tier_rows.append({"sensor level": tier, "pumps": len(rt["assets"]), "events (observable)": rt["events_scored"],
                          "unobservable (not scored)": rt["events_unobservable_excluded"], "composite": round(rt["composite_score"], 1),
                          "recall": round(rt["event_recall"], 2), "fault-type recall": round(rt["fault_type_recall"], 2),
                          "false+excess /asset-day": round(rt["false_alerts_per_asset_day"], 2), "B1 composite": round(rb["composite_score"], 1)})
        pe = pd.DataFrame(rt["per_event"])
        evall = pd.concat([Dataset(d, n).events for d, n in dn])
        cnt_all = evall.groupby("fault_type").size()
        g = pe.groupby("fault_type").agg(detected=("detected", "sum"), typed=("type_correct", "sum"))
        g = g.reindex(cnt_all.index).fillna(0)
        g["all"] = cnt_all
        g["tier"] = tier
        tier_type.append(g.reset_index())
    if tier_rows:
        tt = pd.concat(tier_type)
        ov = tt.groupby("tier").agg(typed=("typed", "sum"), detected=("detected", "sum"), all=("all", "sum")).reset_index()
        ov["typed_share_all"] = (ov.typed / ov["all"]).round(3)
        ov["detected_share_all"] = (ov.detected / ov["all"]).round(3)
        ov = ov.rename(columns={"tier": "sensor level"})
        ov.to_csv(os.path.join(config.RESULTS, "sensor_value_overall.csv"), index=False)
        piv = tt.assign(share=lambda d: (d.typed / d["all"]).round(2)).pivot(index="fault_type", columns="tier", values="share")
        with open(os.path.join(config.RESULTS, "sensor_value.md"), "w") as fh:
            fh.write("Same pumps, same faults, same operating context; only the instrumentation differs (simulator rebuilt with "
                     "identical seeds). Composite counts only events the instruments can reveal, so the share of all faults caught "
                     "with the correct type (table 2) is the fairer comparison of what each sensor level buys.\n\n")
            fh.write(pd.DataFrame(tier_rows).to_markdown(index=False))
            fh.write("\n\nShare of ALL simulated faults (observable or not) detected with the correct fault type:\n\n")
            fh.write(piv.to_markdown())
        print(pd.DataFrame(tier_rows).to_string(index=False))
        print(piv)
    # business case with our measured validation recall and false-alert rate
    rv = rep[("Ours", "validation")]
    out = os.path.join(config.RESULTS, "business_case.md")
    subprocess.run([sys.executable, os.path.join(config.PACK, "examples", "business_case.py"), "--recall", f"{rv['event_recall']:.3f}",
                    "--fa-per-day", f"{rv['false_alerts_per_asset_day']:.3f}", "--out", out], check=False, capture_output=True)
    json.dump({f"{m}|{s}": {k: v.get(k) for k, _, _ in KEYS} for (m, s), v in rep.items()},
              open(os.path.join(config.RESULTS, "report.json"), "w"), indent=1, default=float)
    print("report written to results/")
