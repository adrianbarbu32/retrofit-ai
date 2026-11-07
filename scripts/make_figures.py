"""Static figures for the deck / README (matplotlib, reference data-viz palette, light surface).

  python scripts/make_figures.py      -> docs/figures/*.png
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from retrofit_ai import config  # noqa: E402
from retrofit_ai.data import Dataset  # noqa: E402
from retrofit_ai.twin import load_profiles, twin_for_asset  # noqa: E402

OUT = os.path.join(config.ROOT, "docs", "figures")
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
DEEMPH = "#c9c8c2"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2.0, "lines.solid_capstyle": "round"})


def fig_evidence(aid="PUMP-08", fault="SUCTION_STRAINER_CLOGGING", chans=("suction_pressure_kpa", "discharge_pressure_kpa", "motor_power_kw"),
                 start_h=24, end_h=90, name="fig_evidence_PUMP-08.png", title=None):
    ds = Dataset(config.DATA_FULL, "official")
    fr = ds.asset_frame(aid)
    row = ds.assets[ds.assets.asset_id == aid].iloc[0]
    tw = twin_for_asset(row, fr, (fr["running_fraction"] > 0.5).to_numpy(), load_profiles())
    t0 = pd.Timestamp(row.window_start_utc)
    h = (fr["window_start_utc"] - t0).dt.total_seconds() / 3600
    win = (h >= start_h) & (h <= end_h)
    run = fr["running_fraction"] >= 0.99
    labels = {"suction_pressure_kpa": "Suction pressure [kPa]", "motor_speed_rpm": "Drive speed [rpm]", "motor_power_kw": "Motor power [kW]",
              "vibration_crest_factor": "Crest factor [-]", "vibration_accel_rms_g": "Acceleration [g]", "bearing_temp_c": "Bearing temp [degC]",
              "discharge_pressure_kpa": "Discharge pressure [kPa]", "motor_current_a": "Motor current [A]", "flow_m3_h": "Flow [m3/h]"}
    al = pd.read_csv(os.path.join(config.RESULTS, "console", "alerts_all.csv"))
    al = al[(al.asset_id == aid) & (al.fault_type == fault)]
    b1 = pd.read_csv(os.path.join(config.RESULTS, "baselines", "predictions_B1_mode_aware_z_validation.csv")) \
        if row.split == "validation" else pd.read_csv(os.path.join(config.RESULTS, "baselines", "predictions_B1_mode_aware_z_test.csv"))
    b1 = b1[b1.asset_id == aid]
    ev = ds.events[(ds.events.asset_id == aid) & (ds.events.fault_type == fault)]
    fig, axes = plt.subplots(len(chans), 1, figsize=(9, 2.1 * len(chans) + 0.6), sharex=True)
    for ax, c in zip(axes, chans):
        y = fr[f"{c}__mean"].where(run)
        ax.plot(h[win], y[win], color=BLUE, lw=1.4, label="measured (1-min mean)")
        ax.plot(h[win], tw[c].where(run)[win], color=ORANGE, lw=2.0, ls=(0, (4, 2)), label="healthy-pump model")
        ax.set_ylabel(labels.get(c, c), fontsize=9)
        for e in ev.itertuples():
            a = (pd.Timestamp(e.detectable_utc) - t0).total_seconds() / 3600
            b = (pd.Timestamp(e.end_utc) - t0).total_seconds() / 3600 if isinstance(e.end_utc, str) else 168
            ax.axvspan(max(a, start_h), min(b, end_h), color="#eeeeea", zorder=0, lw=0)
    for ax in axes:
        for a in al.itertuples():
            x = (pd.Timestamp(a.alert_time_utc) - t0).total_seconds() / 3600
            if start_h <= x <= end_h:
                ax.axvline(x, color=INK, lw=1.2)
        for a in b1.itertuples():
            x = (pd.Timestamp(a.alert_time_utc) - t0).total_seconds() / 3600
            if start_h <= x <= end_h:
                ax.axvline(x, color=MUTED, lw=1.0, ls=(0, (2, 2)))
    top = axes[0]
    for a in al.itertuples():
        x = (pd.Timestamp(a.alert_time_utc) - t0).total_seconds() / 3600
        if start_h <= x <= end_h:
            top.annotate(f"RETROFIT AI alert: {a.fault_type.replace('_', ' ').lower()} ({a.confidence:.0%})", (x, 1.02), xycoords=("data", "axes fraction"),
                         fontsize=8.5, color=INK, ha="left", va="bottom")
            break
    for e in ev.itertuples():
        if isinstance(e.action_deadline_utc, str):
            x = (pd.Timestamp(e.action_deadline_utc) - t0).total_seconds() / 3600
            if start_h <= x <= end_h:
                for ax in axes:
                    ax.axvline(x, color=INK2, lw=1.0, ls=(0, (1, 2)))
                axes[-1].annotate("conventional limit reached", (x, -0.32), xycoords=("data", "axes fraction"), fontsize=8, color=INK2, ha="center")
    axes[0].legend(loc="lower left", fontsize=8.5, ncol=2)
    axes[-1].set_xlabel("hours since start of the 7-day window")
    fig.suptitle(title or f"{aid}: evidence behind the alert\nshaded = labelled fault window (ground truth), solid black = our alert, dotted grey = baseline B1 alerts",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=180)
    plt.close(fig)


def fig_scores():
    tab = pd.read_csv(os.path.join(config.RESULTS, "metrics_table.csv"))
    nice = {"Ours": "RETROFIT AI (ours)", "B0_scada_alarms": "B0 plant alarms", "B1_mode_aware_z": "B1 mode-aware z",
            "B2_cusum": "B2 CUSUM", "B3_isolation_forest": "B3 isolation forest"}
    splits = [("validation", "Official validation pumps (9 events)"), ("synthetic hold-out", "Unseen synthetic pumps (75 pump-weeks)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (sp, title) in zip(axes, splits):
        t = tab[tab.split == sp].copy()
        t["Composite"] = pd.to_numeric(t["Composite"], errors="coerce")
        t = t.sort_values("Composite")
        cols = [BLUE if m == "Ours" else DEEMPH for m in t.method]
        ax.barh([nice.get(m, m) for m in t.method], t.Composite, color=cols, height=0.55)
        for yv, v in enumerate(t.Composite):
            ax.text(v + 1.2, yv, f"{v:.1f}", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, 108)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("composite score (official scorer, 0-100)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_scores.png"), dpi=180)
    plt.close(fig)


def fig_type_recall():
    p = os.path.join(config.RESULTS, "per_type_holdout.csv")
    if not os.path.exists(p):
        return
    t = pd.read_csv(p).sort_values("ours_type_ok")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    yy = np.arange(len(t))
    for y0, a, b in zip(yy, t.b1_type_ok, t.ours_type_ok):
        ax.plot([a, b], [y0, y0], color=GRID, lw=2, zorder=1)
    ax.scatter(t.b1_type_ok, yy, s=60, color=ORANGE, zorder=2, label="B1 baseline", edgecolors=SURF, linewidths=2)
    ax.scatter(t.ours_type_ok, yy, s=60, color=BLUE, zorder=3, label="RETROFIT AI", edgecolors=SURF, linewidths=2)
    ax.set_yticks(yy, [f"{k.replace('_', ' ').lower()} ({n})" for k, n in zip(t.fault_type, t.events)], fontsize=9)
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("share of events detected with the correct fault type\n(75 unseen synthetic pumps; number of events in brackets)")
    ax.set_title("Naming the right cause, per fault type")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_type_recall.png"), dpi=180)
    plt.close(fig)


def fig_sensor_value():
    p = os.path.join(config.RESULTS, "sensor_value_overall.csv")
    if not os.path.exists(p):
        return
    t = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = np.arange(len(t))
    ax.bar(x, t["typed_share_all"], width=0.45, color=BLUE)
    for xv, v in zip(x, t["typed_share_all"]):
        ax.text(xv, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x, [{"T0_LEGACY": "T0 existing signals", "T1_RETROFIT": "T1 + suction pressure\n+ wireless vibration", "T2_FULL": "T2 fully instrumented"}[k]
                      for k in t["sensor level"]], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("faults caught with correct cause")
    ax.set_title("Same pumps, same faults: what each sensor level buys")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sensor_value.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_evidence()
    fig_evidence("PUMP-04", "BEARING_WEAR", ("vibration_crest_factor", "vibration_accel_rms_g", "bearing_temp_c"), 0, 60,
                 "fig_evidence_PUMP-04.png", "PUMP-04: bearing damage present from the start, visible against the healthy model")
    fig_scores()
    fig_type_recall()
    fig_sensor_value()
    print("figures ->", OUT, sorted(os.listdir(OUT)))
