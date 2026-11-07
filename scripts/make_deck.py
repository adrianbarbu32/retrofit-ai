"""Pitch deck as a 16:9 PDF (matplotlib), numbers read from results/. Also writes one PNG per slide for review.

  python scripts/make_deck.py  ->  docs/RETROFIT_AI_deck.pdf, docs/deck_png/slide-NN.png
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import pandas as pd  # noqa: E402

from retrofit_ai import config  # noqa: E402

W, H = 13.333, 7.5
NAVY, DEEP, TEAL, ICE = "#13234b", "#065a82", "#1c7293", "#e8f1f8"
ORANGE, INK, INK2, MUTED, WHITE, GRIDC = "#eb6834", "#0b0b0b", "#3d3c39", "#6f6e69", "#ffffff", "#e6e5e1"
FIG = os.path.join(config.ROOT, "docs", "figures")
R = config.RESULTS


def load():
    d = {}
    t = pd.read_csv(os.path.join(R, "metrics_table.csv"))
    d["tab"] = t
    g = lambda m, s, c: t[(t.method == m) & (t.split == s)][c].iloc[0]
    d["val"] = {c: g("Ours", "validation", c) for c in t.columns}
    d["b1val"] = {c: g("B1_mode_aware_z", "validation", c) for c in t.columns}
    d["hold"] = {c: g("Ours", "synthetic hold-out", c) for c in t.columns}
    d["b1hold"] = {c: g("B1_mode_aware_z", "synthetic hold-out", c) for c in t.columns}
    with open(os.path.join(R, "metrics_table.md")) as fh:
        m = re.search(r"interval[^:]*: (\d+(?:\.\d+)?) - (\d+(?:\.\d+)?)", fh.read())
        d["ci"] = (m.group(1), m.group(2)) if m else ("-", "-")
    p = os.path.join(R, "sensor_value_overall.csv")
    d["sv"] = pd.read_csv(p) if os.path.exists(p) else None
    p = os.path.join(R, "edge_footprint.json")
    d["edge"] = json.load(open(p)) if os.path.exists(p) else {}
    p = os.path.join(R, "causality_check.json")
    d["caus"] = json.load(open(p)) if os.path.exists(p) else {}
    p = os.path.join(R, "business_case.md")
    d["bc"] = open(p).read() if os.path.exists(p) else ""
    return d


def new(dark=False):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(NAVY if dark else WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    return fig, ax


def title(ax, text, sub=None, dark=False):
    ax.text(0.6, H - 0.85, text, fontsize=30, fontweight="bold", color=WHITE if dark else NAVY, va="top")
    if sub:
        ax.text(0.6, H - 1.62, sub, fontsize=15, color="#c9d6ea" if dark else INK2, va="top")


def footer(ax, n, dark=False):
    ax.text(0.6, 0.3, "RETROFIT AI  |  Since AI Hackathon 2026  |  synthetic benchmark data", fontsize=9, color="#8ea2c2" if dark else MUTED)
    ax.text(W - 0.6, 0.3, str(n), fontsize=9, color="#8ea2c2" if dark else MUTED, ha="right")


def card(ax, x, y, w, h, head, body, fc=ICE, hc=NAVY, bs=13, hs=16):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="none"))
    ax.text(x + 0.25, y + h - 0.3, head, fontsize=hs, fontweight="bold", color=hc, va="top")
    ax.text(x + 0.25, y + h - 0.85, body, fontsize=bs, color=INK2, va="top", linespacing=1.45, wrap=True)


def stat(ax, x, y, big, label, color=DEEP, size=46, lw=3.2):
    ax.text(x, y, big, fontsize=size, fontweight="bold", color=color, va="bottom")
    ax.text(x, y - 0.15, label, fontsize=12.5, color=INK2, va="top", linespacing=1.35)


def image(fig, path, rect):
    if not os.path.exists(path):
        return
    a = fig.add_axes(rect)
    a.imshow(mpimg.imread(path))
    a.axis("off")


def box(ax, x, y, w, h, text, fc, tc=WHITE, fs=12.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, c=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.6, color=c))


def build(out_pdf, png_dir):
    d = load()
    v, b1, ho, b1h = d["val"], d["b1val"], d["hold"], d["b1hold"]
    os.makedirs(png_dir, exist_ok=True)
    slides = []

    # 1 title
    fig, ax = new(dark=True)
    ax.text(0.8, 4.9, "RETROFIT AI", fontsize=54, fontweight="bold", color=WHITE)
    ax.text(0.8, 4.05, "An operator assistant for the pumps you already have", fontsize=24, color="#c9d6ea")
    ax.text(0.8, 3.2, "Early, explainable, listen-only fault detection for existing cooling-water pumps:\n"
            "no new machinery, no new controls, no write path.", fontsize=15, color="#9fb3d1", linespacing=1.5)
    ax.text(0.8, 1.2, "Since AI Hackathon 2026, Turku  |  Challenge: Business Turku  |  Technical sparring: Elomatic",
            fontsize=12, color="#8ea2c2")
    slides.append(fig)

    # 2 problem
    fig, ax = new()
    title(ax, "Old pumps fail quietly; alarms come late",
          "Ships and plants run 10-25-year-old pumps with a handful of signals and fixed alarm limits")
    stat(ax, 0.8, 4.2, "11 %", "of developing problems on the validation\npumps are caught by the plant's own\nalarm list (organizer baseline B0)", ORANGE)
    stat(ax, 5.0, 4.2, "0.44", "fault-type recall of the best simple\nbaseline: half the time it names the\nwrong cause (B1, validation)", ORANGE)
    stat(ax, 9.1, 4.2, "EUR 9 k", "typical unplanned repair vs EUR 4 k\nplanned, plus 10 h vs 3 h stop\n(organizer business-case assumptions)", ORANGE)
    ax.text(0.8, 1.4, "Replacing working pumps or their controls to add AI is costly, so the intelligence has to be retrofitted around them.",
            fontsize=14, color=INK2)
    footer(ax, 2)
    slides.append(fig)

    # 3 architecture
    fig, ax = new()
    title(ax, "Listen, compare with physics, diagnose, advise",
          "Every minute, on an edge box next to the pump; nothing is ever written back")
    y = 3.9
    box(ax, 0.6, y, 2.2, 1.1, "Existing signals\n+ cheap retrofit\nsensors (read-only)", "#5b6b8c")
    box(ax, 3.3, y, 2.3, 1.1, "Healthy-pump twin\n(datasheet physics\n+ live duty)", DEEP)
    box(ax, 6.1, y, 2.2, 1.1, "Residual features\n(trend, step/ramp,\nnoise, frozen, gaps)", DEEP)
    box(ax, 8.8, y, 2.0, 1.1, "13 fault models\n(trained on\nsimulated faults)", DEEP)
    box(ax, 11.2, y, 1.6, 1.1, "Alert\npolicy", DEEP)
    for x0, x1 in ((2.8, 3.3), (5.6, 6.1), (8.3, 8.8), (10.8, 11.2)):
        arrow(ax, x0, y + 0.55, x1, y + 0.55)
    box(ax, 5.6, 1.45, 7.2, 1.55, "Advisory alert: machine or instrument fault, likely cause, confidence,\n"
        "evidence in kPa / % / mm/s, other condition indicated, recommended check\n-> a human approves (console, work-order request)", ORANGE, fs=12.5)
    arrow(ax, 12.0, y, 11.2, 3.0, ORANGE)
    ax.text(0.6, 2.75, "What is new", fontsize=15, fontweight="bold", color=NAVY)
    ax.text(0.6, 2.4, "- reference = physics, not 'the first day of data':\n  faults present at start are visible\n"
            "- diagnosis learned from 500 simulated pump-weeks\n  (organizers' simulator, all 13 fault types)\n"
            "- one alert per problem, never a command", fontsize=12.5, color=INK2, va="top", linespacing=1.45)
    footer(ax, 3)
    slides.append(fig)

    # 4 example evidence
    fig, ax = new()
    title(ax, "Evidence an operator can check", "PUMP-08 (validation, wireless retrofit): suction strainer clogging")
    image(fig, os.path.join(FIG, "fig_evidence_PUMP-08.png"), [0.03, 0.07, 0.62, 0.66])
    card(ax, 8.85, 0.8, 3.95, 4.6, "Alert card",
         "Suction strainer clogging\n(machine fault)\n\nEvidence: suction pressure falls below\nthe healthy model; discharge pressure\n"
         "and power fall with it (less flow)\n\nRecommended (human approves):\nclean / backflush the strainer at the\nnext opportunity\n\n"
         "Severity estimate from physics:\nclog factor ~2.2 -> sandbox: cleaning\nrestores flow and NPSH margin", bs=12)
    footer(ax, 4)
    slides.append(fig)

    # 5 results
    fig, ax = new()
    title(ax, "Measured with the official scorer", "Composite = 35 % F1 + 20 % timeliness + 25 % fault-type + 10 % class + 10 % alert burden")
    image(fig, os.path.join(FIG, "fig_scores.png"), [0.03, 0.30, 0.64, 0.52])
    pc = lambda x: f"{float(x):.0%}"
    stat(ax, 9.1, 4.75, pc(v['Fault-type recall']), f"of validation faults named with the\nright cause (best baseline {pc(b1['Fault-type recall'])})", DEEP, 40)
    stat(ax, 9.1, 2.95, pc(ho['Fault-type recall']), f"on 75 unseen simulated pumps\n(best baseline {pc(b1h['Fault-type recall'])})", DEEP, 40)
    ax.text(0.8, 1.55, f"Validation: false + excess alerts {v['False+excess /asset-day']} per pump-day, timeliness {v['Timeliness']}.  "
            f"Hold-out composite 95 % bootstrap interval {d['ci'][0]}-{d['ci'][1]} (resampling pumps).",
            fontsize=12.5, color=INK2)
    ax.text(0.8, 1.15, "Validation pumps never used for training or tuning; thresholds tuned on separate simulated hold-out pumps.",
            fontsize=12.5, color=INK2)
    footer(ax, 5)
    slides.append(fig)

    # 6 per type
    fig, ax = new()
    title(ax, "It names the right cause, including 'it is the sensor'", "Instrument faults change one reading while the physically coupled readings stay healthy")
    image(fig, os.path.join(FIG, "fig_type_recall.png"), [0.02, 0.06, 0.58, 0.66])
    card(ax, 8.1, 0.8, 4.7, 4.6, "Instrument vs machine",
         "A drifting or frozen gauge can trigger a\nneedless repair or hide a real fault.\n\n"
         "The assistant checks each reading\nagainst the others through the pump\nphysics (curve, affinity laws, motor):\n"
         "- one channel off, peers healthy\n  -> instrument fault, check transmitter\n"
         "- several channels move as the physics\n  predicts -> machine fault\n\n"
         "Step vs ramp shape separates a\ncalibration shift from a drift.", bs=12)
    footer(ax, 6)
    slides.append(fig)

    # 7 retrofit value
    fig, ax = new()
    title(ax, "What each sensor buys, on identical pumps", "Same simulated pumps and faults at three instrumentation levels")
    image(fig, os.path.join(FIG, "fig_sensor_value.png"), [0.03, 0.12, 0.50, 0.64])
    card(ax, 7.4, 0.8, 5.4, 4.6, "Retrofit bill of materials (list prices)",
         "T0 existing signals: edge box only,\n  EUR 500-600 (RevPi / IOT2050)\n"
         "T1 + suction transmitter (~EUR 145) + one\n  wireless vibration point (EUR 300-1,500)\n  -> about EUR 1-2.5 k per pump\n"
         "T2 full (IO-Link vibration, power, flow,\n  PT100): EUR 2.1-3.1 k + flow meter\n\n"
         "Recommendation: start every pump at T0;\nadd the T1 pair on critical pumps without\nredundancy; T2 where energy metering pays.", bs=12)
    footer(ax, 7)
    slides.append(fig)

    # 8 safety
    fig, ax = new()
    title(ax, "Safe by design", "Advisory only; existing alarms, trips and interlocks stay authoritative")
    e, c = d["edge"], d["caus"]
    items = [("No write path", "Reads OPC UA / Modbus / replay;\none-way MQTT-style alerts.\nIEC 62443 zones, IACS E26/E27."),
             ("Human approval", "Acknowledge, reject, request\nwork order; every decision\nlogged for shadow review."),
             ("Sandbox only", "What-if checks run on the\nsimulator with duty, NPSH,\nload and vibration limits."),
             ("Honest uncertainty", "Confidence, other condition\nindicated, low confidence shown\nas 'possible', blind spots stated."),
             ("Causal, audited", f"Truncated re-runs give identical\nalerts: {'PASSED' if c.get('all_identical') else 'see report'}\n({len(c.get('runs', []))} cut points)."),
             ("Edge-sized", f"Median {e.get('latency_ms_median', '-')} ms per run,\n{e.get('peak_memory_mb', '-')} MB peak, offline,\nno GPU, no cloud.")]
    for i, (h_, b_) in enumerate(items):
        x = 0.6 + (i % 3) * 4.15
        y_ = 3.2 if i < 3 else 0.75
        card(ax, x, y_, 3.9, 2.2, h_, b_, bs=12.5)
    footer(ax, 8)
    slides.append(fig)

    # 9 business + pilot
    fig, ax = new()
    title(ax, "From hackathon to pilot", "Value comes from avoided unplanned stops minus nuisance handling, measured in shadow mode")
    bc = d["bc"]
    m_net = re.search(r"Net annual effect[^:]*: (EUR [\d,\-]+)", bc)
    m_pay = re.search(r"Simple payback[^:]*: ([\d.]+ years)", bc)
    stat(ax, 0.8, 4.4, m_net.group(1) if m_net else "-", "net per pump and year in the organizer's\nworked example, with our measured recall\nand false-alert rate (illustrative, synthetic)", DEEP, 40)
    stat(ax, 0.8, 2.15, m_pay.group(1) if m_pay else "-", "simple payback on EUR 4.5 k installed\n(organizer assumptions)", DEEP, 40)
    steps = ["1  Read-only data, 4-6 weeks: calibrate the healthy model", "2  Shadow mode: log alerts, compare with work orders",
             "3  Add one T1 point where blind spots show", "4  Advisory pilot with a named role and accept/reject",
             "5  Scale on measured value; independent review first"]
    for i, s_ in enumerate(steps):
        ax.add_patch(FancyBboxPatch((6.6, 4.85 - i * 0.85), 6.2, 0.65, boxstyle="round,pad=0.02,rounding_size=0.1", fc=ICE, ec="none"))
        ax.text(6.8, 5.17 - i * 0.85, s_, fontsize=13, color=NAVY, va="center")
    footer(ax, 9)
    slides.append(fig)

    # 10 transfer + limits
    fig, ax = new(dark=True)
    title(ax, "Transfers; with honest limits", "Swap the physics layer, keep the rest", dark=True)
    card(ax, 0.6, 0.8, 6.0, 4.6, "Other rotating machines",
         "Fans and blowers: same affinity laws,\nfilter clogging = strainer analogue\n"
         "Compressors: map + surge margin\ninstead of curve + NPSH margin\n"
         "Motors, gearboxes: load model, gear-\nmesh frequency instead of vane pass\n\n"
         "Reused unchanged: features, policy,\nconsole, edge runtime, sensor checks", fc="#1f3566", hc=WHITE, bs=12.5)
    card(ax, 6.9, 0.8, 5.9, 4.6, "Limits",
         "Synthetic data, simulator-to-simulator:\nscores overstate field performance\n"
         "Datasheet defaults need a commissioning\nbaseline on a real pump\n"
         "No vibration sensor -> no bearing alerts\n9 validation events: wide uncertainty\n"
         "Advisory, uncertified, no proven savings", fc="#1f3566", hc=WHITE, bs=12.5)
    for t_ in ax.texts:
        if t_.get_color() == INK2:
            t_.set_color("#d7e0ef")
    footer(ax, 10, dark=True)
    slides.append(fig)

    with PdfPages(out_pdf) as pdf:
        for i, f in enumerate(slides, 1):
            pdf.savefig(f, facecolor=f.get_facecolor())
            f.savefig(os.path.join(png_dir, f"slide-{i:02d}.png"), dpi=70, facecolor=f.get_facecolor())
            plt.close(f)
    print("deck ->", out_pdf, len(slides), "slides")


if __name__ == "__main__":
    build(os.path.join(config.ROOT, "docs", "RETROFIT_AI_deck.pdf"), os.path.join(config.ROOT, "docs", "deck_png"))
