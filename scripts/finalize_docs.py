"""Inject the final measured numbers into README.md and docs/TECHNICAL_NOTE.md (between RESULTS markers), then export
the technical note to PDF. Idempotent: re-run after any model / policy change.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from retrofit_ai import config  # noqa: E402

R = config.RESULTS


def block() -> str:
    t = pd.read_csv(os.path.join(R, "metrics_table.csv"))
    nice = {"Ours": "**RETROFIT AI (ours)**", "B0_scada_alarms": "B0 plant alarms", "B1_mode_aware_z": "B1 mode-aware z",
            "B2_cusum": "B2 CUSUM", "B3_isolation_forest": "B3 isolation forest"}
    cols = ["Composite", "Event F1", "Fault-type recall", "Class recall", "Timeliness", "False+excess /asset-day"]
    out = []
    for split, title in (("validation", "Official validation pumps (PUMP-08/09, 9 events; never used for training or tuning)"),
                         ("synthetic hold-out", "75 unseen synthetic pump-weeks (334 events; used only to tune alert thresholds)")):
        s = t[t.split == split].copy()
        s["_c"] = pd.to_numeric(s["Composite"], errors="coerce")
        s = s.sort_values("_c", ascending=False)
        out.append(f"**{title}**\n")
        out.append("| Method | " + " | ".join(cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for _, r in s.iterrows():
            out.append(f"| {nice.get(r.method, r.method)} | " + " | ".join(str(r[c]) for c in cols) + " |")
        out.append("")
    md = open(os.path.join(R, "metrics_table.md")).read()
    m = re.search(r"interval[^:]*: (\d+(?:\.\d+)?) - (\d+(?:\.\d+)?)", md)
    if m:
        out.append(f"- Hold-out composite, 95 % bootstrap interval over pumps: {m.group(1)}-{m.group(2)}.")
    sv = os.path.join(R, "sensor_value_overall.csv")
    if os.path.exists(sv):
        v = pd.read_csv(sv).set_index("sensor level")
        out.append("- Same simulated pumps and faults at three sensor levels: share of *all* faults caught with the correct "
                   f"cause T0 {v.loc['T0_LEGACY', 'typed_share_all']:.0%} -> T1 {v.loc['T1_RETROFIT', 'typed_share_all']:.0%} "
                   f"-> T2 {v.loc['T2_FULL', 'typed_share_all']:.0%} (`results/sensor_value.md`).")
    c = os.path.join(R, "causality_check.json")
    if os.path.exists(c):
        cc = json.load(open(c))
        out.append(f"- Causality audit (re-runs on truncated data): {'PASSED' if cc['all_identical'] else 'FAILED'} on {len(cc['runs'])} cut points.")
    e = os.path.join(R, "edge_footprint.json")
    if os.path.exists(e):
        ee = json.load(open(e))
        out.append(f"- Edge run on the organizer replay stream: median {ee['latency_ms_median']} ms per diagnosis run, "
                   f"{ee['peak_memory_mb']} MB peak memory (laptop CPU core).")
    return "\n".join(out)


def inject(path, text):
    s = open(path, encoding="utf-8").read()
    s = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda m: f"<!-- RESULTS:START -->\n{text}\n<!-- RESULTS:END -->", s, flags=re.S)
    open(path, "w", encoding="utf-8").write(s)


if __name__ == "__main__":
    b = block()
    inject(os.path.join(config.ROOT, "README.md"), b)
    inject(os.path.join(config.ROOT, "docs", "TECHNICAL_NOTE.md"), b)
    subprocess.run([sys.executable, os.path.join(config.ROOT, "scripts", "md_to_pdf.py"), os.path.join(config.ROOT, "docs", "TECHNICAL_NOTE.md"),
                    os.path.join(config.ROOT, "docs", "TECHNICAL_NOTE.pdf")], check=True)
    print(b)
