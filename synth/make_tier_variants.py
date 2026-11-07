"""Value-of-instrumentation experiment: the same pumps, faults and operating context at three sensor levels.

Takes the first N assets of a (hold-out) synthetic fleet YAML and writes three copies that differ only in
`instrumentation_tier` (T0_LEGACY / T1_RETROFIT / T2_FULL). Built with the same seed, the simulator's scenario,
context and plant random streams (keyed by seed and asset id) are identical, so the physical truth is the same and
only what the instruments can see changes.
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "tiers"))
    a = ap.parse_args()
    with open(a.src) as fh:
        cfg = yaml.safe_load(fh)
    os.makedirs(a.out, exist_ok=True)
    procs = []
    for tier in ("T0_LEGACY", "T1_RETROFIT", "T2_FULL"):
        c = copy.deepcopy(cfg)
        c["assets"] = c["assets"][: a.n]
        for asset in c["assets"]:
            asset["instrumentation_tier"] = tier
        yml = os.path.join(a.out, f"fleet_{tier}.yaml")
        with open(yml, "w") as fh:
            yaml.safe_dump(c, fh, sort_keys=False, default_flow_style=None, width=200)
        out = os.path.join(a.out, tier)
        log = open(os.path.join(a.out, f"log_{tier}.txt"), "w")
        procs.append(subprocess.Popen([sys.executable, os.path.join(ROOT, "pack", "make_dataset.py"), "--tier", "full",
                                       "--fleet-config", yml, "--seed", str(a.seed), "--output", out], stdout=log, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    print("tier variants built:", [p.returncode for p in procs])
