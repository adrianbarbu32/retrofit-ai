"""Build many synthetic fleets in parallel with the organizer's dataset writer.

Usage: python synth/run_factory.py --batches 16 --per-batch 25 --workers 7 --first-seed 1000
Output: build/synth/b<seed>/{public,private}/...  (one directory per batch)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def run_batch(seed, n, out_root, days, snapshots, focus=False):
    out = os.path.join(out_root, f"b{seed}")
    if os.path.exists(os.path.join(out, "private", "events_stress_all.csv")) or os.path.exists(os.path.join(out, "public", "events_train.csv")):
        return seed, 0.0, "exists"
    os.makedirs(out_root, exist_ok=True)
    yml = os.path.join(out_root, f"fleet_b{seed}.yaml")
    subprocess.run([PY, os.path.join(HERE, "sample_fleet.py"), "--out", yml, "--seed", str(seed), "--n", str(n),
                    "--prefix", f"SYN{seed}"] + (["--focus"] if focus else []), check=True, capture_output=True)
    t0 = time.time()
    cmd = [PY, os.path.join(ROOT, "pack", "make_dataset.py"), "--tier", "full", "--fleet-config", yml, "--seed", str(seed),
           "--days", str(days), "--output", out]
    if not snapshots:
        cmd.append("--no-snapshots")
    with open(os.path.join(out_root, f"log_b{seed}.txt"), "w") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return seed, time.time() - t0, "ok" if p.returncode == 0 else f"rc={p.returncode}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--per-batch", type=int, default=25)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "synth"))
    ap.add_argument("--no-snapshots", action="store_true")
    ap.add_argument("--focus", action="store_true", help="oversample the control/sensor combinations rare in the official train split")
    a = ap.parse_args()
    seeds = [a.first_seed + i for i in range(a.batches)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for seed, dt, status in ex.map(lambda s: run_batch(s, a.per_batch, a.out, a.days, not a.no_snapshots, a.focus), seeds):
            print(f"batch {seed}: {status} in {dt:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    print("factory done", flush=True)
