"""Random synthetic fleet sampler for the RETROFIT AI simulator (training-data factory).

Writes fleet YAML files in the format of pack/configs/fleet_public.yaml. Each file is then built with
`pack/make_dataset.py --tier full --fleet-config <yaml> --seed <seed>`. Every generated event is labelled by the
organizer's own simulator, including detectability, deadlines and emergent cavitation.

Differences from the organizer's stress tier (src/retrofit_sim/stress.py):
  * full fault taxonomy: impeller wear, air entrainment (heavy weather / low basin level), pre-existing bearing wear;
  * the compound incidents documented in TECHNICAL_MODEL.md section 9 (bearing replacement -> misalignment, planned
    seal work -> misalignment, restriction repair -> transmitter calibration offset, clogging -> cavitation, low
    basin level -> air entrainment, drifting suction transmitter masking a clogging strainer);
  * every profile / control / instrumentation-tier combination, weighted towards the control types in the fleet;
  * pump ratings sampled from the documented ranges and around the public datasheets in assets.csv (static data only;
    no telemetry of any asset is used).
"""
from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PUBLIC_YAML = os.path.join(ROOT, "pack", "configs", "fleet_public.yaml")
ASSETS_CSV = os.path.join(ROOT, "pack", "data", "full", "assets.csv")

# (profile, control, weight, tier probabilities T0/T1/T2)
COMBOS = [
    ("SHIP_SW_COOLING", "VFD_TEMPERATURE", 0.28, (0.20, 0.50, 0.30)),
    ("INDUSTRIAL_CW", "VFD_PRESSURE", 0.24, (0.45, 0.20, 0.35)),
    ("PROCESS_TRANSFER", "FLOW_VALVE", 0.24, (0.15, 0.35, 0.50)),
    ("SHIP_LT_FRESHWATER", "FIXED_SPEED", 0.09, (0.40, 0.30, 0.30)),
    ("INDUSTRIAL_CW", "FIXED_SPEED", 0.10, (0.40, 0.25, 0.35)),
    ("SHIP_SW_COOLING", "FIXED_SPEED", 0.05, (0.40, 0.30, 0.30)),
]
# "focus" mix: more pumps with the control / sensor combinations that are rare in the official training split
COMBOS_FOCUS = [
    ("SHIP_SW_COOLING", "VFD_TEMPERATURE", 0.40, (0.15, 0.60, 0.25)),
    ("INDUSTRIAL_CW", "VFD_PRESSURE", 0.40, (0.60, 0.15, 0.25)),
    ("PROCESS_TRANSFER", "FLOW_VALVE", 0.14, (0.15, 0.25, 0.60)),
    ("SHIP_LT_FRESHWATER", "FIXED_SPEED", 0.03, (0.40, 0.30, 0.30)),
    ("INDUSTRIAL_CW", "FIXED_SPEED", 0.03, (0.40, 0.25, 0.35)),
]
TIERS = ["T0_LEGACY", "T1_RETROFIT", "T2_FULL"]
RANGES = {  # q_bep m3/h, h_bep m, rpm options, eta range, npsh3 range
    "SHIP_SW_COOLING": ((300, 750), (20, 30), (1780, 1785), (0.80, 0.85), (3.6, 4.8)),
    "SHIP_LT_FRESHWATER": ((250, 500), (28, 38), (1775, 1780), (0.79, 0.84), (3.0, 4.0)),
    "INDUSTRIAL_CW": ((150, 450), (25, 40), (1480, 1485, 2955), (0.76, 0.84), (2.8, 4.0)),
    "PROCESS_TRANSFER": ((30, 80), (35, 50), (2930, 2935, 2950), (0.65, 0.74), (2.6, 3.4)),
}
MOTOR_STEPS = [5.5, 7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90]
# motor efficiency by size, as in the public fleet configuration (IE3-like)
EFF_TABLE = ([5.5, 11, 15, 30, 45, 55, 75, 90], [0.89, 0.91, 0.92, 0.936, 0.945, 0.95, 0.955, 0.957])
TIER_CH = {
    "T0_LEGACY": ["motor_current_a", "discharge_pressure_kpa"],
    "T1_RETROFIT": ["motor_current_a", "discharge_pressure_kpa", "suction_pressure_kpa", "bearing_temp_c"],
    "T2_FULL": ["motor_current_a", "discharge_pressure_kpa", "suction_pressure_kpa", "bearing_temp_c", "flow_m3_h"],
}
H = 168.0  # window length [h]


def _r(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _ratings(rng, prof, datasheets):
    """Ratings: 40 % jittered public datasheet of the same profile, 60 % uniform in the documented range."""
    same = [d for d in datasheets if d["profile"] == prof]
    if same and rng.random() < 0.4:
        d = same[int(rng.integers(0, len(same)))]
        j = lambda x, s: float(x) * float(rng.uniform(1 - s, 1 + s))
        q = round(j(d["pump_rated_flow_m3_h"], 0.12), 0)
        h = round(j(d["pump_rated_head_m"], 0.08), 1)
        rpm = float(d["motor_rated_rpm"])
        eta = round(min(0.86, max(0.6, j(d["pump_bep_efficiency"], 0.03))), 3)
        npsh = round(j(d["npsh3_at_bep_m"], 0.1), 1)
    else:
        (q0, q1), (h0, h1), rpms, (e0, e1), (n0, n1) = RANGES[prof]
        q = float(np.round(rng.uniform(q0, q1), -1 if q1 > 100 else 0))
        h = float(np.round(rng.uniform(h0, h1), 1))
        rpm = float(rng.choice(rpms))
        eta = float(np.round(rng.uniform(e0, e1), 3))
        npsh = float(np.round(rng.uniform(n0, n1), 1))
    return q, h, rpm, eta, npsh


def _sensor_entry(rng, typ, tag, onset, scale):
    e = {"type": typ, "sensor_tag": tag, "onset_h": round(onset, 1)}
    q_bep, i_rated = scale
    unit = {"discharge_pressure_kpa": 1.0, "suction_pressure_kpa": 1.0, "bearing_temp_c": 1.0,
            "motor_current_a": i_rated / 100.0, "flow_m3_h": q_bep / 100.0}[tag]
    sgn = float(rng.choice([-1.0, 1.0]))
    if typ == "SENSOR_DRIFT":
        rate = {"discharge_pressure_kpa": (0.12, 0.6), "suction_pressure_kpa": (0.12, 0.5), "bearing_temp_c": (0.1, 0.4),
                "motor_current_a": (0.08, 0.35), "flow_m3_h": (0.12, 0.5)}[tag]
        e["params"] = {"drift_per_h": round(sgn * _r(rng, *rate) * unit, 4)}
        if rng.random() < 0.6:
            e["end_h"] = round(min(H - 1, onset + _r(rng, 25, 100)), 1)
    elif typ == "SENSOR_CALIBRATION_SHIFT":
        step = {"discharge_pressure_kpa": (7, 26), "suction_pressure_kpa": (6, 20), "bearing_temp_c": (3, 8),
                "motor_current_a": (3, 8), "flow_m3_h": (5, 12)}[tag]
        e["params"] = {"step": round(sgn * _r(rng, *step) * unit, 3)}
        if rng.random() < 0.6:
            e["end_h"] = round(min(H - 1, onset + _r(rng, 15, 60)), 1)
    elif typ == "SENSOR_STUCK":
        e["end_h"] = round(min(H - 1, onset + _r(rng, 2, 26)), 1)
    elif typ == "SENSOR_NOISE_SPIKES":
        size = {"discharge_pressure_kpa": (25, 90), "suction_pressure_kpa": (20, 60), "bearing_temp_c": (5, 15),
                "motor_current_a": (8, 20), "flow_m3_h": (10, 25)}[tag]
        e["end_h"] = round(min(H - 1, onset + _r(rng, 2, 14)), 1)
        e["params"] = {"noise_mult": round(_r(rng, 3, 6), 2), "spike_rate": round(_r(rng, 0.001, 0.005), 4),
                       "spike_size": round(_r(rng, *size) * unit, 2)}
    e["jitter_h"] = round(_r(rng, 0, 4), 1)
    return e


def sample_scenario(rng, prof, ctrl, tier, scale):
    """3-7 scenario entries mixing machine faults, instrument faults and compound incidents."""
    tags = list(TIER_CH[tier])
    E = []

    def add(e):
        E.append(e)
        return len(E)            # 1-based index, used for root_cause

    used = set()
    n_mach = int(rng.choice([1, 2, 2, 3, 3]))
    w = {"BEARING_WEAR": 0.17, "SUCTION_STRAINER_CLOGGING": 0.21, "DISCHARGE_RESTRICTION": 0.16, "MISALIGNMENT": 0.13,
         "IMBALANCE": 0.10, "IMPELLER_WEAR": 0.11, "AIR_ENTRAINMENT": 0.12}
    if prof == "PROCESS_TRANSFER":
        w["SUCTION_STRAINER_CLOGGING"] = 0.30
    names, p = list(w), np.array(list(w.values()))
    picks = list(rng.choice(names, n_mach, replace=False, p=p / p.sum()))
    ctx_override = {}
    clog_entries = []
    for typ in picks:
        typ = str(typ)
        onset = _r(rng, 3, 140)
        if typ == "BEARING_WEAR":
            pre = rng.random() < 0.45
            onset = _r(rng, -150, -20) if pre else _r(rng, 0, 100)
            t_fail = _r(rng, 120, 420)
            e = {"type": typ, "onset_h": round(onset, 1), "params": {"rate_per_h": round(_r(rng, 0.006, 0.014), 4),
                                                                       "t_fail_h": round(t_fail, 1)},
                 "jitter_h": round(_r(rng, 0, 4), 1)}
            fail_h = onset + t_fail
            if fail_h < H - 6 and rng.random() < 0.85:
                e["end_h"] = round(max(onset + 30, fail_h - _r(rng, 2, 25)), 1)
                e["repair_h"] = round(_r(rng, 5, 8), 1)
            elif rng.random() < 0.3:
                e["end_h"] = round(min(H - 10, max(8.0, onset + _r(rng, 60, 150))), 1)
                e["repair_h"] = round(_r(rng, 5, 8), 1)
            k = add(e)
            if e.get("end_h") is not None and e["end_h"] < H - 12 and rng.random() < 0.5:
                add({"type": "MISALIGNMENT", "onset_h": round(e["end_h"] + e["repair_h"], 1), "root_cause": k, "jitter_h": 0,
                     "params": {"level": round(_r(rng, 0.3, 0.65), 3), "growth_per_h": round(_r(rng, 0, 0.004), 4)}})
        elif typ == "SUCTION_STRAINER_CLOGGING":
            cmax = _r(rng, 2.0, 6.5) if prof != "PROCESS_TRANSFER" else _r(rng, 3.0, 7.0)
            e = {"type": typ, "onset_h": round(onset, 1), "params": {"rate_per_h": round(_r(rng, 0.03, 0.12), 4),
                                                                       "c_max": round(cmax, 2), "shape": "linear"}}
            if rng.random() < 0.6:
                e["end_h"] = round(min(H - 2, onset + _r(rng, 25, 65)), 1)
                e["repair_h"] = round(_r(rng, 1.0, 2.5), 1)
            k = add(e)
            clog_entries.append((k, e))
            if e.get("end_h") is not None and e["end_h"] < H - 40 and rng.random() < 0.25:
                o2 = e["end_h"] + _r(rng, 10, 40)
                if o2 < H - 10:
                    add({"type": typ, "onset_h": round(o2, 1), "params": {"rate_per_h": round(_r(rng, 0.03, 0.1), 4),
                                                                           "c_max": round(_r(rng, 2.0, 5.0), 2), "shape": "linear"}})
        elif typ == "DISCHARGE_RESTRICTION":
            e = {"type": typ, "onset_h": round(onset, 1),
                 "params": {"rate_per_h": float(rng.choice([0.03, 0.06, 0.1, 0.5, 1.5, 3.0])), "extra_max": round(_r(rng, 0.4, 1.5), 3),
                            "shape": "exp"}}
            if rng.random() < 0.6:
                e["end_h"] = round(min(H - 2, onset + _r(rng, 15, 60)), 1)
                e["repair_h"] = round(_r(rng, 1.0, 4.0), 1)
            k = add(e)
            if e.get("end_h") is not None and e["end_h"] < H - 15 and rng.random() < 0.35:
                add({"type": "SENSOR_CALIBRATION_SHIFT", "sensor_tag": "discharge_pressure_kpa",
                     "onset_h": round(e["end_h"] + e["repair_h"] + 0.5, 1), "root_cause": k, "jitter_h": 0,
                     "end_h": round(min(H - 1, e["end_h"] + e["repair_h"] + _r(rng, 20, 60)), 1),
                     "params": {"step": round(float(rng.choice([-1, 1])) * _r(rng, 8, 25), 2)}})
                used.add("SENSOR_CALIBRATION_SHIFT")
        elif typ == "MISALIGNMENT":
            if rng.random() < 0.45:
                k = add({"type": "PLANNED_MAINTENANCE", "onset_h": round(onset, 1), "repair_h": round(_r(rng, 1.5, 4.0), 1),
                         "jitter_h": 1, "params": {"work_type": str(rng.choice(["MECHANICAL_SEAL_REPLACEMENT", "COUPLING_REPLACEMENT",
                                                                                 "MOTOR_SERVICE"])),
                                                   "note": "Planned work completed. Coupling refitted, pump restarted."}})
                add({"type": "MISALIGNMENT", "onset_h": round(onset, 1), "root_cause": k, "jitter_h": 0,
                     "params": {"level": round(_r(rng, 0.3, 0.7), 3), "growth_per_h": round(_r(rng, 0, 0.004), 4)}})
            else:
                e = {"type": typ, "onset_h": round(onset, 1),
                     "params": {"level": round(_r(rng, 0.3, 0.7), 3), "growth_per_h": round(_r(rng, 0, 0.004), 4)}}
                if rng.random() < 0.5:
                    e["end_h"] = round(min(H - 2, onset + _r(rng, 20, 70)), 1)
                    e["repair_h"] = round(_r(rng, 2.0, 4.0), 1)
                add(e)
        elif typ == "IMBALANCE":
            add({"type": typ, "onset_h": round(onset, 1), "params": {"rate_per_h": round(_r(rng, 0.005, 0.015), 4),
                                                                       "u_max": round(_r(rng, 0.6, 1.5), 3), "shape": "linear"}})
        elif typ == "IMPELLER_WEAR":
            if rng.random() < 0.8:
                add({"type": typ, "onset_h": round(_r(rng, -600, -100), 1), "jitter_h": 0,
                     "params": {"rate_per_h": round(_r(rng, 0.00004, 0.00008), 6), "iw0": round(_r(rng, 0.02, 0.08), 3),
                                "iw_max": 0.12}})
            else:
                add({"type": typ, "onset_h": round(_r(rng, 0, 80), 1), "jitter_h": 0,
                     "params": {"rate_per_h": round(_r(rng, 0.0004, 0.0012), 5), "iw0": 0.0, "iw_max": round(_r(rng, 0.06, 0.12), 3)}})
        elif typ == "AIR_ENTRAINMENT":
            dur = _r(rng, 3, 9)
            if prof.startswith("SHIP"):
                ctx_override = {"storm_at_h": round(max(0.0, onset - _r(rng, 2, 6)), 1)}
                add({"type": typ, "onset_h": round(onset, 1), "end_h": round(min(H - 1, onset + dur), 1),
                     "params": {"alpha_max": round(_r(rng, 0.015, 0.035), 4)}, "jitter_h": 1})
            else:
                k = add({"type": "SUCTION_LEVEL_DROP", "onset_h": round(onset, 1), "end_h": round(min(H - 1, onset + dur), 1),
                         "params": {"drop_m": round(_r(rng, 1.4, 2.3), 2), "ramp_h": round(_r(rng, 0.8, 2.0), 2)}, "jitter_h": 3})
                add({"type": typ, "onset_h": round(onset + _r(rng, 0.5, 1.5), 1), "end_h": round(min(H - 1, onset + dur), 1),
                     "root_cause": k, "align": "onset", "params": {"alpha_max": round(_r(rng, 0.018, 0.032), 4)}})
    # instrument faults (on channels this asset has), plus the documented masking combination
    if clog_entries and "suction_pressure_kpa" in tags and rng.random() < 0.3:
        k, ce = clog_entries[0]
        add({"type": "SENSOR_DRIFT", "sensor_tag": "suction_pressure_kpa", "onset_h": round(max(1.0, ce["onset_h"] + _r(rng, -3, 3)), 1),
             "params": {"drift_per_h": round(_r(rng, 0.15, 0.6), 3)}, "jitter_h": 0,
             **({"end_h": round(min(H - 1, ce["onset_h"] + _r(rng, 40, 90)), 1)} if rng.random() < 0.5 else {})})
        used.add("SENSOR_DRIFT")
    n_sens = int(rng.choice([1, 1, 2, 2, 3]))
    menu = ["SENSOR_DRIFT", "SENSOR_STUCK", "SENSOR_CALIBRATION_SHIFT", "SENSOR_NOISE_SPIKES", "COMMUNICATION_LOSS"]
    pw = np.array([0.22, 0.2, 0.2, 0.18, 0.2])
    for typ in rng.choice(menu, n_sens, replace=False, p=pw / pw.sum()):
        typ = str(typ)
        if typ in used:
            continue
        onset = _r(rng, 2, 150)
        if typ == "COMMUNICATION_LOSS":
            add({"type": typ, "sensor_tag": "GATEWAY", "onset_h": round(onset, 1), "end_h": round(min(H - 1, onset + _r(rng, 0.4, 4.0)), 1),
                 "jitter_h": round(_r(rng, 0, 5), 1)})
            continue
        tag = str(rng.choice(tags))
        add(_sensor_entry(rng, typ, tag, onset, scale))
    return E, ctx_override


def sample_asset(rng, k, prefix, datasheets, combos=None):
    combos = combos or COMBOS
    combo = combos[int(rng.choice(len(combos), p=np.array([c[2] for c in combos]) / sum(c[2] for c in combos)))]
    prof, ctrl, _, tp = combo
    tier = str(rng.choice(TIERS, p=tp))
    q, h, rpm, eta, npsh = _ratings(rng, prof, datasheets)
    p_shaft = 1000 * 9.81 * q / 3600 * h / eta / 1000
    motor = next(m for m in MOTOR_STEPS if m >= 1.12 * p_shaft)
    eff = float(np.interp(motor, *EFF_TABLE)) + _r(rng, -0.004, 0.004)
    month = int(rng.integers(1, 13))
    year = int(rng.choice([2025, 2026]))
    a = {
        "asset_id": f"{prefix}-{k + 1:03d}", "split": "train", "site": "Synthetic site (fictional)",
        "system": f"{prof.replace('_', ' ').title()} pump ({ctrl})", "profile": prof, "control": ctrl,
        "start_utc": f"{year}-{month:02d}-{int(rng.integers(1, 22)):02d}T00:00:00", "commissioned_year": int(rng.integers(1995, 2022)),
        "instrumentation_tier": tier, "q_bep_m3h": q, "h_bep_m": h, "eta_bep": eta, "npshr_bep_m": npsh,
        "q_design_m3h": round(q * _r(rng, 0.92, 0.97), 0), "h_design_m": round(h * _r(rng, 0.96, 0.99), 1),
        "motor_kw": motor, "motor_eff": round(eff, 4), "motor_pf": round(_r(rng, 0.84, 0.87), 3), "rated_rpm": rpm,
        "vib_base_mm_s": round(_r(rng, 0.9, 1.7), 2), "unbalance_base_mm_s": round(_r(rng, 0.45, 0.85), 2),
        "hydraulic_vib_mm_s": round(_r(rng, 0.85, 1.35), 2), "cavitation_vib_mm_s": round(_r(rng, 2.5, 3.6), 2),
        "accel_base_g": round(_r(rng, 0.13, 0.25), 3),
    }
    if ctrl == "VFD_PRESSURE":
        a["dp_setpoint_m"] = round(h * _r(rng, 0.78, 0.9), 1)
    if ctrl.startswith("VFD"):
        a["r_min"] = round(_r(rng, 0.5, 0.6), 2)
    if prof == "SHIP_SW_COOLING" and ctrl == "FIXED_SPEED":
        a["system"] = "Seawater cooling pump (direct-on-line, legacy)"
    i_rated = motor * 1000 / (a["motor_eff"] * np.sqrt(3) * (440 if prof.startswith("SHIP") else 400) * a["motor_pf"])
    scen, ctx_override = sample_scenario(rng, prof, ctrl, tier, (q, i_rated))
    a["scenario"] = scen
    ctx = {}
    if ctx_override:
        ctx.update(ctx_override)
    if prof.startswith("SHIP") and rng.random() < 0.3:
        ctx["standby_prob"] = 0.0
    if ctx:
        a["context"] = ctx
    return a


def write_fleet(path, seed, n_assets, prefix, focus=False):
    import pandas as pd
    with open(PUBLIC_YAML, encoding="utf-8") as fh:
        pub = yaml.safe_load(fh)
    datasheets = pd.read_csv(ASSETS_CSV).to_dict("records")
    rng = np.random.default_rng([int(seed), 4242])
    assets = [sample_asset(rng, k, prefix, datasheets, COMBOS_FOCUS if focus else None) for k in range(n_assets)]
    cfg = {"dataset_version": pub.get("dataset_version", "1.0.0"), "public_seed": int(seed),
           "profiles": copy.deepcopy(pub["profiles"]), "assets": _plain(assets)}
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=None, width=200)
    return assets


def _plain(o):
    """Convert numpy scalars recursively so that yaml.safe_dump can write them."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.str_):
        return str(o)
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--prefix", default="SYN")
    ap.add_argument("--focus", action="store_true")
    a = ap.parse_args()
    assets = write_fleet(a.out, a.seed, a.n, a.prefix, a.focus)
    print(f"wrote {len(assets)} assets to {a.out}")
