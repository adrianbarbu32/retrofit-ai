"""Human-readable evidence and recommended (human-approved) actions for each alert."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import NICE, UNIT

ACTIONS = {
    "BEARING_WEAR": "Plan a pump drive-end bearing replacement at the next convenient stop; check lubrication and confirm "
                    "with a portable vibration / envelope (BPFO) reading. Keep the existing vibration and temperature trips as they are.",
    "SUCTION_STRAINER_CLOGGING": "Inspect and clean (backflush) the suction strainer at the next opportunity; record the strainer "
                                 "differential pressure before and after.",
    "CAVITATION": "Check suction conditions now: strainer, suction valve fully open, tank / basin / sea-chest level, inlet "
                  "temperature. Any flow or speed change only via the normal procedure (what-if checked in the sandbox first).",
    "DISCHARGE_RESTRICTION": "Check discharge-side valve line-up (valve left partly closed?) and heat-exchanger fouling / "
                             "differential pressure.",
    "MISALIGNMENT": "Check shaft alignment and coupling condition (laser alignment, soft foot), in particular after recent "
                    "maintenance work.",
    "IMBALANCE": "Inspect the impeller for fouling, deposits or damage; plan cleaning or balancing.",
    "IMPELLER_WEAR": "Schedule a pump performance test and an impeller / wear-ring inspection; plan the overhaul with the "
                     "maintenance planner.",
    "AIR_ENTRAINMENT": "Check suction submergence and level (basin make-up, sea-chest venting), vortexing and air ingress at "
                       "suction joints.",
    "SENSOR_DRIFT": "Verify {ch} against a reference gauge and recalibrate; treat its readings as unreliable until checked. "
                    "No machine intervention needed for this alert.",
    "SENSOR_STUCK": "Check the {ch} transmitter, wiring and signal path (value frozen). No machine intervention needed.",
    "SENSOR_CALIBRATION_SHIFT": "Check the {ch} zero / span calibration (step change, e.g. after maintenance). No machine "
                                "intervention needed.",
    "SENSOR_NOISE_SPIKES": "Check {ch} cabling, shield / grounding and connector (noisy readings). No machine intervention needed.",
    "COMMUNICATION_LOSS": "Check the gateway / network link; monitoring data missing. Local alarms and trips remain authoritative.",
    "UNKNOWN": "Inspect the pump; deviation from the healthy model with unclear cause.",
}

# feature families that carry the evidence for each type (short channel codes)
EVIDENCE = {
    "BEARING_WEAR": ["a", "c", "bt", "v"],
    "SUCTION_STRAINER_CLOGGING": ["ps", "n", "pw", "vp", "q", "pd", "i"],
    "CAVITATION": ["ps", "a", "pd", "v", "q"],
    "DISCHARGE_RESTRICTION": ["pd", "q", "i", "pw", "n", "vp"],
    "MISALIGNMENT": ["v", "bt", "wt", "pw"],
    "IMBALANCE": ["v", "a"],
    "IMPELLER_WEAR": ["curve", "dh", "pd", "i", "pw", "n", "vp"],
    "AIR_ENTRAINMENT": ["pd", "q", "dh", "v", "a", "pw"],
}
SENSOR_CH = ["pd", "ps", "i", "pw", "q", "bt", "tf", "wt", "n", "vp"]
# physically expected direction of the deviation for each fault (0 = depends on the control strategy)
DIRECTION = {
    "BEARING_WEAR": {"a": 1, "c": 1, "bt": 1, "v": 1},
    "SUCTION_STRAINER_CLOGGING": {"ps": -1, "q": -1},
    "CAVITATION": {"ps": -1, "a": 1, "pd": -1, "v": 1, "q": -1},
    "DISCHARGE_RESTRICTION": {"q": -1},
    "MISALIGNMENT": {"v": 1, "bt": 1, "wt": 1, "pw": 1},
    "IMBALANCE": {"v": 1},
    "IMPELLER_WEAR": {"curve": -1, "dh": -1},
    "AIR_ENTRAINMENT": {"pd": -1, "q": -1, "dh": -1, "v": 1, "a": 1},
}


def _scale(row) -> dict:
    hd = 9.81 * float(row["design_head_m"])
    return {"n": float(row["motor_rated_rpm"]) / 100.0, "i": float(row["motor_rated_current_a"]) / 100.0,
            "pw": float(row["motor_rated_kw"]) / 100.0, "ps": hd / 100.0, "pd": hd / 100.0, "dh": hd / 100.0,
            "q": float(row["design_flow_m3_h"]) / 100.0, "vp": 1.0, "tf": 1.0, "bt": 1.0, "wt": 1.0, "c": 1.0}


def _fmt_dev(s, f, sc):
    lvl, dref = f.get(f"{s}_lvl"), f.get(f"{s}_dref")
    if lvl is None or pd.isna(lvl):
        return None, 0.0
    if s in ("v", "a"):
        txt = f"{NICE[s]} {lvl:+.0f}% vs healthy model"
        if dref is not None and pd.notna(dref) and abs(dref) > 5:
            txt += f" ({dref:+.0f}% since baseline)"
        return txt, abs(lvl) / 15.0 + abs(dref or 0) / 10.0
    if s == "curve":
        return f"pump head {lvl:+.1f}% of design head vs the pump curve at the measured flow", abs(lvl) / 1.5
    k = sc.get(s, 1.0)
    u = UNIT.get(s, "")
    us = f" {u}" if u else ""
    txt = f"{NICE.get(s, s)} {lvl * k:+.1f}{us} vs healthy model"
    if dref is not None and pd.notna(dref) and abs(dref * k) > 0.5 * abs(lvl * k) and abs(dref) > 0.3:
        txt += f" ({dref * k:+.1f}{us} since baseline)"
    norm = {"n": 0.5, "i": 1.5, "pw": 1.5, "ps": 1.5, "pd": 1.5, "dh": 1.5, "q": 1.5, "vp": 2.0, "tf": 0.5, "bt": 1.5, "wt": 2.0, "c": 0.4}.get(s, 1.0)
    return txt, (abs(lvl) + abs(dref or 0)) / norm


MIN_ATTRIBUTION = {"SENSOR_DRIFT": 0.5, "SENSOR_CALIBRATION_SHIFT": 0.5, "SENSOR_STUCK": 1.0, "SENSOR_NOISE_SPIKES": 0.3}


def explainable(ftype: str, f: pd.Series) -> bool:
    """An instrument alert must point at a specific instrument with a measurable deviation (no unexplainable alerts)."""
    if ftype not in MIN_ATTRIBUTION:
        return True
    return channel_score(ftype, f)[1] >= MIN_ATTRIBUTION[ftype]


def sensor_channel(ftype: str, f: pd.Series) -> str | None:
    return channel_score(ftype, f)[0]


def channel_score(ftype: str, f: pd.Series) -> tuple[str | None, float]:
    """Which channel the instrument fault most likely concerns, and how strongly (feature units)."""
    best, bs = None, 0.0
    for s in SENSOR_CH:
        if ftype == "SENSOR_STUCK":
            v = f.get(f"{s}_flat")
            sc = (v or 0) / 10.0 if pd.notna(v) else 0
        elif ftype == "SENSOR_NOISE_SPIKES":
            v1, v2 = f.get(f"{s}_noise"), f.get(f"{s}_range")
            sc = max(v1 if pd.notna(v1) else 0, (v2 if pd.notna(v2) else 0))
        elif ftype == "SENSOR_DRIFT":
            v1, v2 = f.get(f"{s}_d6h"), f.get(f"{s}_dref")
            sc = max(abs(v1) if pd.notna(v1) else 0, abs(v2) if pd.notna(v2) else 0)
        else:  # calibration shift: step since baseline
            v1, v2 = f.get(f"{s}_dref"), f.get(f"{s}_d1h")
            sc = max(abs(v1) if pd.notna(v1) else 0, abs(v2) if pd.notna(v2) else 0)
        if s == "ivp":
            continue
        if sc > bs:
            best, bs = s, sc
    return best, float(bs)


def explain(ftype: str, f: pd.Series, row: pd.Series, probs: dict | None = None) -> tuple[str, str]:
    """(evidence, recommended_action) for an alert of type `ftype` raised on feature row `f`."""
    sc = _scale(row)
    parts = []
    if ftype == "COMMUNICATION_LOSS":
        parts.append(f"no telemetry rows for {int(f.get('gap_run', 0) or 0)} min (gateway / network)")
        return "; ".join(parts), ACTIONS[ftype]
    if ftype.startswith("SENSOR"):
        s = sensor_channel(ftype, f)
        ch = NICE.get(s, s) if s else "affected"
        if s:
            if ftype == "SENSOR_STUCK":
                parts.append(f"{ch} reading unchanged for {int(f.get(f'{s}_flat', 0) or 0)} min while the pump runs")
            elif ftype == "SENSOR_NOISE_SPIKES":
                parts.append(f"{ch} scatter x{np.exp(f.get(f'{s}_noise', 0) or 0):.1f} of its normal level, peak-to-peak "
                             f"x{np.exp(f.get(f'{s}_range', 0) or 0):.1f}")
            else:
                txt, _ = _fmt_dev(s, f, sc)
                if txt:
                    parts.append(txt)
            # physical consistency: coupled channels unchanged
            others = []
            for o in ("pd", "ps", "i", "pw", "q", "n", "v"):
                if o == s:
                    continue
                v = f.get(f"{o}_dref")
                if v is not None and pd.notna(v):
                    others.append(abs(v))
            if others:
                parts.append("physically coupled signals consistent with the healthy model -> instrument, not machine")
        if not explainable(ftype, f):
            return ("early, weak instrument-fault signature: readings drift away from the pump physics but no single "
                    "instrument stands out yet; watch whether it grows",
                    "No action yet. If the alert persists, verify the transmitters at the next routine round.")
        return "; ".join(parts) or "instrument behaviour inconsistent with the pump physics", ACTIONS[ftype].format(ch=ch)
    avail = set(str(row.get("channels_available", "")).split("|"))
    no_vib = not ({"vibration_rms_mm_s", "vibration_accel_rms_g"} & avail)
    if ftype in ("BEARING_WEAR", "MISALIGNMENT", "IMBALANCE") and no_vib:
        parts.append("no vibration sensor on this pump: inferred from motor power / temperature only (weak evidence)")
    cands = []
    for s in EVIDENCE.get(ftype, []):
        txt, w = _fmt_dev(s, f, sc)
        d = DIRECTION.get(ftype, {}).get(s, 0)
        lvl = f.get(f"{s}_lvl")
        if d and lvl is not None and pd.notna(lvl) and np.sign(lvl) != d:
            continue                      # deviation in the wrong direction is not evidence for this fault
        if txt and w > 0.8:
            cands.append((w, txt))
    cands.sort(reverse=True)
    parts += [t for _, t in cands[:3]]
    if ftype == "CAVITATION" and pd.notna(f.get("npsh_meas_min30", np.nan)):
        parts.append(f"NPSH margin {f['npsh_meas_min30']:.2f} (cavitation risk below 1.15)")
    if ftype == "BEARING_WEAR":
        if pd.notna(f.get("bpfo_env", np.nan)) and f["bpfo_env"] > 3:
            parts.append(f"envelope spectrum peak at bearing outer-race frequency (BPFO) {f['bpfo_env']:.1f}x noise floor")
        if pd.notna(f.get("c_abs", np.nan)):
            parts.append(f"crest factor {f['c_abs']:.1f} (healthy ~3.2)")
    if ftype in ("MISALIGNMENT", "IMBALANCE") and pd.notna(f.get("v21", np.nan)):
        parts.append(f"2x/1x running-speed amplitude ratio {np.exp(f['v21']):.2f}")
    if ftype == "MISALIGNMENT" and pd.notna(f.get("since_lockout", np.nan)) and f["since_lockout"] < 48 * 60:
        m = float(f["since_lockout"])
        parts.append(f"began after a maintenance stop that ended {m:.0f} min ago" if m < 120 else
                     f"began after a maintenance stop that ended {m / 60:.0f} h ago")
    if ftype == "AIR_ENTRAINMENT":
        if pd.notna(f.get("suction_level_drop", np.nan)) and f["suction_level_drop"] > 0.5:
            parts.append(f"suction level {f['suction_level_drop']:.1f} m below its normal value")
        if pd.notna(f.get("sea_max60", np.nan)) and f["sea_max60"] > 2.0:
            parts.append(f"heavy weather (Hs {f['sea_max60']:.1f} m)")
    if probs:
        alt = sorted(((p, k) for k, p in probs.items() if k != ftype and pd.notna(p)), reverse=True)
        if alt and alt[0][0] > 0.2:
            parts.append(f"other condition also indicated: {alt[0][1].replace('_', ' ').lower()} ({alt[0][0]:.0%})")
    action = ACTIONS.get(ftype, ACTIONS["UNKNOWN"])
    if ftype in ("BEARING_WEAR", "MISALIGNMENT", "IMBALANCE") and no_vib:
        action = "First confirm with a portable vibration reading (route measurement). " + action
    return "; ".join(parts) or "deviation from the healthy-pump model", action
