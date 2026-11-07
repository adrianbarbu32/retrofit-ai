"""Alert policy: turns per-minute fault probabilities into few, early, well-typed advisory alerts.

Per asset and fault type (all decisions at minute t use data up to t only):
  * smoothing: rolling mean of the type probability over `win` valid minutes;
  * trigger: smoothed probability >= thr for `persist` consecutive valid minutes;
  * latch: one alert per episode and type; the type re-arms after `rearm` consecutive valid minutes below thr/2, or
    (machine types) after a locked-out maintenance stop once the probability is below thr;
  * spacing: alerts on one asset are at least `spacing_min` minutes apart (the scorer merges alerts < 5 min apart
    into one burst, and a burst naming two fault types earns no type credit);
  * hard rule: communication loss when telemetry rows have been absent for `gap_min` minutes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import ALL_TYPES, MACHINE_TYPES, TYPE_CLASS

SHORT_EPISODE = {"CAVITATION": 30, "AIR_ENTRAINMENT": 60, "SENSOR_NOISE_SPIKES": 180, "SENSOR_STUCK": 180}


@dataclass
class PolicyParams:
    thr: dict = field(default_factory=lambda: {k: 0.6 for k in ALL_TYPES})
    persist: dict = field(default_factory=lambda: {k: 5 for k in ALL_TYPES})
    win: int = 10
    rearm: dict = field(default_factory=lambda: {k: SHORT_EPISODE.get(k, 360) for k in ALL_TYPES})
    gap_min: int = 2
    spacing_min: int = 6
    enabled: dict = field(default_factory=lambda: {k: True for k in ALL_TYPES})
    # competing hypotheses: after an alert of type k, alerts of the listed types are suppressed for suppress_min
    suppress: dict = field(default_factory=dict)
    suppress_min: int = 720
    # episodic faults (cavitation, air entrainment): re-arm after the pump has been stopped for >= N minutes
    rearm_on_stop: dict = field(default_factory=dict)

    def to_dict(self):
        return {k: getattr(self, k) for k in ("thr", "persist", "win", "rearm", "gap_min", "spacing_min", "enabled",
                                               "suppress", "suppress_min", "rearm_on_stop")}

    @staticmethod
    def from_dict(d):
        p = PolicyParams()
        for k, v in d.items():
            setattr(p, k, v)
        return p


def _run_len(mask: np.ndarray) -> np.ndarray:
    """Length of the run of consecutive True values ending at each index (vectorised)."""
    m = mask.astype(np.int64)
    c = np.cumsum(m)
    reset = np.where(~mask, c, 0)
    return c - np.maximum.accumulate(reset)


def smooth(P: pd.DataFrame, k: str, win: int) -> np.ndarray:
    return P[k].rolling(win, min_periods=max(2, win // 3)).mean().to_numpy()


def type_candidates(s: np.ndarray, thr: float, persist: int, rearm: int, machine: bool, locked: np.ndarray,
                    stop_min: int | None = None) -> list:
    """Alert minutes for one type (before cross-type spacing)."""
    valid = ~np.isnan(s)
    above = valid & (s >= thr)
    rl = _run_len(above)
    cand = np.nonzero(rl == persist)[0]
    if cand.size == 0:
        return []
    # re-arm points: `rearm` consecutive *valid* minutes below thr/2 (invalid minutes neither count nor break)
    vidx = np.nonzero(valid)[0]
    low_v = s[vidx] < 0.5 * thr
    lr = _run_len(low_v)
    rearm_pts = vidx[lr >= rearm]
    if machine and locked.any():
        # after a lockout (repair), the first valid minute below thr re-arms
        lk = np.nonzero(locked)[0]
        below = np.nonzero(valid & (s < thr))[0]
        if below.size:
            pos = np.searchsorted(below, lk)
            pos = pos[pos < below.size]
            rearm_pts = np.union1d(rearm_pts, below[pos])
    if stop_min:
        # a stop (>= stop_min invalid minutes, i.e. pump not in steady operation) ends an episode
        inv_run = _run_len(~valid)
        restart = np.nonzero(valid & np.r_[False, inv_run[:-1] >= stop_min])[0]
        rearm_pts = np.union1d(rearm_pts, restart)
    out, last = [], -1
    for c in cand:
        if last < 0 or (rearm_pts.size and np.any((rearm_pts > last) & (rearm_pts <= c))):
            out.append(int(c))
            last = int(c)
    return out


def alerts_for_asset(P: pd.DataFrame, params: PolicyParams, smoothed: dict | None = None) -> list:
    n = len(P)
    t = pd.to_datetime(P["window_start_utc"], utc=True).reset_index(drop=True)
    locked = (P["locked"].to_numpy() == 1) if "locked" in P else np.zeros(n, bool)
    events = []          # (minute, strength, type)
    S = smoothed if smoothed is not None else {}
    for k in ALL_TYPES:
        if k == "COMMUNICATION_LOSS" or k not in P.columns or not params.enabled.get(k, True):
            continue
        s = S.get(k)
        if s is None:
            s = smooth(P, k, params.win)
            S[k] = s
        thr = params.thr.get(k, 0.6)
        for c in type_candidates(s, thr, int(params.persist.get(k, 5)), int(params.rearm.get(k, 360)), k in MACHINE_TYPES, locked,
                                 params.rearm_on_stop.get(k)):
            events.append((c, float(s[c] / thr), k))
    if params.enabled.get("COMMUNICATION_LOSS", True) and "gap_run" in P:
        g = P["gap_run"].to_numpy()
        ok_run = _run_len(g == 0)
        armed = True
        for j in np.nonzero((g >= params.gap_min) | (ok_run == 30))[0]:
            if g[j] >= params.gap_min and armed:
                events.append((int(j), 10.0, "COMMUNICATION_LOSS"))
                armed = False
            elif ok_run[j] == 30:
                armed = True
    # cross-type spacing: sort by time (ties: strongest first) and delay later alerts to keep the spacing
    events.sort(key=lambda e: (e[0], -e[1]))
    out, last = [], -10 ** 9
    blocked = {}          # type -> minute until which it is suppressed
    for j, strength, k in events:
        if blocked.get(k, -1) >= j:
            continue
        j2 = max(j, last + params.spacing_min)
        if j2 >= n:
            continue
        last = j2
        for other in params.suppress.get(k, []):
            blocked[other] = max(blocked.get(other, -1), j2 + params.suppress_min)
        sk = S.get(k)
        conf = 0.95 if k == "COMMUNICATION_LOSS" else float(np.clip(np.nan_to_num(sk[j2] if sk is not None else 0.9, nan=sk[j] if sk is not None else 0.9), 0.05, 0.99))
        out.append({"asset_id": P["asset_id"].iloc[0], "alert_time_utc": (t[j2] + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "event_class": TYPE_CLASS[k], "fault_type": k, "confidence": round(conf, 3), "_row": int(j2),
                    "_strength": round(strength, 3)})
    return out


def run_policy(prob_frames: list, params: PolicyParams, cache: list | None = None) -> pd.DataFrame:
    rows = []
    for i, P in enumerate(prob_frames):
        rows += alerts_for_asset(P, params, cache[i] if cache is not None else None)
    cols = ["asset_id", "alert_time_utc", "event_class", "fault_type", "confidence", "_row", "_strength"]
    return pd.DataFrame(rows, columns=cols)
