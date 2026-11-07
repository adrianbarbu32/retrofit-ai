"""Spectral features from 1-s vibration waveform snapshots (10.24 kHz acceleration, every 4 h on T1/T2 assets).

Features per snapshot:
  v1x, v2x, v3x, vvp   velocity amplitude [mm/s RMS] at 1x, 2x, 3x running speed and vane pass (unbalance /
                       misalignment / hydraulic signatures)
  bpfo_env             envelope-spectrum peak at the bearing outer-race defect frequency (BPFO) relative to the
                       envelope noise floor (resonance band 2.0-3.6 kHz) -> rolling-bearing damage
  kurt_res             kurtosis of the resonance-band signal (impulsiveness)
  hf_rms               RMS of the 1-4.5 kHz band [g] (cavitation / air broadband)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

G = 9.80665


def _peak(f, X, f0, tol=0.02):
    m = (f >= f0 * (1 - tol)) & (f <= f0 * (1 + tol))
    return float(X[m].max()) if m.any() else 0.0


def snapshot_features(x: np.ndarray, fs: float, rpm: float, bpfo_order: float, vanes: int) -> dict:
    x = np.asarray(x, dtype=float)
    n = x.size
    fr = rpm / 60.0
    out = {"snap_rpm": rpm}
    if fr < 2 or n < 1024:
        return out
    x = x - x.mean()
    w = np.hanning(n)
    X = np.abs(np.fft.rfft(x * w)) * 2.0 / (n * 0.5)          # peak amplitude [g]
    f = np.fft.rfftfreq(n, 1.0 / fs)
    to_v = lambda a_pk, ff: a_pk * G / (2 * np.pi * ff) / np.sqrt(2) * 1000.0  # g peak -> mm/s RMS
    out["v1x"] = to_v(_peak(f, X, fr), fr)
    out["v2x"] = to_v(_peak(f, X, 2 * fr), 2 * fr)
    out["v3x"] = to_v(_peak(f, X, 3 * fr), 3 * fr)
    out["vvp"] = to_v(_peak(f, X, vanes * fr), vanes * fr)
    F = np.fft.rfft(x)
    band = (f >= 2000) & (f <= 3600)
    xr = np.fft.irfft(np.where(band, F, 0), n)
    hf = np.fft.irfft(np.where((f >= 1000) & (f <= 4500), F, 0), n)
    out["hf_rms"] = float(np.sqrt(np.mean(hf ** 2)))
    s = xr.std() + 1e-12
    out["kurt_res"] = float(np.mean(((xr - xr.mean()) / s) ** 4))
    # analytic-signal envelope of the resonance band
    Z = np.zeros(n, dtype=complex)
    Fr = np.fft.fft(xr)
    Z[0] = Fr[0]
    Z[1:(n + 1) // 2] = 2 * Fr[1:(n + 1) // 2]
    if n % 2 == 0:
        Z[n // 2] = Fr[n // 2]
    env = np.abs(np.fft.ifft(Z))
    env = env - env.mean()
    E = np.abs(np.fft.rfft(env * w))
    fe = np.fft.rfftfreq(n, 1.0 / fs)
    bpfo = bpfo_order * fr
    floor = np.median(E[(fe > 5) & (fe < 400)]) + 1e-12
    out["bpfo_env"] = float(max(_peak(fe, E, bpfo, 0.03), _peak(fe, E, 2 * bpfo, 0.03)) / floor)
    return out


def asset_snapshot_features(snaps: pd.DataFrame, asset_row: pd.Series) -> pd.DataFrame:
    if snaps is None or len(snaps) == 0:
        return pd.DataFrame()
    rows = []
    bpfo = float(asset_row.get("bearing_bpfo_order", 3.05))
    vanes = int(asset_row.get("impeller_vanes", 6))
    for _, s in snaps.iterrows():
        d = snapshot_features(np.asarray(s["samples"], dtype=float), float(s["sample_rate_hz"]), float(s["rpm_at_capture"]), bpfo, vanes)
        d["snap_time"] = pd.Timestamp(s["timestamp_utc"]).tz_convert("UTC") if pd.Timestamp(s["timestamp_utc"]).tzinfo else pd.Timestamp(s["timestamp_utc"]).tz_localize("UTC")
        rows.append(d)
    return pd.DataFrame(rows).sort_values("snap_time").reset_index(drop=True)
