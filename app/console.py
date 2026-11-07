"""RETROFIT AI operator console (advisory only).

Run:  streamlit run app/console.py
Replays the supplied synthetic data minute by minute. Everything shown at replay time t is computed from data up to t.
The console has no write path to any machine, controller or safety system; operator actions are recorded as requests
in a local log for a human maintenance planner.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from retrofit_ai import config  # noqa: E402
from retrofit_ai.config import NICE, TYPE_CLASS  # noqa: E402
from retrofit_ai.data import Dataset  # noqa: E402
from retrofit_ai.explain import EVIDENCE  # noqa: E402
from retrofit_ai.twin import load_profiles, twin_for_asset  # noqa: E402

st.set_page_config(page_title="RETROFIT AI - Operator Assist", layout="wide", page_icon=":wrench:")
CONSOLE = os.path.join(config.RESULTS, "console")
LOG = os.path.join(config.RESULTS, "operator_log.jsonl")
PRETTY = lambda k: k.replace("_", " ").capitalize()
CH_FOR = {  # telemetry columns to plot per fault type
    "BEARING_WEAR": ["vibration_accel_rms_g", "vibration_crest_factor", "bearing_temp_c", "vibration_rms_mm_s"],
    "SUCTION_STRAINER_CLOGGING": ["suction_pressure_kpa", "motor_speed_rpm", "motor_power_kw", "valve_position_pct", "flow_m3_h", "discharge_pressure_kpa"],
    "CAVITATION": ["suction_pressure_kpa", "discharge_pressure_kpa", "vibration_accel_rms_g", "flow_m3_h"],
    "DISCHARGE_RESTRICTION": ["discharge_pressure_kpa", "flow_m3_h", "motor_current_a", "motor_speed_rpm", "valve_position_pct"],
    "MISALIGNMENT": ["vibration_rms_mm_s", "bearing_temp_c", "motor_winding_temp_c", "motor_power_kw"],
    "IMBALANCE": ["vibration_rms_mm_s", "vibration_accel_rms_g", "motor_speed_rpm"],
    "IMPELLER_WEAR": ["discharge_pressure_kpa", "motor_current_a", "motor_power_kw", "motor_speed_rpm", "flow_m3_h"],
    "AIR_ENTRAINMENT": ["discharge_pressure_kpa", "flow_m3_h", "motor_power_kw", "vibration_rms_mm_s"],
}
SENSOR_DEFAULT = ["discharge_pressure_kpa", "suction_pressure_kpa", "motor_current_a", "bearing_temp_c", "flow_m3_h"]
COL_NAME = {"motor_speed_rpm": "speed [rpm]", "motor_current_a": "current [A]", "motor_power_kw": "power [kW]",
            "suction_pressure_kpa": "suction pressure [kPa]", "discharge_pressure_kpa": "discharge pressure [kPa]",
            "flow_m3_h": "flow [m3/h]", "fluid_temp_c": "fluid temp [degC]", "bearing_temp_c": "bearing temp [degC]",
            "motor_winding_temp_c": "winding temp [degC]", "vibration_rms_mm_s": "vibration [mm/s]",
            "vibration_accel_rms_g": "acceleration [g]", "vibration_crest_factor": "crest factor [-]",
            "valve_position_pct": "valve position [%]"}


@st.cache_resource
def dataset():
    return Dataset(config.DATA_FULL, "official")


@st.cache_data
def alerts_all():
    p = os.path.join(CONSOLE, "alerts_all.csv")
    if not os.path.exists(p):
        a = pd.DataFrame(columns=["asset_id", "alert_time_utc", "event_class", "fault_type", "confidence", "evidence", "recommended_action"])
    else:
        a = pd.read_csv(p)
    a["t"] = pd.to_datetime(a["alert_time_utc"], utc=True)
    return a


@st.cache_data
def pump_data(aid: str):
    ds = dataset()
    fr = ds.asset_frame(aid)
    row = ds.assets[ds.assets.asset_id == aid].iloc[0]
    ener = (fr["running_fraction"] > 0.5).to_numpy()
    tw = twin_for_asset(row, fr, ener, load_profiles())
    pp = os.path.join(config.BUILD, "probs", "official", f"{aid}.parquet")
    probs = pd.read_parquet(pp) if os.path.exists(pp) else None
    return fr, tw, probs


@st.cache_data
def observability():
    p = os.path.join(config.RESULTS, "observability_by_tier.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    o = pd.read_csv(p)
    o["vfd"] = o["vfd"].astype(str).str.lower() == "true"
    return o


def log_action(entry: dict):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def read_log():
    if not os.path.exists(LOG):
        return pd.DataFrame()
    with open(LOG) as fh:
        return pd.DataFrame([json.loads(x) for x in fh if x.strip()])


# ---------------------------------------------------------------- sidebar: pump and replay time
ds = dataset()
A = ds.assets.copy()
label = {r.asset_id: f"{r.asset_id} - {r.system} [{r.instrumentation_tier.split('_')[0]}, {r.split}]" for r in A.itertuples()}
st.sidebar.title("RETROFIT AI")
st.sidebar.caption("Operator assist for existing pumps - advisory only, read-only, no control path")
aid = st.sidebar.selectbox("Pump", list(A.asset_id), index=7, format_func=lambda x: label[x])
row = A[A.asset_id == aid].iloc[0]
t0 = pd.Timestamp(row.window_start_utc)
t1 = pd.Timestamp(row.window_end_utc)
if "hours" not in st.session_state or st.session_state.get("aid") != aid:
    st.session_state.hours = 168.0
    st.session_state.aid = aid
play = st.sidebar.toggle("Replay (live mode)", value=False, help="Advance the replay clock automatically")
hours = st.sidebar.slider("Replay clock: hours since window start", 0.0, 168.0, float(st.session_state.hours), 0.5, key="hours_slider")
st.session_state.hours = hours
now = t0 + pd.Timedelta(hours=hours)
st.sidebar.markdown(f"**Now:** {now:%Y-%m-%d %H:%M} UTC")
show_truth = st.sidebar.checkbox("Overlay labelled events (train / validation pumps only)", value=False,
                                 disabled=row.split == "test")
st.sidebar.divider()
st.sidebar.markdown(f"**Site:** {row.site}  \n**Duty:** {row.system}  \n**Control:** {row.control}  \n"
                    f"**Sensors:** {row.instrumentation_tier}  \n**Rated:** {row.pump_rated_flow_m3_h:.0f} m3/h @ {row.pump_rated_head_m:.0f} m, "
                    f"{row.motor_rated_kw:.0f} kW")

AL = alerts_all()
al_pump = AL[(AL.asset_id == aid) & (AL.t <= now)].sort_values("t", ascending=False)

tab_pump, tab_fleet, tab_whatif, tab_edge, tab_perf = st.tabs(["Pump view", "Fleet overview", "What-if sandbox (simulation only)",
                                                                  "Integration & edge", "Performance & limits"])

# ---------------------------------------------------------------- pump view
with tab_pump:
    fr, tw, probs = pump_data(aid)
    m = fr["window_start_utc"] <= now
    frm, twm = fr[m], tw[m]
    c1, c2, c3, c4 = st.columns(4)
    running = bool(frm["running_fraction"].iloc[-1] > 0.5) if len(frm) else False
    mode = frm["operating_mode"].iloc[-1] if len(frm) else "-"
    recent = al_pump[al_pump.t >= now - pd.Timedelta(hours=24)]
    status = "ALERT" if len(recent) else "NORMAL"
    c1.metric("Status (last 24 h)", status, f"{len(recent)} alert(s)" if len(recent) else "no alerts", delta_color="inverse")
    c2.metric("Operating mode", str(mode))
    c3.metric("Duty demand", f"{frm['duty_demand_pct'].iloc[-1]:.0f} %" if len(frm) else "-")
    c4.metric("Alerts so far", f"{len(al_pump)}")
    st.info("Advisory system. Existing alarms, trips and interlocks remain authoritative. Every recommendation needs "
            "human review; nothing here can command the pump, drive, valves or safety system.", icon="🛡️")
    obs = observability()
    if len(obs):
        o = obs[(obs.instrumentation_tier == row.instrumentation_tier) & (obs.vfd == str(row.control).startswith("VFD"))]
        blind = sorted(PRETTY(k) for k, v in zip(o.fault_type, o.observable_share) if v < 0.5)
        weak = sorted(PRETTY(k) for k, v in zip(o.fault_type, o.observable_share) if 0.5 <= v < 0.9)
        if blind or weak:
            st.caption("Blind spots with this pump's instruments (share of simulated faults that are observable at all): "
                       + (f"**not visible:** {', '.join(blind)}. " if blind else "")
                       + (f"**partly visible:** {', '.join(weak)}." if weak else "")
                       + " Absence of an alert is therefore no proof of health for these faults.")
    left, right = st.columns([3, 2])
    with right:
        st.subheader("Alerts")
        if not len(al_pump):
            st.success("No alerts up to now. Readings consistent with the healthy-pump model.")
        for i, a in enumerate(al_pump.head(8).itertuples()):
            icon = "🔧" if a.event_class == "MACHINE" else "📟"
            with st.container(border=True):
                name = PRETTY(a.fault_type) if a.confidence >= 0.55 else f"Possible {PRETTY(a.fault_type).lower()} (unconfirmed)"
                st.markdown(f"{icon} **{name}** ({a.event_class.lower()} fault) - confidence **{a.confidence:.0%}**  \n"
                            f"raised {a.t:%Y-%m-%d %H:%M} UTC ({(now - a.t).total_seconds() / 3600:.1f} h ago)")
                st.markdown("**Evidence:** " + str(a.evidence))
                st.markdown("**Recommended (needs human approval):** " + str(a.recommended_action))
                b1, b2, b3 = st.columns(3)
                key = f"{aid}-{a.alert_time_utc}-{a.fault_type}"
                comment = ""
                if b1.button("Acknowledge", key="ack" + key):
                    log_action({"time": datetime.now(timezone.utc).isoformat(), "asset_id": aid, "alert": key, "action": "ACKNOWLEDGED"})
                    st.toast("Acknowledged and logged")
                if b2.button("Reject (false alarm)", key="rej" + key):
                    log_action({"time": datetime.now(timezone.utc).isoformat(), "asset_id": aid, "alert": key, "action": "REJECTED_FALSE_ALARM"})
                    st.toast("Logged as false alarm (feeds the shadow-mode review)")
                if b3.button("Request work order", key="wo" + key):
                    log_action({"time": datetime.now(timezone.utc).isoformat(), "asset_id": aid, "alert": key, "action": "WORK_ORDER_REQUESTED",
                                "recommendation": a.recommended_action})
                    st.toast("Work-order request sent to the maintenance planner queue (no machine action)")
    with left:
        ftype = al_pump.iloc[0].fault_type if len(al_pump) else None
        avail = [c for c in str(row.channels_available).split("|") if c in COL_NAME]
        default = [c for c in (CH_FOR.get(ftype, SENSOR_DEFAULT) if ftype else SENSOR_DEFAULT) if c in avail][:4]
        if not default:
            default = avail[:3]
        chans = st.multiselect("Signals (measured vs healthy-pump model)", avail, default=default, format_func=lambda c: COL_NAME[c])
        win_h = st.select_slider("Window", options=[6, 12, 24, 48, 96, 168], value=48, format_func=lambda h: f"last {h} h")
        mw = frm["window_start_utc"] >= now - pd.Timedelta(hours=win_h)
        x = frm.loc[mw, "window_start_utc"]
        nrow = len(chans) + 1
        fig = make_subplots(rows=nrow, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            subplot_titles=[COL_NAME[c] for c in chans] + ["fault probabilities (diagnosis model)"])
        for i, c in enumerate(chans):
            y = frm.loc[mw, f"{c}__mean"]
            run = frm.loc[mw, "running_fraction"] >= 0.99
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="measured", line=dict(width=1.2, color="#1f77b4"),
                                     showlegend=i == 0), row=i + 1, col=1)
            if c in twm:
                fig.add_trace(go.Scatter(x=x, y=twm.loc[mw, c].where(run), mode="lines", name="healthy model",
                                         line=dict(width=1.2, dash="dash", color="#ff7f0e"), showlegend=i == 0), row=i + 1, col=1)
        if probs is not None:
            pm = probs[(pd.to_datetime(probs.window_start_utc, utc=True) <= now) & (pd.to_datetime(probs.window_start_utc, utc=True) >= now - pd.Timedelta(hours=win_h))]
            types = [k for k in TYPE_CLASS if k in pm]
            top = sorted(types, key=lambda k: -np.nan_to_num(pm[k].rolling(10, min_periods=1).mean().max()))[:4]
            for k in top:
                fig.add_trace(go.Scatter(x=pd.to_datetime(pm.window_start_utc, utc=True), y=pm[k].rolling(10, min_periods=1).mean(), mode="lines",
                                         name=PRETTY(k), line=dict(width=1.3)), row=nrow, col=1)
            fig.update_yaxes(range=[0, 1], row=nrow, col=1)
        for a in al_pump[al_pump.t >= now - pd.Timedelta(hours=win_h)].itertuples():
            fig.add_vline(x=a.t, line=dict(color="red" if a.event_class == "MACHINE" else "purple", width=1, dash="dot"))
        if show_truth and row.split != "test":
            ev = ds.events[ds.events.asset_id == aid]
            for e in ev.itertuples():
                s = pd.Timestamp(e.detectable_utc if isinstance(e.detectable_utc, str) else e.onset_utc)
                en = pd.Timestamp(e.end_utc) if isinstance(e.end_utc, str) else t1
                s, en = max(s, now - pd.Timedelta(hours=win_h)), min(en, now)
                if s < en:
                    fig.add_vrect(x0=s, x1=en, fillcolor="green", opacity=0.07, line_width=0, annotation_text=PRETTY(e.fault_type),
                                  annotation_position="top left", annotation_font_size=9)
        fig.update_layout(height=170 * nrow + 60, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, width="stretch")
        gaps = int((frm.loc[mw, "rows"] < 30).sum())
        st.caption(f"Data quality in window: {gaps} min without telemetry. Dashed line = what a healthy pump with this datasheet "
                   f"would read under the same duty, inlet temperature and run state (physics model). Dotted verticals = alerts.")

# ---------------------------------------------------------------- fleet overview
with tab_fleet:
    st.subheader("Fleet status at each pump's replay clock")
    rows = []
    for r in A.itertuples():
        nw = pd.Timestamp(r.window_start_utc) + pd.Timedelta(hours=hours)
        a = AL[(AL.asset_id == r.asset_id) & (AL.t <= nw)].sort_values("t")
        last = a.iloc[-1] if len(a) else None
        rows.append({"pump": r.asset_id, "site": r.site, "duty": r.system, "sensors": r.instrumentation_tier, "control": r.control,
                     "split": r.split, "alerts": len(a),
                     "latest alert": f"{PRETTY(last.fault_type)} ({last.confidence:.0%})" if last is not None else "-",
                     "since": f"{(nw - last.t).total_seconds() / 3600:.1f} h" if last is not None else "-"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Test pumps (PUMP-10/11/12) are unlabelled: their alerts are the submitted predictions.")
    lg = read_log()
    st.subheader("Operator decisions log (human-in-the-loop audit trail)")
    st.dataframe(lg if len(lg) else pd.DataFrame(columns=["time", "asset_id", "alert", "action"]), width="stretch", hide_index=True)

# ---------------------------------------------------------------- what-if sandbox
with tab_whatif:
    st.warning("SIMULATION ONLY. The sandbox evaluates a proposed action against the reduced-order pump model. It is not a "
               "set-point, not a work order and has no connection to any controller.", icon="🧪")
    from retrofit_sim.sandbox import HYPOTHESIS_KEYS, INTERVENTIONS, what_if
    from retrofit_ai.severity import estimate
    fr, tw, probs = pump_data(aid)
    j = int(np.searchsorted(fr["window_start_utc"].values, np.datetime64(now.tz_convert(None)), side="right") - 1)
    j = max(0, min(j, len(fr) - 1))
    mach = al_pump[al_pump.event_class == "MACHINE"]
    hyp = {}
    if len(mach):
        a0 = mach.iloc[0]
        est = estimate(a0.fault_type, fr, j, row)
        if est:
            hyp[est["hypothesis_key"]] = round(est["value"], 3)
            st.markdown(f"**Latest machine alert:** {PRETTY(a0.fault_type)}. Physics-based severity estimate: "
                        f"**{est['hypothesis_key']} = {est['value']:.2f}** (fit improves error {est['healthy_error']:.0f} -> {est['fit_error']:.0f}).")
            cmp = pd.DataFrame({"measured (30-min median)": est["measured"], "model with estimated condition": est["predicted"],
                                "healthy model": est["healthy_pred"]}).round(2)
            st.dataframe(cmp, width="stretch")
    c1, c2 = st.columns(2)
    default_action = {"SUCTION_STRAINER_CLOGGING": "clean_strainer", "CAVITATION": "clean_strainer", "MISALIGNMENT": "realign",
                      "BEARING_WEAR": "replace_bearing", "IMBALANCE": "inspect", "IMPELLER_WEAR": "inspect",
                      "DISCHARGE_RESTRICTION": "inspect", "AIR_ENTRAINMENT": "inspect"}.get(mach.iloc[0].fault_type if len(mach) else "", "none")
    acts = sorted(INTERVENTIONS)
    action = c1.selectbox("Proposed action", acts, index=acts.index(default_action) if default_action in acts else 0)
    value = c1.number_input("Value (speed ratio 0-1 or valve %, only for set-point actions)", value=0.0) if action in ("speed_setpoint", "valve_position") else None
    hyp_txt = c2.text_input("Condition hypothesis (key=value, comma separated)", ", ".join(f"{k}={v}" for k, v in hyp.items()),
                            help=f"keys: {', '.join(sorted(HYPOTHESIS_KEYS))}")
    if st.button("Evaluate in sandbox"):
        hy = {}
        for part in hyp_txt.split(","):
            if "=" in part:
                k, v = part.split("=")
                hy[k.strip()] = float(v)
        r = what_if(config.DATA_FULL, aid, now.isoformat(), action, value, hy)
        if r.get("accepted"):
            st.success("Sandbox: proposal passes the safety and duty checks (simulation). A human decides.")
        else:
            st.error("Sandbox rejects the proposal: " + "; ".join(r.get("rejection_reasons", [])))
        if r.get("warnings"):
            st.warning("; ".join(r["warnings"]))
        if "baseline" in r:
            keys = ["speed_ratio", "flow_m3h", "duty_satisfaction", "input_kw", "npsh_margin", "vib_rms_mm_s", "bearing_temp_c", "risk_proxy", "energy_eur_per_day"]
            st.dataframe(pd.DataFrame({"now (hypothesis)": {k: r["baseline"].get(k) for k in keys},
                                       "after proposed action": {k: r["proposed"].get(k) for k in keys}}), width="stretch")
            st.caption(f"Intervention cost estimate: {r['intervention']['estimated_cost_eur']} EUR (assumptions in the sandbox). "
                       + " ".join(r.get("caveats", [])))

# ---------------------------------------------------------------- integration & edge
with tab_edge:
    st.subheader("Read-only retrofit architecture")
    st.code("""
 [ Safety system / SIS ]      own zone - untouched, authoritative trips & interlocks
 [ PLC / drive / MCC ] --(read-only: OPC UA read nodes / Modbus read registers)--+
 [ Retrofit sensors: suction pressure, wireless vibration+temp ] ----------------+
                                                                                  v
                              [ Edge box: 1-min aggregation -> healthy twin -> diagnosis -> alert policy ]
                                        local buffer, works offline, no write path
                                                  | one-way (MQTT-style JSON, DMZ / data diode)
                                                  v
                              [ Historian / this console / maintenance planner queue ]  advisory only
""", language="text")
    fp = os.path.join(config.RESULTS, "edge_footprint.json")
    if os.path.exists(fp):
        with open(fp) as fh:
            st.json(json.load(fh))
    st.markdown("Replay as an MQTT-style stream (read-only): `python -m retrofit_sim replay --data pack/data/full --asset PUMP-08 "
                "| python -m retrofit_ai.edge --asset PUMP-08`")

# ---------------------------------------------------------------- performance
with tab_perf:
    for name, title in (("metrics_table.md", "Measured detection quality vs. baselines"), ("sensor_value.md", "What each sensor level buys"),
                        ("limitations.md", "Limitations")):
        p = os.path.join(config.RESULTS, name)
        if os.path.exists(p):
            st.subheader(title)
            with open(p) as fh:
                st.markdown(fh.read())

if play and hours < 168:
    time.sleep(0.8)
    st.session_state.hours = min(168.0, hours + 1.0)
    st.rerun()
