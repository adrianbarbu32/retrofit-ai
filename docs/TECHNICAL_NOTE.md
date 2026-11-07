# RETROFIT AI: technical note

**Team submission · Since AI Hackathon 2026 · Challenge: Business Turku, technical sparring: Elomatic**

All data are the organizers' synthetic benchmark. Nothing here is a field result, a certified system or a proven saving.

## 1. What we built

The engine is an operator-assist layer that sits next to an existing pump and only listens. Every minute it:

1. **Compares every reading with a healthy-pump model.** The model is the documented pump physics: pump curve, affinity laws, system curve, motor and drive losses, thermal lags, NPSH and vibration components. It is parameterised from the public datasheet (`assets.csv`) and fed the live operating context: duty demand, inlet temperature, room temperature, suction level and run state. The difference between what a healthy pump *would* read and what this pump *does* read is the evidence. Because the reference is physics, not "the first day of data", faults that already exist when monitoring starts are visible from the first hours.
2. **Diagnoses the cause.** Thirteen gradient-boosted classifiers (one per fault type) read the residual patterns: level vs. model, change since baseline, 1-12 h trends, step-vs-ramp shape, noise and spikes, frozen readings, data gaps, measured NPSH margin, electrical consistency (current vs. power), pump-curve consistency (head at the measured flow) and waveform spectra (BPFO envelope, 1x/2x running speed). They are trained on **500 synthetic pump-weeks we generated with the organizers' own simulator** (175 of them focused on the control / sensor combinations that are rare in the official training split): all 13 fault types, the documented compound incidents, every control strategy and sensor tier. The 7 official training pumps are added to that set.
3. **Decides whether to alert.** A causal policy smooths the probabilities and requires persistence. It raises **one alert per developing problem**, re-arms after a repair (lockout) or a long quiet period, keeps alerts at least 6 min apart, and has a hard rule for data loss. Thresholds were tuned on synthetic hold-out pumps that were never used for training.
4. **Explains and recommends.** Each alert carries:
   - its class (machine or instrument) and fault type;
   - a confidence;
   - evidence in engineering units (e.g. "suction pressure -12 kPa vs healthy model, speed +2.1 %");
   - an alternative hypothesis when one is plausible;
   - a recommended check for a human to approve.

   For machine faults the console estimates the severity by fitting the latent condition (clog factor, restriction, wear) to the readings. A sandbox what-if then tests the recommended intervention on the simulator only.

## 2. Measured results

<!-- RESULTS:START -->
**Official validation pumps (PUMP-08/09, 9 events; never used for training or tuning)**

| Method | Composite | Event F1 | Fault-type recall | Class recall | Timeliness | False+excess /asset-day |
|---|---|---|---|---|---|---|
| **RETROFIT AI (ours)** | 98.3 | 1.0 | 1.0 | 1.0 | 0.91 | 0.0 |
| B1 mode-aware z | 78.0 | 0.89 | 0.44 | 0.89 | 0.86 | 0.07 |
| B3 isolation forest | 69.2 | 0.74 | 0.44 | 0.78 | 0.78 | 0.21 |
| B2 CUSUM | 50.9 | 0.45 | 0.33 | 0.67 | 0.77 | 1.07 |
| B0 plant alarms | 23.1 | 0.2 | 0.11 | 0.11 | 0.11 | 0.0 |

**75 unseen synthetic pump-weeks (334 events; used only to tune alert thresholds)**

| Method | Composite | Event F1 | Fault-type recall | Class recall | Timeliness | False+excess /asset-day |
|---|---|---|---|---|---|---|
| **RETROFIT AI (ours)** | 88.6 | 0.91 | 0.84 | 0.9 | 0.86 | 0.06 |
| B1 mode-aware z | 55.9 | 0.67 | 0.28 | 0.53 | 0.54 | 0.15 |
| B0 plant alarms | 27.3 | 0.29 | 0.08 | 0.2 | 0.18 | 0.11 |

- Hold-out composite, 95 % bootstrap interval over pumps: 86.4-90.7.
- Same simulated pumps and faults at three sensor levels: share of *all* faults caught with the correct cause T0 56% -> T1 80% -> T2 88% (`results/sensor_value.md`).
- Causality audit (re-runs on truncated data): PASSED on 10 cut points.
- Edge run on the organizer replay stream: median 259.3 ms per diagnosis run, 341.9 MB peak memory (laptop CPU core).
<!-- RESULTS:END -->

## 3. Reliability and safety

- **No write path.** The engine consumes telemetry and context and emits advisory messages. There is no code path to a PLC, drive, valve or safety system. The what-if sandbox evaluates proposals on the simulator only and checks them against duty, NPSH, motor-load and vibration limits. Its output is not a set-point.
- **Human in the loop.** Operators acknowledge alerts, reject them as false alarms, or turn them into a work-order *request*. Every decision is logged (`results/operator_log.jsonl`) for shadow-mode review.
- **False alarms.**
  - Machine diagnoses run only in steady operation, at least 10 min after a start.
  - Transient context (manoeuvring, shift changes, set-point changes, heavy weather, voltage dips) is part of the model input.
  - The policy latches, so an ongoing problem is not re-announced.
  - Measured rate: see the false + excess alerts per asset-day in `results/metrics_table.md`.
- **Instrument vs. machine.** A sensor fault changes one reading while the physically coupled readings still agree with the healthy model. A machine fault moves several readings in the pattern the physics predicts. The evidence text states which case it is, so a drifting gauge does not trigger a needless repair.
- **Missing sensors.** The same models run on legacy (T0), retrofit (T1) and full (T2) instrumentation. Absent channels are simply missing inputs. `results/sensor_value.md` shows, on identical simulated pumps, which faults each sensor level can and cannot reveal. This is an explicit blind-spot statement: no vibration sensor means no bearing, imbalance or misalignment alerts.
- **Uncertainty.** Confidence is the smoothed class probability at alert time, and the second-most-likely cause is reported when it is above 20 %.
- **Causality.** Every feature is a backward-looking window. `scripts/causality_check.py` re-runs the engine on telemetry truncated at random cut points and verifies the alerts before each cut are identical.

## 4. Retrofit feasibility, cost and deployment

### 4.1 What to install, per sensor level

| Level | Hardware added to an existing pump | Indicative hardware cost | What becomes visible (measured on identical simulated pumps, `results/sensor_value.md`) |
|---|---|---|---|
| T0, existing signals | nothing: read motor current (MCC/drive), discharge pressure and drive speed/power read-only | edge box only: ~EUR 500-600 (Revolution Pi Connect 4 EUR 497 / SIMATIC IOT2050 EUR 602, list) | hydraulic faults on drives (restriction, impeller wear, clogging via speed/power), frozen / drifting / offset transmitters, data loss |
| T1, low-cost retrofit | + suction pressure transmitter (~EUR 145, WIKA A-10 list) + one wireless vibration & temperature point on the pump bearing (EUR 300-1,500 per point + gateway EUR 1-3 k, assumption) | ~EUR 1.0-2.5 k per pump (gateway shared across pumps) | + clogging and cavitation from the suction side (measured NPSH margin), bearing damage from stage 2 (acceleration, crest factor, envelope spectrum), imbalance, misalignment |
| T2, full | + wired IO-Link vibration (ifm VVB001 ~EUR 530), energy meter (EUR 430-630), clamp-on flow (EUR 0.2-10 k), PT100 winding | ~EUR 2.1-3.1 k (organizer BOM options A/B) + flow | + pump-curve consistency (head at measured flow), efficiency, faster and better instrument-vs-machine separation |

Prices are the list prices collected in `pack/RETROFIT_FEASIBILITY.md` (30 Oct 2026, ex VAT unless stated), not quotes.
Cabling, enclosure, installation labour and, for ships, class approval come on top.

**Recommendation.** Start every pump at T0: it costs only a share of an edge box and already catches the hydraulic
and instrument problems. Add the T1 pair (suction pressure + one wireless vibration point) on critical pumps without
redundancy, where bearing, imbalance and cavitation risks pay for it. T2 is for pumps whose energy use justifies flow
and power metering anyway.

### 4.2 Edge deployment

- One edge box per pump room. It aggregates 1 Hz readings into 1-min windows (mean, std, min, max, count), runs the
  twin, features, models and policy, and buffers locally for days, so monitoring keeps working offline at sea.
  Measured on a laptop core: see `results/edge_footprint.json` (latency per diagnosis run, peak memory). The model
  file is a few MB of trees, with no GPU and no cloud dependency.
- Integration is read-only: OPC UA read nodes, Modbus read registers, or the drive's fieldbus mirror. Alerts leave
  over a one-way conduit (MQTT-style JSON, DMZ or data diode) to the historian and the console. The engine has no
  write client of any kind. This follows IEC 62443 zones/conduits, NIST SP 800-82r3 and IEC 61511 independence of the
  safety layer. On ships (IACS UR E26/E27) the box sits outside the safety zones, or one-way, and is recorded in the
  asset inventory and management of change.
- Inputs it needs besides telemetry: the pump datasheet (curve points, NPSH3, motor rating, bearing type) and, where
  available, the duty signal (heat load / batch set-point / header pressure set-point) and inlet temperature from the
  ship or plant automation. All are read-only.

### 4.3 Pilot plan

1. Weeks 1-6, read-only data: connect to existing signals and check the twin residuals on the real pump. Replace
   profile defaults (motor efficiency, vibration base levels) with commissioning values. No alerts to operators.
2. Weeks 6-12, shadow mode: alerts are logged and compared afterwards with operator logs, alarms and work orders
   (precision, lead time, nuisance rate per pump-week). The model is re-trained on simulated faults with the real
   pump's parameters.
3. Add one T1 retrofit point where shadow mode shows blind spots (e.g. a critical seawater pump without vibration
   sensing).
4. Advisory pilot: alerts go to a named role with the evidence view and accept/reject buttons. Closing a false
   alert is defined in advance.
5. Review: scale up on measured value (avoided unplanned stops and repairs, minus nuisance handling), with an
   independent engineering and regulatory review before any operational use.

### 4.4 Business case (illustrative, synthetic)

`results/business_case.md` re-runs the organizer's worked example (`pack/examples/business_case.py`) with our
measured validation recall and false-alert rate. All other inputs remain the organizer's stated assumptions.


## 5. Transfer to other rotating machines

The engine has three layers. Only the first is pump-specific:

1. **Healthy twin.** Machine-class physics plus datasheet: "what should it read now?"
2. **Residual features.** Generic: level, trend, step/ramp shape, scatter, frozen, gaps, cross-channel consistency.
3. **Diagnosis and policy.** Trained on simulated faults of that machine class.

What changes for another rotating machine:

| Machine | Twin (layer 1) | Fault physics to simulate for training (layer 3) | Reused as is |
|---|---|---|---|
| Centrifugal fans / blowers (cooling towers, HVAC, engine-room ventilation) | the same affinity laws (Q ~ n, dp ~ n^2, P ~ n^3), fan curve + damper / system curve | bearing (BPFO), imbalance (fouled blades), misalignment, belt slip, filter clogging (= strainer analogue) | all features, policy, console, edge runtime, sensor-fault detectors |
| Centrifugal compressors | compressor map + surge line instead of pump curve | fouling (head/efficiency loss = impeller-wear analogue), surge margin (= NPSH-margin analogue), bearing, seal gas | same |
| Electric motors (any driven load) | load-dependent current, power factor, winding temperature (already in the pump model) | winding insulation ageing, cooling-fan blockage, bearing, supply unbalance | same |
| Gearboxes / conveyors | load-speed torque model | gear-mesh frequency (replaces vane pass) and sidebands, bearing, lubrication temperature | same |

The pump case already exercises the transfer mechanism. The models are trained on randomly sized pumps (30-750 m3/h,
5.5-90 kW, 2- and 4-pole, four control strategies, three sensor levels) and evaluated on pumps and seeds they never
saw (the synthetic hold-out and the official validation pumps). The test pumps include sensor/control combinations
absent from the official training set (VFD-pressure on legacy sensors; flow-valve on full sensors); our generated
training data covers them.


## 6. Limitations

- **Synthetic only, sim-to-sim.** The training data, validation and test pumps come from the same simulator family.
  Our measured scores therefore overstate field performance. Real pumps have unmodelled faults (seal leaks, electrical
  rotor/stator faults, water hammer) and noise the simulator does not produce.
- **Twin parameters.** Unknown per-asset values (exact motor efficiency, vibration base levels, controller details)
  use documented profile defaults. A real deployment needs a commissioning baseline (pilot step 1).
- **Labels.** "Detectable" and "deadline" times are simulator constructs (3-sigma of a 10-min mean in a
  counterfactual run). Our early-warning numbers are relative to them.
- **Small official validation set.** 9 events on 2 pumps, so a single event moves the composite by about 5 points.
  We therefore also report a 75-pump synthetic hold-out with a bootstrap interval.
- **Limited by what is measured.** No vibration sensor means no bearing, imbalance or misalignment alerts. Without
  suction pressure, cavitation and clogging are weaker on fixed-speed pumps. The console states this.
- **Advisory and uncertified.** No class approval, no certified safety function, no guaranteed fault
  identification, no proven savings. The cost figures are list prices and assumptions.
- **Compute.** The edge runtime recomputes the full history each run for simplicity. An incremental version would
  be needed on small hardware at 1-min cadence over months.

