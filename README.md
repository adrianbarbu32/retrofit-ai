# RETROFIT AI: an operator assistant for pumps that already exist

*Since AI Hackathon 2026 (Turku). Challenge by Business Turku, technical sparring by Elomatic.*

An affordable intelligence layer for in-service cooling-water pumps. It reads the signals the pump already has, or a
few cheap retrofit sensors. It notices developing machine and instrument problems early, explains the evidence in
engineering units, states its confidence, and recommends what a human should check.
**It only listens: there is no write path to the pump, drive, valves or safety system.**

> All data are the organizers' synthetic benchmark (`pack/`). Results show the method works on a physically
> coherent simulator. They are not evidence of field performance.

## Results (official scorer)

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

Full tables: [`results/metrics_table.md`](results/metrics_table.md), [`results/per_type_holdout.md`](results/per_type_holdout.md),
[`results/sensor_value.md`](results/sensor_value.md), [`results/business_case.md`](results/business_case.md).
Test-pump submission: [`submission/predictions_test.csv`](submission/predictions_test.csv) (official template format).

## How it works

```
 telemetry (1 Hz, read-only) --> 1-min aggregates --+
 operating context (duty, inlet temp, level, state) -+--> healthy-pump twin --> residuals (measured - healthy)
 pump datasheet (assets.csv) ------------------------+          |
 vibration snapshots (every 4 h) --> spectra (BPFO, 1x, 2x) ----+--> causal features (trend, step/ramp, noise,
                                                                     stuck, gaps, NPSH margin, consistency checks)
                                                                              |
                              13 fault-type models (trained on simulator data) v
                                                                    alert policy (persist, latch, re-arm, spacing)
                                                                              |
                         evidence + confidence + alternative + recommended action (human approves)
                                    -> operator console / MQTT-style alert topic (advisory only)
```

1. **Healthy twin** (`retrofit_ai/twin.py`). The documented pump physics, evaluated minute by minute with a healthy
   condition under the observed duty: pump curve, affinity laws, system curve, motor and drive losses, NPSH,
   thermal lags and vibration components. "What would this pump read if it were healthy right now?" Because it
   relies on physics rather than the first day of data, problems already present at start are visible.
2. **Features** (`retrofit_ai/features.py`). Strictly backward-looking: residual level, change since baseline,
   1-12 h trends, step-vs-ramp shape, scatter and spikes, frozen readings, data gaps, measured NPSH margin,
   current-vs-power and pump-curve consistency, and waveform spectra.
3. **Diagnosis** (`retrofit_ai/model.py`). One gradient-boosted classifier per fault type, trained on
   **500 synthetic pump-weeks generated with the organizers' simulator** (`synth/`; 175 of them focused on control/sensor combinations that are rare in the official training split) plus the 7 official training
   pumps. The synthetic set covers all 13 fault types, the documented compound incidents (e.g. a drifting suction
   transmitter masking a clogging strainer), every control strategy and every sensor tier.
4. **Alert policy** (`retrofit_ai/policy.py`). One alert per developing problem. It re-arms after a repair, keeps
   alerts at least 6 min apart, and has a hard rule for data loss. Thresholds are tuned on synthetic hold-out pumps
   never used for training.
5. **Explanations and actions** (`retrofit_ai/explain.py`, `retrofit_ai/severity.py`). Evidence text,
   instrument-vs-machine reasoning, an alternative hypothesis, and the recommended human check. A physics-based
   severity estimate feeds the organizers' what-if sandbox (simulation only).

## Run it

**Data.** The organizer data is not included (size and licence). Unzip `RETROFIT_AI_PARTICIPANTS_2026.zip` from the
challenge pack and copy its `data/` folder to `pack/data/`, so that `pack/data/full/assets.csv` exists. `pack/` holds
the unchanged organizer code (simulator, scorer, baselines).

**Fast path, about 5 min.** This uses the shipped trained model and alert policy in `models/`. It rebuilds the
features of the 12 official pumps, re-creates the validation score and `submission/predictions_test.csv`, runs the
causality audit and prepares the console:

```bash
./quickstart.sh
.venv/bin/python -m streamlit run app/console.py      # operator console at http://localhost:8501
```

**Full pipeline, about 2 h on 8 cores.** It regenerates the 500 synthetic training pump-weeks, retrains, re-tunes,
and rebuilds the report, figures, deck and docs:

```bash
./run_all.sh
```

| Step | Command |
|---|---|
| Synthetic training data (organizer simulator) | `python synth/run_factory.py --batches 16 --per-batch 25` and `... --batches 8 --first-seed 2000 --focus` |
| Features | `python scripts/build_feature_cache.py --official --synthetic` |
| Train models (+ hold-out evaluation) | `python scripts/train_eval.py --synthetic-train b1000-b1012,b2000-b2005,b2007 --synthetic-holdout b1013-b1015` |
| Tune alert policy on hold-out | `python scripts/tune_policy.py --holdout b1013-b1015` |
| Validation score / test submission | `python scripts/predict.py --split validation` / `--split test` |
| Report (tables, sensor value, business case) | `python scripts/report.py` |
| Causality audit (truncated re-runs) | `python scripts/causality_check.py` |
| Operator console | `streamlit run app/console.py` |
| Edge demo (read-only stream) | `python -m retrofit_sim replay --data pack/data/full --asset PUMP-08 --limit 86400 \| python -m retrofit_ai.edge --asset PUMP-08 --every 30` |

## Repository

| Path | Content |
|---|---|
| `retrofit_ai/` | engine: twin, features, labels, models, policy, explanations, severity estimate, edge runtime |
| `synth/` | synthetic fleet sampler and factory (organizer simulator, unchanged) |
| `scripts/` | feature cache, training, tuning, prediction, report, causality check |
| `app/console.py` | operator console (Streamlit) |
| `docs/` | technical note (retrofit cost, deployment, safety, limits), pitch deck, demo script |
| `results/`, `submission/` | measured results, test-pump alert CSV |
| `pack/` | organizer data pack (unchanged) |

## Safety and scope

- Advisory only. Existing alarms, trips and interlocks remain authoritative. No commands to devices.
- What-if actions are evaluated in the organizer sandbox (simulation) and never applied.
- Test pumps (PUMP-10/11/12) were never used for tuning or labelling. Their telemetry is only read to produce the
  submitted alerts.
- Limitations: see [`docs/TECHNICAL_NOTE.md`](docs/TECHNICAL_NOTE.md) section 6.
