#!/usr/bin/env bash
# Fast path for judges (~5 min incl. install): uses the shipped trained model + policy (models/) and the organizer data
# (not included in the zip - copy the `data/` folder of RETROFIT_AI_PARTICIPANTS_2026.zip to pack/data/ first).
# Rebuilds the 1-min features of the 12 official pumps, re-creates the validation score and the test submission,
# runs the causality audit and prepares the operator console. The full pipeline (synthetic data generation and
# training, ~2 h) is ./run_all.sh.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f pack/data/full/assets.csv ]; then
  echo "Missing organizer data: unzip RETROFIT_AI_PARTICIPANTS_2026.zip and copy its data/ folder to pack/data/" >&2
  exit 1
fi
PY=${PY:-.venv/bin/python}
if [ ! -x "$PY" ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt
  .venv/bin/python -m pip install -q -e pack --no-deps
  PY=.venv/bin/python
fi
echo "== features for the 12 official pumps"
$PY scripts/build_feature_cache.py --official --workers 4
echo "== validation score (official scorer) and test submission"
$PY scripts/predict.py --split validation
$PY scripts/predict.py --split test
echo "== causality audit (re-runs on truncated telemetry)"
$PY scripts/causality_check.py --cuts 2
echo "== console data"
$PY scripts/build_console_data.py
echo "done. Open the operator console with:  $PY -m streamlit run app/console.py"
