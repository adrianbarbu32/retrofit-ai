#!/usr/bin/env bash
# Reproduce everything from the organizer data pack: synthetic training data, features, models, tuned alert policy,
# validation score, test submission, causality audit, report, figures, deck and docs. ~2 h on an 8-core laptop.
#   ./run_all.sh            full pipeline
#   ./run_all.sh --quick    reuse generated data under build/ if present
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
if [ ! -x "$PY" ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt
  .venv/bin/python -m pip install -q -e pack --no-deps
fi
W=${WORKERS:-6}

echo "== 1. organizer baselines (reference numbers)"
$PY pack/baselines/run_baselines.py --data pack/data/full --out results/baselines

if [ "${1:-}" != "--quick" ] || [ ! -d build/synth ]; then
  echo "== 2. synthetic training data from the organizer simulator"
  # 16 x 25 general pump-weeks (b1000-b1015; b1013-b1015 are the tuning hold-out)
  $PY synth/run_factory.py --batches 16 --per-batch 25 --workers "$W" --first-seed 1000
  # 8 x 25 pump-weeks focused on control/sensor combinations rare in the official training split (b2000-b2007;
  # b2006 hits a simulator edge case - an alarm raised on the last sample - and is skipped)
  $PY synth/run_factory.py --batches 8 --per-batch 25 --workers "$W" --first-seed 2000 --focus || true
  echo "== 2b. value-of-instrumentation variants (same pumps and faults at T0/T1/T2)"
  $PY synth/make_tier_variants.py --src build/synth/fleet_b1015.yaml --seed 1015 --n 25 --out build/tiers
  $PY synth/make_tier_variants.py --src build/synth/fleet_b1013.yaml --seed 1013 --n 25 --out build/tiers_b1013
  $PY synth/make_tier_variants.py --src build/synth/fleet_b1014.yaml --seed 1014 --n 25 --out build/tiers_b1014
fi

echo "== 3. causal per-minute features"
TIERDIRS=""
for root in tiers tiers_b1013 tiers_b1014; do
  for t in T0_LEGACY T1_RETROFIT T2_FULL; do TIERDIRS="$TIERDIRS --dir build/$root/$t:${t}_$root"; done
done
$PY scripts/build_feature_cache.py --official --synthetic --workers "$W" $TIERDIRS

echo "== 4. train diagnosis models (500 synthetic pump-weeks + 7 official train pumps)"
$PY scripts/train_eval.py --synthetic-train b1000-b1012,b2000-b2005,b2007 --synthetic-holdout b1013-b1015

echo "== 5. tune the alert policy on the synthetic hold-out (two stages)"
$PY scripts/tune_policy.py --holdout b1013-b1015 --passes 1
$PY scripts/tune_policy.py --holdout b1013-b1015 --passes 1 --init models/policy.json

echo "== 6. predictions: validation (scored) and test (submission)"
$PY scripts/predict.py --split validation
$PY scripts/predict.py --split test

echo "== 7. audits, report, figures, deck, docs"
$PY scripts/causality_check.py
$PY scripts/build_console_data.py
$PY scripts/report.py --holdout b1013-b1015
$PY -m retrofit_sim replay --data pack/data/full --asset PUMP-08 --start 2025-09-09T00:00:00Z --end 2025-09-11T12:00:00Z \
  | $PY -m retrofit_ai.edge --asset PUMP-08 --every 30 > results/edge_alerts_PUMP-08.jsonl
$PY scripts/make_figures.py
$PY scripts/make_deck.py
$PY scripts/finalize_docs.py

echo "done. Console: $PY -m streamlit run app/console.py"
