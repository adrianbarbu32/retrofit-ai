# Submission checklist (deadline: Sun 8 Nov ~10:00)

| Deliverable (challenge brief section 7) | File | Status |
|---|---|---|
| Test-pump alert CSV (template format) | `submission/predictions_test.csv` | ready, regenerated with the final model |
| Source code + reproducible run instructions | repo root, `README.md`, `run_all.sh` | ready |
| Baseline / metrics comparison | `results/metrics_table.md`, `results/per_type_holdout.md` | ready |
| Technical note: retrofit cost, deployment, safety, limits | `docs/TECHNICAL_NOTE.md` | ready |
| Concise PDF presentation | `docs/RETROFIT_AI_deck.pdf` | ready |
| Short narrated demo video | recorded by the team, script in `docs/DEMO_SCRIPT.md` | **team** |
| Working prototype | `streamlit run app/console.py` | ready |
| Optional: what-if comparison, transfer | console what-if tab; `docs/TECHNICAL_NOTE.md` section 5 | ready |

## Team actions

1. **Discord question to the organizers.** Suggested text:
   > Hi! RETROFIT AI team here. We trained our diagnosis models on extra synthetic data that we generated
   > ourselves with the provided simulator (`make_dataset.py` with our own fleet configs and seeds; pump ratings
   > sampled from documented ranges and around the public `assets.csv` datasheets). We never used the test pumps'
   > telemetry for training or tuning. Can you confirm this is within the rules (BENCHMARK_RULES §2 says the
   > simulator source code is explicitly permitted)?
2. **Record the demo video.**
   - Start the console with `.venv/bin/python -m streamlit run app/console.py` and follow `docs/DEMO_SCRIPT.md`.
   - Screen recorder: OBS, or the Xbox Game Bar (Win+G) on Windows/WSL.
   - Aim for 3-4 min.
3. **Repository.** Create a GitHub repo (or tell Claude to, if `gh` is logged in) and push.
   - Large generated data stays out: `build/` and `pack/data/` are in `.gitignore`.
   - The organizer pack is re-downloadable, and `run_all.sh` regenerates everything else.
4. **Upload on sinceai.app:**
   - repo link (or zip)
   - deck PDF
   - video
   - `submission/predictions_test.csv`
   - technical note (the markdown file, or its PDF if the form wants a PDF)
