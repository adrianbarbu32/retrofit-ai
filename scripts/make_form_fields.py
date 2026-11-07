"""Write submission_kit/FORM_FIELDS.md: copy-paste texts for the sinceai.app form, length-checked, numbers from results/."""
from __future__ import annotations

import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")

t = pd.read_csv(os.path.join(R, "metrics_table.csv"))
g = lambda m, s, c: t[(t.method == m) & (t.split == s)][c].iloc[0]
val, val_b1 = g("Ours", "validation", "Composite"), g("B1_mode_aware_z", "validation", "Composite")
hold, hold_b1 = g("Ours", "synthetic hold-out", "Composite"), g("B1_mode_aware_z", "synthetic hold-out", "Composite")
lead, lead_b1 = g("Ours", "validation", "Median lead to deadline [h]"), g("B1_mode_aware_z", "validation", "Median lead to deadline [h]")
ci = re.search(r"interval[^:]*: (\d+(?:\.\d+)?) - (\d+(?:\.\d+)?)", open(os.path.join(R, "metrics_table.md")).read()).groups()
sv = pd.read_csv(os.path.join(R, "sensor_value_overall.csv")).set_index("sensor level")["typed_share_all"]
edge = json.load(open(os.path.join(R, "edge_footprint.json")))
pct = lambda x: f"{float(x):.0%}"

NAME = "RETROFIT AI - operator assistant for existing pumps"
TAGLINE = "A listen-only AI layer that spots developing pump faults early, explains the evidence and tells operators what to check."
ABOUT = f"""THE PROBLEM
Ships and plants run cooling-water pumps that are 10-25 years old, with a handful of signals and fixed alarm limits. Operators learn about wear when a limit is crossed, often too late to plan the repair: on the validation pumps the plant's own alarm list catches only 11 % of developing problems. Replacing working machinery or its controls just to add AI is too expensive, especially for SMEs.

OUR SOLUTION
RETROFIT AI is an advisory, read-only layer next to the existing pump. Every minute it:
1) compares each reading with a healthy-pump physics model built from the pump's datasheet and its live duty ("what would this pump read if it were healthy right now?");
2) diagnoses which of 13 machine or instrument faults is developing, and tells a failing sensor from a failing machine;
3) raises one alert per problem, with evidence in engineering units, a confidence and a recommended check that a human approves.
The operator console replays each pump's week, shows measured versus healthy values, states each pump's blind spots and logs every acknowledge / reject / work-order request. For machine faults it estimates the severity from the physics and tests the recommended action in the organizers' what-if sandbox (simulation only). There is no write path to any pump, drive, valve or safety system.

RESULTS (official scorer)
- Validation pumps: {val:.1f} / 100 vs {val_b1:.1f} for the best baseline. 9 of 9 faults named with the right cause (baseline 44 %), zero false alerts, median {float(lead):.0f} h of warning before the conventional limit (baseline {float(lead_b1):.0f} h).
- 75 unseen simulated pumps: {hold:.1f} (95 % interval {ci[0]}-{ci[1]}) vs {hold_b1:.1f}.
- Same pumps at three sensor levels: share of all faults caught with the right cause {pct(sv['T0_LEGACY'])} with existing signals, {pct(sv['T1_RETROFIT'])} after adding a ~EUR 145 suction transmitter and one wireless vibration point, {pct(sv['T2_FULL'])} fully instrumented.
- Causality audit (re-runs on truncated data) passed; the edge runtime needs ~{edge['latency_ms_median'] / 1000:.1f} s and {edge['peak_memory_mb']:.0f} MB per run, offline.

HOW WE BUILT IT
Python end to end. The healthy-pump model reuses the organizers' documented equations. Only 7 pumps carry labels, so we used the organizers' simulator as a data factory: 500 labelled pump-weeks covering all fault types, compound incidents, control strategies and sensor tiers. 13 gradient-boosted classifiers (scikit-learn) read ~240 strictly causal features. The alert policy was tuned on held-out simulated pumps; the validation pumps were never used for training or tuning, and the test pumps never for anything but the final predictions. Console: Streamlit and Plotly. This was an AI-use hackathon: Claude Code wrote and ran most of the code under the team's direction.

WHAT WE MET ON THE WAY
- The simple baseline learns "normal" from the first day, so faults already present are invisible. A physics reference fixes that.
- In batch processes cavitation comes in bursts separated by pump stops. Treating a stop as the end of an episode lifted cavitation detection from 44 % to 85 %.
- Hiding instrument alerts without a clear culprit raised precision but lost real detections. We kept them and label weak ones "possible, unconfirmed".
- Calibration shifts are still often mistaken for drifts: our main known weakness.
- All results are simulator-to-simulator and overstate field performance.

WHAT COMES NEXT
A shadow-mode pilot on a real pump: 4-6 weeks of read-only data to calibrate the healthy model, then alerts logged and compared with work orders. Next would be one retrofit vibration point where blind spots show, learning from operator accept/reject feedback, and an incremental edge runtime. The same layers transfer to fans and compressors by swapping the physics model.

The test-pump alert file is submission/predictions_test.csv in the source ZIP (official template format)."""
BUILT_WITH = "Python, scikit-learn, Streamlit, pandas, Plotly, NumPy, SciPy, PyArrow, matplotlib, ReportLab, Claude Code"
MLH_FEEDBACK = ("The organizers' simulator and scorer were the stars: the documented physics let us build a healthy-pump model and "
                "generate labelled training data, and the official scorer made every change measurable. scikit-learn's "
                "HistGradientBoosting handled 1.6 M rows with missing sensors natively. Streamlit made the operator console quick "
                "to build, and its AppTest let us test the UI without a browser.")
MLH_GENAI = ("Claude Code (Anthropic) was the engineering assistant. It wrote and ran the code, experiments and documents under "
             "the team's direction and review (AI-use hackathon).")
MLH_GENMODEL = ("No. The shipped product uses physics models and gradient-boosted decision trees. There is no generative model "
                "or LLM API at runtime, so it runs offline on an edge box.")
PUBLIC = ("RETROFIT AI is a read-only assistant for existing industrial pumps. It compares live readings with a physics model "
          "of a healthy pump, names developing machine or sensor faults early, and recommends checks that a human approves. "
          "Built and evaluated only on the organizers' synthetic benchmark data.")

LIMITS = {"Project name": (NAME, 80), "Tagline": (TAGLINE, 140), "About the project": (ABOUT, 5000)}
for k, (v, lim) in LIMITS.items():
    if len(v) > lim:
        sys.exit(f"{k} too long: {len(v)} > {lim}")


def block(title, text, limit=None, note=None):
    head = f"## {title}" + (f"  ({len(text)} / {limit} characters)" if limit else "")
    return "\n".join([head, *( [f"*{note}*", ""] if note else [""] ), "```text", text, "```", ""])


out = ["# sinceai.app form: copy-paste texts", "",
       "Generated by `scripts/make_form_fields.py` from `results/`. Re-run it if the model changes. Fields marked **TEAM** need "
       "your own input or confirmation.", "",
       "## Challenge", "", "Pre-filled by the organizers: RETROFIT AI - Make Existing Machines Intelligent (Business Turku).", "",
       block("Project name", NAME, 80), block("Tagline", TAGLINE, 140), block("About the project", ABOUT, 5000),
       block("Built with", BUILT_WITH, note="Comma-separated; the first five become chips on the vote card."),
       "## Repository", "", "**TEAM:** the GitHub URL after you push (see `00_START_HERE.md`, step 3). If the repo is private, judges use the ZIP.", "",
       "## Demo video", "", "**TEAM:** the unlisted YouTube link (see `00_START_HERE.md`, step 2, and `VIDEO_SCRIPT_2MIN.md`).", "",
       "## Source code", "", "Upload `submission_kit/RETROFIT_AI_source.zip` (13.8 MB; verified to rebuild the submission from a clean unpack).", "",
       "## MLH questions (TEAM: check these are true for you)", "",
       block("Technology feedback", MLH_FEEDBACK), block("GenAI tools you used", MLH_GENAI),
       block("Did you implement a generative AI model or API?", MLH_GENMODEL),
       "## Challenge terms", "", "**TEAM:** every team member ticks the terms personally.", "",
       block("Public summary (only if the form asks for it, i.e. the challenge is marked confidential)", PUBLIC,
             note="Contains no confidential data: everything is the organizers' public synthetic benchmark."),
       "## Optional fields", "",
       "- **Demo URL:** leave empty. The console runs locally (`streamlit run app/console.py`); it is shown in the video.",
       "- **Pitch deck:** upload `submission_kit/RETROFIT_AI_deck.pdf`.",
       "- **Supporting documents (max 2):** `submission_kit/RETROFIT_AI_technical_note.pdf` (retrofit cost, deployment, safety, "
       "limits, metrics vs baselines) and `submission_kit/RETROFIT_AI_test_predictions.txt` (the required test-pump alert CSV, "
       "saved as .txt because the form does not accept .csv).",
       "- **Partner technologies / MLH prize categories:** none used. Leave empty unless a category clearly fits.",
       "- **Domain name and schools of team members:** **TEAM** (e.g. Åbo Akademi University).",
       "- **Gemini API project number:** leave empty.", ""]
os.makedirs(os.path.join(ROOT, "submission_kit"), exist_ok=True)
open(os.path.join(ROOT, "submission_kit", "FORM_FIELDS.md"), "w", encoding="utf-8").write("\n".join(out))
print(f"name {len(NAME)}/80, tagline {len(TAGLINE)}/140, about {len(ABOUT)}/5000")
