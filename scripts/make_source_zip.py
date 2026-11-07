"""Build the source-code ZIP for sinceai.app (<= 100 MB; no dependencies or data files).

  python scripts/make_source_zip.py  ->  submission_kit/RETROFIT_AI_source.zip
"""
from __future__ import annotations

import fnmatch
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "submission_kit", "RETROFIT_AI_source.zip")
EXCLUDE_DIRS = {".venv", "build", "__pycache__", ".git", "deck_png", "submission_kit", ".pytest_cache", ".streamlit"}
EXCLUDE_PATHS = ["pack/data", "pack/retrofit_sim.egg-info", "pack/src/retrofit_sim.egg-info"]
EXCLUDE_FILES = ["*.pyc", "results/operator_log.jsonl", "models/type_models_v*.joblib", "submission/predictions_test_v*.csv",
                 "BusinessTurku_SinceAI2026_FINAL_BUILDER_PUBLISH.zip", "*.egg-info"]


def included(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in EXCLUDE_DIRS for p in parts[:-1]):
        return False
    if any(rel == e or rel.startswith(e + "/") for e in EXCLUDE_PATHS):
        return False
    return not any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(parts[-1], pat) for pat in EXCLUDE_FILES)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n, raw = 0, 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")]
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, ROOT).replace(os.sep, "/")
                if not included(rel):
                    continue
                z.write(full, "RETROFIT_AI/" + rel)
                n += 1
                raw += os.path.getsize(full)
    size = os.path.getsize(OUT) / 1e6
    print(f"{OUT}: {n} files, {raw / 1e6:.1f} MB uncompressed, {size:.1f} MB zipped")
    if size > 100:
        sys.exit("ZIP exceeds the 100 MB limit")


if __name__ == "__main__":
    main()
