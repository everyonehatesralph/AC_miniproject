"""
run_pipeline.py -- run the complete Group 3 pipeline, from raw recordings + metadata to the
validated final report.

Usage (from the project root, inside the project virtual environment):
    .venv/Scripts/python.exe run_pipeline.py              # everything
    .venv/Scripts/python.exe run_pipeline.py --skip-fetch # recordings already in data/raw
    .venv/Scripts/python.exe run_pipeline.py --from 05    # resume from a step
    .venv/Scripts/python.exe run_pipeline.py --from 01b   # after adding listening-review answers
    .venv/Scripts/python.exe run_pipeline.py --quick      # smoke test (2 CV repeats, 20 permutations)

Each step is a separate script in src/ (see docs/DOCUMENTATION.md). The run stops at the first
failing step and reports which one failed; every step also writes its own log
(reports/logs/; steps 01 and 02 log to reports/phase0/ and reports/phase1/).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
STEPS = [
    ("00", "00_fetch_recordings.py", "fetch recordings from Google Drive (verified, idempotent)"),
    ("00", "00_extract_metadata_pdf.py", "metadata PDF -> verbatim CSV"),
    ("01", "01_audit_data.py", "Phase 0: data audit"),
    ("01b", "01b_merge_listening_reviews.py", "blind listening review: answers -> agreement -> decisions"),
    ("02", "02_clean_metadata.py", "Phase 1: metadata cleaning"),
    ("03", "03_prepare_audio.py", "Phases 2-5: matching, duplicates, WAV conversion, naming, quality report"),
    ("03b", "03b_build_listening_review.py", "listening-review page (data/review/listening_review.html)"),
    ("03c", "03c_audit_supplied_audio.py", "audit of audio supplied later (reports/supplied_audio_audit.md)"),
    ("04", "04_extract_features.py", "Phase 6: acoustic features"),
    ("05", "05_eda.py", "Phase 7: exploratory analysis"),
    ("06", "06_train_models.py", "Phases 8-11: speaker-independent training"),
    ("07", "07_evaluate_models.py", "Phase 12: evaluation"),
    ("08", "08_lexical_control.py", "Phase 13: lexically controlled analysis"),
    ("09", "09_build_final_report.py", "Phases 14-19: final group research report"),
    ("09b", "09b_build_html_report.py", "browser version of all reports (reports/html/index.html)"),
    ("09c", "09c_build_presentation.py", "results presentation (reports/presentation/)"),
    ("10", "10_validate_project.py", "Phase 20: validation"),
    ("11", "11_build_submission.py", "submission package (submission/G3_submission_<date>.zip)"),
]


def supplied_audio() -> str:
    """The supplied delivery that data/audio_source.json selects ('' = the original recordings)."""
    cfg = ROOT / "data" / "audio_source.json"
    if not cfg.is_file():
        return ""
    data = json.loads(cfg.read_text(encoding="utf-8"))
    return data.get("delivery", "") if data.get("source") == "supplied" else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skip-fetch", action="store_true", help="do not contact Google Drive")
    ap.add_argument("--from", dest="start", default="00", help="first step to run: 00, 01, 01b, 02, 03, 03b, 03c, 04 ... 09, 09b, 09c, 10, 11")
    ap.add_argument("--quick", action="store_true", help="smoke test: fewer CV repeats/permutations in step 06")
    args = ap.parse_args()
    t_all = time.time()
    delivery = supplied_audio()
    if delivery:
        print(f"Audio source: the supplied delivery {delivery} (data/audio_source.json); the original recordings are "
              "not downloaded or analysed.", flush=True)
    for num, script, label in STEPS:
        if num < args.start or ((args.skip_fetch or delivery) and script == "00_fetch_recordings.py"):
            continue
        cmd = [sys.executable, script] + (["--quick"] if args.quick and script == "06_train_models.py" else [])
        print(f"\n=== step {num}: {label} ({script}) ===", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=SRC)
        if proc.returncode != 0:
            print(f"\nFAILED at step {num} ({script}), exit code {proc.returncode}. See reports/logs/.", flush=True)
            return proc.returncode
        print(f"=== step {num} done in {time.time() - t0:.0f} s ===", flush=True)
    print(f"\nPipeline finished in {time.time() - t_all:.0f} s. Start with README.md and reports/.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
