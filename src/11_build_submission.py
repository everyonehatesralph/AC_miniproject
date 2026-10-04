"""
11_build_submission.py -- one folder and one zip with everything to submit.

    submission/G3_submission_<date>/   and   submission/G3_submission_<date>.zip   (not in Git)

Copies, never moves. Contents: the final report and slides, the browser version, every figure, the CSV outputs
(features, labelled features, metadata, manifest, quality, results) and the main documentation, with a
CONTENTS.txt that maps the protocol's 12 expected outputs to the files. No audio by default: --with-audio adds
the named WAV recordings (expected output 4), which are participants' voices, so share them only once consent
is documented.

Run it after the pipeline (it is step 11 of run_pipeline.py):
    .venv/Scripts/python.exe src/11_build_submission.py [--with-audio]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission"

# (folder in the package, files or glob patterns relative to the project root)
CONTENT = [
    ("01_report", ["reports/final_group_research_report.md", "reports/deliverable_status_matrix.csv",
                   "reports/presentation/G3_results_presentation.pptx", "reports/validation_report.md"]),
    ("02_open_in_browser", ["reports/html/*.html"]),
    ("03_data", ["data/acoustic_features.csv", "data/acoustic_features_with_labels.csv",
                 "data/acoustic_features_dictionary.csv", "data/metadata_cleaned.csv", "data/analysis_manifest.csv",
                 "data/processed_audio/file_mapping.csv"]),
    ("04_figures/eda", ["figures/eda/*.png"]),
    ("04_figures/evaluation", ["figures/evaluation/*.png"]),
    ("04_figures/lexical", ["figures/lexical/*.png"]),
    ("04_figures/quality", ["figures/qa/*.png", "figures/qa/retakes/*.png"]),
    ("05_results", ["reports/exploratory_analysis.md", "reports/model_evaluation.md",
                    "reports/lexical_control_analysis.md", "reports/data_quality_report.md",
                    "reports/supplied_audio_audit.md", "reports/listening_review_report.md"]),
    ("05_results/tables", ["reports/audio_quality_table.csv", "reports/eda/*.csv", "reports/evaluation/*.csv",
                           "reports/lexical/*.csv", "reports/supplied_audio_audit.csv"]),
    ("06_documentation", ["docs/DOCUMENTATION.md", "docs/data_dictionary.md", "docs/decision_log.md",
                          "docs/analysis_plan_pooled_study.md"]),
]
AUDIO = ("07_audio_named_recordings", ["data/processed_audio/recordings/*.wav"])


def contents_text(n_files: int, with_audio: bool) -> str:
    status = {}
    matrix = ROOT / "reports" / "deliverable_status_matrix.csv"
    if matrix.is_file():
        import csv
        with matrix.open(encoding="utf-8") as fh:
            status = {r["Deliverable"].split(".")[0]: r["Status"] for r in csv.DictReader(fh)}
    source = "the original phone recordings"
    cfg = ROOT / "data" / "audio_source.json"
    if cfg.is_file():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        if data.get("source") == "supplied":
            source = f"the {data['delivery']} delivery (group decision, {data.get('decided_on', '')})"
    rows = [("1", "40 valid participant observations", "03_data/analysis_manifest.csv"),
            ("2", "Consent documentation", "not part of this package (consent/consent_log.csv; signed forms stay with the group)"),
            ("3", "Coded research metadata", "03_data/metadata_cleaned.csv"),
            ("4", "Properly named audio recordings", "07_audio_named_recordings/ (with --with-audio); names in 03_data/file_mapping.csv"),
            ("5", "Data quality report", "05_results/data_quality_report.md, 05_results/tables/audio_quality_table.csv"),
            ("6", "Acoustic feature dataset", "03_data/acoustic_features.csv (+ _with_labels, _dictionary)"),
            ("7", "Exploratory analysis", "05_results/exploratory_analysis.md, 04_figures/eda/"),
            ("8", "Machine learning experiment", "01_report (section 4.9-8), 05_results/tables/ (evaluation)"),
            ("9", "Model evaluation", "05_results/model_evaluation.md, 04_figures/evaluation/"),
            ("10", "Interpretation of findings", "01_report/final_group_research_report.md section 10"),
            ("11", "Research limitations", "01_report/final_group_research_report.md section 11"),
            ("12", "Group research report", "01_report/final_group_research_report.md (+ slides)")]
    lines = ["Group 3 - Bisaya post-class speech and self-reported affect: submission package",
             f"Built {date.today().isoformat()} by src/11_build_submission.py from the pipeline outputs ({n_files} files).",
             f"Audio analysed: {source}.", "",
             "Start here: 02_open_in_browser/index.html (works offline), or 01_report/final_group_research_report.md.", "",
             "Expected outputs (protocol section XVIII) -> files:"]
    for num, name, where in rows:
        lines.append(f"  {num:>2}. {name} [{status.get(num, '?')}]: {where}")
    lines += ["", "Audio: " + ("included (participants' voices: share only within the course)." if with_audio else
                               "not included. Rebuild with --with-audio to add the named WAV recordings."), ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--with-audio", action="store_true", help="add the named WAV recordings (expected output 4)")
    args = ap.parse_args(argv)
    name = f"G3_submission_{date.today().isoformat()}"
    folder = OUT / name
    if folder.exists():
        shutil.rmtree(folder)
    copied, missing = 0, []
    for dest, patterns in CONTENT + ([AUDIO] if args.with_audio else []):
        for pattern in patterns:
            matches = sorted(ROOT.glob(pattern))
            if not matches and "*" not in pattern:
                missing.append(pattern)
            for src in matches:
                target = folder / dest / src.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                copied += 1
    (folder / "CONTENTS.txt").write_text(contents_text(copied, args.with_audio), encoding="utf-8")
    archive = OUT / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(OUT))
    print(f"{copied} files -> {folder.relative_to(ROOT)} and {archive.relative_to(ROOT)} "
          f"({archive.stat().st_size / 1e6:.1f} MB)" + (f"; missing: {missing}" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
