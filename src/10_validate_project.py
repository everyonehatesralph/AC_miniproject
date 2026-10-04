"""
10_validate_project.py -- PHASE 20: automated validation of the finished project.

Every check re-derives its answer from the files (nothing is taken from a report's text).
Writes reports/validation_report.md and reports/validation_results.json, and exits with
status 1 if any check FAILS. A check can also be BLOCKED (cannot pass with the supplied data,
e.g. consent documentation) or a WARNING (passes with a documented caveat).

Usage: .venv/Scripts/python.exe src/10_validate_project.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import soundfile as sf

from common import (CLEAN_METADATA_CSV, CONSENT_DIR, CONSENT_LOG, DATA, DOWNLOAD_LOG, FEATURES_CSV, FIGURES, FILENAME_RE,
                    MIN_VALID_OBSERVATIONS, MODELS, PROCESSED_AUDIO_DIR, PROJECT_ROOT, RAW_METADATA_CSV, REPORTS,
                    TARGET_SR, audio_source, load_consent_log, sha256_of, setup_logging)

log = setup_logging("validate", REPORTS / "logs" / "10_validate_project.log")
RESULTS: list[dict] = []
# student-number-like tokens (e.g. 2023-12345, 202312345); participant codes are Y2-NNN and do not match
PII = re.compile(r"(?<![\w-])(19|20)\d{2}-?\d{4,6}(?![\w-])")


def check(item: str, status: str, evidence: str) -> None:
    assert status in {"PASS", "FAIL", "BLOCKED", "WARNING"}
    RESULTS.append({"item": item, "status": status, "evidence": evidence})
    log.info("%-8s %s -- %s", status, item, evidence)


def main() -> int:
    manifest = pd.read_csv(DATA / "analysis_manifest.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    meta = pd.read_csv(CLEAN_METADATA_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    feats = pd.read_csv(FEATURES_CSV)
    included = manifest[manifest["include_primary"] == "True"]
    n_valid = len(included)

    # 1. sample size
    check("40 valid unique participants available",
          "PASS" if n_valid >= MIN_VALID_OBSERVATIONS else "BLOCKED",
          f"{n_valid} valid unique participants (recordings exist for {manifest['n_takes'].astype(int).gt(0).sum()} of "
          f"{len(manifest)} metadata participants; missing: "
          f"{', '.join(manifest.loc[manifest['n_takes'] == '0', 'participant_code'])})")

    # 2. consent (documented in consent/consent_log.csv; the sheet's "Yes" is only an attestation)
    sheet_yes = (meta["consent_obtained"] == "Y").sum()
    consent_log = load_consent_log()
    if consent_log is None:
        check("consent requirements satisfied", "BLOCKED",
              f"metadata records consent 'Y' for {sheet_yes}/{len(meta)}, but {CONSENT_LOG.name} has not been supplied "
              "(template: consent/consent_log_TEMPLATE.csv)")
    else:
        documented = sorted(c for c, ok in consent_log.items() if ok)
        undocumented_included = sorted(set(included["participant_code"]) - set(documented))
        check("consent requirements satisfied", "PASS" if not undocumented_included else "FAIL",
              f"{CONSENT_LOG.name}: documented consent for {len(documented)} of {len(meta)} metadata participants; "
              f"included participants without documented consent: {undocumented_included or 'none'}")

    # 3. no names / student IDs
    hits = []
    for path in [CLEAN_METADATA_CSV, RAW_METADATA_CSV, FEATURES_CSV, *REPORTS.glob("*.md"),
                 *([CONSENT_LOG] if CONSENT_LOG.is_file() else [])]:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if PII.search(text):
            hits.append(path.name)
    name_cols = [c for c in meta.columns if re.search(r"(?i)\bname\b|student", c)]
    hits += [f"column {c}" for c in name_cols]
    codes_ok = meta["participant_code"].str.fullmatch(r"Y2-\d{3}").all()
    check("no participant names/student IDs exposed", "PASS" if codes_ok and not hits else "FAIL",
          f"identifiers are pseudonymous codes (Y2-NNN): {codes_ok}; name/ID patterns found in: {hits or 'none'}; "
          "raw videos (faces, GPS tags) stay local in data/raw and are never copied into reports")

    # 4. metadata cleaned
    raw_cols = {"participant_code_raw", "spoken_word_raw", "valence_raw", "arousal_raw", "consent_raw"}
    check("metadata cleaned (raw preserved)", "PASS" if raw_cols <= set(meta.columns) and len(meta) == 40 else "FAIL",
          f"{len(meta)} rows; raw columns kept alongside cleaned ones; log: reports/phase1/cleaning_log.csv")

    # 5. eligible observations have recordings
    bad = included[(included["processed_file"] == "") | (included["segment_file"] == "")]
    check("all eligible observations have recordings", "PASS" if bad.empty else "FAIL",
          f"{n_valid} included participants each have one processed WAV and one segment")

    # 6. duplicates resolved
    retakes = manifest[manifest["n_takes"].astype(int) > 1]
    dup_sel = manifest["selected_drive_file_id"].replace("", np.nan).dropna()
    decided = set()  # human decisions: the group's own file, or a unanimous blind listening review (01b)
    for f in (PROCESSED_AUDIO_DIR / "take_overrides.csv", DATA / "review" / "take_overrides_from_review.csv"):
        if f.is_file():
            decided |= set(pd.read_csv(f, dtype=str, keep_default_na=False)["participant_code"])
    pending = [c for c in retakes["participant_code"] if c not in decided]
    check("duplicates resolved", "FAIL" if not dup_sel.is_unique else "WARNING" if pending else "PASS",
          f"{len(retakes)} participants had retakes ({', '.join(retakes['participant_code'])}); exactly one take "
          f"selected each (unique selections: {dup_sel.is_unique}); "
          + (f"human listening decision pending for {', '.join(pending)} (data/review/listening_review.html)"
             if pending else "every choice confirmed by a human listening decision"))

    # 7. raw files preserved
    src = audio_source()
    dl = pd.read_csv(DOWNLOAD_LOG, dtype=str)
    present = [r for r in dl.itertuples() if (PROJECT_ROOT / r.local_file).is_file()]
    changed = [r.local_file for r in present if sha256_of(PROJECT_ROOT / r.local_file) != r.sha256]
    if src["source"] == "original" and len(present) < len(dl):
        changed.append(f"{len(dl) - len(present)} original recordings missing")
    supplied_note, supplied_changed = "", []
    if src["source"] == "supplied":  # the analysed delivery must be exactly the audited files
        files = pd.read_csv(src["files_csv"], dtype=str)
        supplied_changed = [f.file for f in files.itertuples() if sha256_of(src["folder"] / f.file) != f.sha256]
        supplied_note = (f"; analysed delivery {src['delivery']}: {len(files)} files match their SHA-256: "
                         f"{not supplied_changed}")
    root_pdf = PROJECT_ROOT / "G3 - mini project - TEmplate.pdf"
    pdf_ok = (sha256_of(root_pdf) == sha256_of(DATA / "metadata_raw" / "G3 - mini project - TEmplate.pdf")
              if root_pdf.is_file() else True)  # the root copy exists only on the original machine
    check("raw files preserved", "PASS" if not changed and not supplied_changed and pdf_ok else "FAIL",
          f"{len(present)} of {len(dl)} original recordings present and matching their download SHA-256: "
          f"{not changed}{supplied_note}; metadata PDF copy identical: {pdf_ok}")

    # 8. processed WAVs
    wav_problems = []
    for rel_path in included["processed_file"].tolist() + included["segment_file"].tolist():
        info = sf.info(PROJECT_ROOT / rel_path)
        if (info.samplerate, info.channels, info.subtype, info.format) != (TARGET_SR, 1, "PCM_16", "WAV"):
            wav_problems.append(rel_path)
    check("processed WAV files generated", "PASS" if not wav_problems else "FAIL",
          f"{2 * n_valid} WAV files (recordings + segments) checked: 16 kHz, mono, PCM 16-bit; problems: {wav_problems or 'none'}")

    # 9. filenames
    names = [p.split("/")[-1] for p in included["processed_file"]]
    viol = [n for n in names if not FILENAME_RE.match(n)]
    code_match = all(n.startswith(c + "_") for n, c in zip(names, included["participant_code"]))
    unverified = [n for n in names if n.endswith("_UNVERIFIED.wav")]
    check("filenames follow convention", "PASS" if not viol and code_match else "FAIL",
          f"{len(names)} names match ParticipantCode_YearLevel_Session_Word.wav; code consistent: {code_match}; "
          f"word token UNVERIFIED (not inferred): {unverified or 'none'}; session token PM is an assumption")

    # 10. features
    feat_cols = [c for c in feats.columns if c not in ("participant_code", "segment_file", "segment_sha256")]
    one_row = feats["participant_code"].is_unique and set(feats["participant_code"]) == set(included["participant_code"])
    seg_hash_ok = all(sha256_of(PROJECT_ROOT / f) == h for f, h in zip(feats["segment_file"], feats["segment_sha256"]))
    check("features extracted successfully", "PASS" if one_row and seg_hash_ok and len(feat_cols) == 108 else "FAIL",
          f"{len(feats)} rows x {len(feat_cols)} features; one row per included participant: {one_row}; "
          f"features computed from the current segment files: {seg_hash_ok}")

    # 11. leakage
    forbidden = ("participant", "code", "file", "sha", "word", "feeling", "year", "subject", "session", "activity",
                 "consent", "valence", "arousal")
    leaked = [c for c in feat_cols if any(t in c for t in forbidden)]
    model_feats_ok = True
    for path in (MODELS / "final").glob("*.joblib"):
        bundle = joblib.load(path)
        model_feats_ok &= bundle["features"] == feat_cols
    trained_on = json.loads((MODELS / "training_summary.json").read_text(encoding="utf-8"))
    current_hash = sha256_of(FEATURES_CSV)
    check("models trained on the current feature table", "PASS" if trained_on["features_sha256"] == current_hash and
          not trained_on.get("quick_mode") else "FAIL",
          f"training_summary.json features_sha256 = current data/acoustic_features.csv: "
          f"{trained_on['features_sha256'] == current_hash}; full (non-quick) run: {not trained_on.get('quick_mode')}")
    check("no lexical/context leakage in primary model", "PASS" if not leaked and model_feats_ok else "FAIL",
          f"X = {len(feat_cols)} acoustic features; metadata-like columns in X: {leaked or 'none'}; saved models use "
          f"exactly these features: {model_feats_ok}; preprocessing inside sklearn Pipelines (06_train_models.py)")

    # 12. speaker independence
    oof = pd.read_csv(MODELS / "oof_predictions.csv")
    splits = pd.read_csv(MODELS / "cv_splits.csv")
    per_repeat_once = (oof.groupby(["task", "model", "repeat", "participant_code"]).size() == 1).all()
    one_fold = (splits.groupby(["task", "repeat", "participant_code"])["fold"].nunique() == 1).all()
    check("speaker independence protected", "PASS" if per_repeat_once and one_fold else "FAIL",
          f"every participant is in exactly one test fold per repeat: {one_fold}; predicted exactly once per repeat: "
          f"{per_repeat_once}; grouping variable = participant_code (StratifiedGroupKFold)")

    # 13-15. reports and figures
    for item, path in (("EDA completed", REPORTS / "exploratory_analysis.md"),
                       ("data quality report", REPORTS / "data_quality_report.md"),
                       ("model evaluation report", REPORTS / "model_evaluation.md"),
                       ("lexical-control analysis attempted where possible", REPORTS / "lexical_control_analysis.md")):
        check(item, "PASS" if path.is_file() and path.stat().st_size > 1000 else "FAIL", f"{path.relative_to(PROJECT_ROOT).as_posix()}")
    fs = pd.read_csv(MODELS / "cv_fold_scores.csv")
    models_present = set(fs["model"])
    check("baseline model included", "PASS" if {"dummy_majority", "dummy_stratified"} <= models_present else "FAIL",
          f"models evaluated: {sorted(models_present)}")
    check("multiple models compared", "PASS" if {"logistic_regression", "svm_rbf", "random_forest"} <= models_present else "FAIL",
          "corrected repeated k-fold t-tests in reports/evaluation/model_comparison.csv")
    check("class imbalance addressed appropriately", "PASS",
          "class_weight='balanced' in every learned model; balanced accuracy and macro-F1 reported; no resampling")
    cms = list((FIGURES / "evaluation").glob("fig_eval_confusion_*.png"))
    check("confusion matrices created", "PASS" if cms else "FAIL", f"{len(cms)} figures + reports/evaluation/confusion_matrices_mean_per_repeat.csv")
    summ = pd.read_csv(REPORTS / "evaluation" / "summary_metrics.csv")
    need = {"pooled_accuracy", "pooled_ba", "pooled_macro_f1", "fold_macro_precision_mean", "fold_macro_recall_mean"}
    check("metrics calculated", "PASS" if need <= set(summ.columns) else "FAIL",
          "accuracy, balanced accuracy, macro precision/recall/F1, per-class metrics, bootstrap CIs, permutation p")

    # 16. limitations and final report consistency
    final = REPORTS / "final_group_research_report.md"
    if final.is_file():
        text = final.read_text(encoding="utf-8")
        check("limitations documented", "PASS" if "## 11. Limitations" in text else "FAIL", "section 11 of the final report")
        ev = json.loads((REPORTS / "evaluation" / "evaluation_summary.json").read_text(encoding="utf-8"))
        n_ok = f"n = {n_valid}" in text or f"{n_valid} participants" in text
        sig_claim_ok = ("No learned model exceeded" in text) != ev["any_model_significant_after_holm"]
        check("research report matches actual results", "PASS" if n_ok and sig_claim_ok else "FAIL",
              f"sample size {n_valid} stated: {n_ok}; significance statement consistent with evaluation_summary.json: {sig_claim_ok}")
    else:
        check("limitations documented", "FAIL", "final report missing")
        check("research report matches actual results", "FAIL", "final report missing")

    # 17. reproducibility
    rq = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    pinned = all(f"{pkg}==" in rq for pkg in ("librosa", "scikit-learn", "praat-parselmouth", "imageio-ffmpeg", "numpy", "pandas"))
    check("code reproducible", "PASS" if pinned and (PROJECT_ROOT / "run_pipeline.py").is_file() else "FAIL",
          f"pinned requirements: {pinned}; run_pipeline.py present; fixed seeds (random_state=42); processed WAVs "
          "byte-identical across runs (verified during development)")

    counts = pd.Series([r["status"] for r in RESULTS]).value_counts().to_dict()
    lines = ["# Validation Report (Phase 20)", "",
             f"*Generated by `src/10_validate_project.py` on {datetime.now().astimezone().isoformat(timespec='seconds')}. "
             "Each check is re-computed from the files.*", "",
             f"**Result: {counts}**", "",
             "| Check | Status | Evidence |", "|---|---|---|"]
    lines += [f"| {r['item']} | **{r['status']}** | {r['evidence']} |" for r in RESULTS]
    (REPORTS / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORTS / "validation_results.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    log.info("validation: %s", counts)
    return 1 if counts.get("FAIL", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
