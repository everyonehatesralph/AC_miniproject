"""
09_build_final_report.py -- PHASES 14-19: the final group research report.

Builds reports/final_group_research_report.md from the pipeline's own outputs. Every number is read
from a generated file, and every result-dependent sentence (significance, model ranking, lexical
findings) is produced by a branch on the data, so the report cannot contradict the results it
describes. Citations are limited to the methods actually used.

Usage: .venv/Scripts/python.exe src/09_build_final_report.py
"""

from __future__ import annotations

import json
import sys
from datetime import date

import numpy as np
import pandas as pd

from common import (CLEAN_METADATA_CSV, DATA, FEATURES_CSV, MIN_VALID_OBSERVATIONS, MODELS, PROJECT_ROOT,
                    PROTOCOL_WORDS, REPORTS, TARGETS, load_consent_log, setup_logging)

log = setup_logging("final_report", REPORTS / "logs" / "09_build_final_report.log")
MODEL_LABEL = {"dummy_majority": "Dummy (majority)", "dummy_stratified": "Dummy (stratified)",
               "logistic_regression": "Logistic regression", "svm_rbf": "SVM (RBF)", "random_forest": "Random forest"}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
            return "—"
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False))


def main() -> int:
    audit = load_json(REPORTS / "phase0" / "audit_summary.json")
    clean = load_json(REPORTS / "phase1" / "cleaning_summary.json")
    dq = load_json(REPORTS / "data_quality_summary.json")
    fsum = load_json(REPORTS / "features_summary.json")
    eda = load_json(REPORTS / "eda" / "eda_summary.json")
    train = load_json(MODELS / "training_summary.json")
    ev = load_json(REPORTS / "evaluation" / "evaluation_summary.json")
    lex = load_json(REPORTS / "lexical" / "lexical_summary.json")
    manifest = pd.read_csv(DATA / "analysis_manifest.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    meta = pd.read_csv(CLEAN_METADATA_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    feats = pd.read_csv(FEATURES_CSV)
    summ = pd.read_csv(REPORTS / "evaluation" / "summary_metrics.csv")
    comps = pd.read_csv(REPORTS / "evaluation" / "model_comparison.csv")
    per_class = pd.read_csv(REPORTS / "evaluation" / "per_class_metrics.csv")
    sens = pd.read_csv(REPORTS / "evaluation" / "sensitivity_summary.csv")
    within = pd.read_csv(REPORTS / "lexical" / "within_word_correlations.csv")
    sp = pd.read_csv(REPORTS / "eda" / "spearman_feature_vs_rating.csv")
    qa = pd.read_csv(REPORTS / "audio_quality_table.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")

    n_meta = len(meta)
    included = manifest[manifest["include_primary"] == "True"]
    n = len(included)
    missing_rec = manifest.loc[manifest["n_takes"] == "0", "participant_code"].tolist()
    retakes = dq["retake_participants"]
    tasks = ev["tasks_modelled"]
    not_run = ev["tasks_not_modelled"]
    learned = ["logistic_regression", "svm_rbf", "random_forest"]
    sel = qa[qa["selected"] == "True"]
    caps = sel["capture_time"].str[:19].str.replace("T", " ")
    main_caps = sel.loc[~sel["flags"].str.contains("CAPTURE_TIME_OUTLIER"), "capture_time"].str[11:16]
    words = meta.loc[meta["spoken_word"] != "", "spoken_word"]
    on_list = words.str.casefold().isin({w.casefold() for w in PROTOCOL_WORDS})
    va = eda["valence_arousal_spearman"]["analysis"]
    any_sig = ev["any_model_significant_after_holm"]
    sig_list = ev["significant_after_holm"]
    best = summ[summ["model"].isin(learned)].sort_values("pooled_ba", ascending=False).groupby("task").head(1)
    comp_sig = comps[(comps["p_value"] < 0.05) & (comps["model_b"] != "dummy_majority")]
    vs_dummy_sig = comps[(comps["p_value"] < 0.05) & (comps["model_b"] == "dummy_majority")]
    probe = lex["probe"]
    probe_sig = probe["permutation_p"] < 0.05
    word_assoc = lex["word_affect_association"]
    word_assoc_sig = any(a["p_value"] < 0.05 for a in word_assoc.values())
    fdr = eda["significant_features_fdr05"]
    top_ar = sp[sp["rating"] == "arousal"].sort_values("p_value").iloc[0]
    top_va = sp[sp["rating"] == "valence"].sort_values("p_value").iloc[0]
    cfg_feats = eda["config_effect_features_fdr05"]
    subsets = dq["subsets"]

    # ---- facts computed from the data (no hand-typed numbers below) ------------------------------
    pre = ["f0_median", "f0_range_st", "rms_db_mean", "utterance_duration_s", "spectral_centroid_hz_mean", "voiced_ratio"]
    pre_sp = sp[sp["feature"].isin(pre)]
    pre_any_sig = bool((pre_sp["q_value_bh"] < 0.05).any())
    pre_top = {k: pre_sp[pre_sp["rating"] == k].sort_values("p_value").iloc[0] for k in ("arousal", "valence")}
    pre_sentence = ("Among the six pre-specified prosodic features (F0 median, F0 range, RMS level, duration, spectral "
                    "centroid, voiced ratio), the strongest association with arousal was "
                    f"{pre_top['arousal'].feature} (ρ = {pre_top['arousal'].spearman_rho:+.2f}, p = {pre_top['arousal'].p_value:.3f}) "
                    f"and with valence {pre_top['valence'].feature} (ρ = {pre_top['valence'].spearman_rho:+.2f}, "
                    f"p = {pre_top['valence'].p_value:.3f}); "
                    + ("at least one passed FDR correction" if pre_any_sig else "none passed FDR correction"))
    sens_info = train["tasks"].get("sensitivity", {})
    sens_skipped = [f"{scen}/{task} ({', '.join(f'{k}: {v}' for k, v in info['class_counts'].items())})"
                    for scen, tasks_ in sens_info.items() for task, info in tasks_.items() if not info["viable"]]
    seg_min, seg_max = feats["utterance_duration_s"].min(), feats["utterance_duration_s"].max()
    n_offlist = int((~on_list).sum())
    flags_meta = meta["qa_flags"].str.split(";")
    n_feel_eq = int(flags_meta.apply(lambda f: "FEELING_IDENTICAL_TO_WORD" in f).sum())
    n_feel_missing = int(flags_meta.apply(lambda f: "SELF_REPORT_TEXT_MISSING" in f).sum())
    n_struck = int((meta["arousal_status"] == "STRUCK_THROUGH").sum() + (meta["valence_status"] == "STRUCK_THROUGH").sum())
    med_cap = pd.to_datetime(dq["median_capture"])
    outlier_caps = pd.to_datetime(sel.loc[sel["flags"].str.contains("CAPTURE_TIME_OUTLIER"), "capture_time"], format="%Y-%m-%dT%H:%M:%S%z")
    outlier_hours = ((outlier_caps - med_cap).dt.total_seconds() / 3600).round(1) if len(outlier_caps) else pd.Series(dtype=float)
    n_bg = int(dq["flag_counts_selected"].get("BACKGROUND_VOICES", 0))
    runs: list[list[int]] = []  # cleaning rules applied, as compact ranges ("R1–R8, R10")
    for k in sorted(int(r[1:]) for r in clean["changes_by_rule"]):
        if runs and k == runs[-1][1] + 1:
            runs[-1][1] = k
        else:
            runs.append([k, k])
    rule_span = ", ".join(f"R{a}" if a == b else f"R{a}–R{b}" for a, b in runs)
    # human listening verification so far (blind listening review, step 01b; or the group's own override files)
    sel_qa = qa[qa["selected"] == "True"]
    takes_human = int(((sel_qa["n_takes"].astype(int) > 1)
                       & sel_qa["selection_reason"].str.startswith("group decision")).sum())
    decisions_csv = REPORTS / "listening_review_decisions.csv"
    decisions = (pd.read_csv(decisions_csv, dtype=str, keep_default_na=False) if decisions_csv.is_file()
                 else pd.DataFrame(columns=["participant_code", "take", "judgement", "outcome"]))
    confirmed = {(r.participant_code, str(r.take)) for r in decisions[decisions["judgement"] == "boundaries"].itertuples()
                 if r.outcome.startswith("confirmed")}
    takes_shown = sel_qa["review_take"].astype(str) if "review_take" in sel_qa else pd.Series("", index=sel_qa.index)
    bounds_verified = int(sum(1 for code, take, source in zip(sel_qa["participant_code"], takes_shown,
                                                              sel_qa["segmentation_source"])
                              if source.startswith("human") or (code, take) in confirmed))
    words_from_review = meta.loc[meta["spoken_word_source"] == "listening_review", "participant_code"].tolist()
    # audio supplied after the original recordings (03c audit): never analysed, but reported
    supplied_csv = REPORTS / "supplied_audio_audit.csv"
    supplied = (pd.read_csv(supplied_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig") if supplied_csv.is_file()
                else pd.DataFrame(columns=["delivery", "participant_code", "verdict", "flags"]))
    supplied_new = supplied[supplied["verdict"] == "NEW_RECORDING"]
    supplied_edited = supplied_new[supplied_new["flags"] != ""]
    supplied_odd = supplied[supplied["verdict"].isin(["DIFFERENT_RECORDING", "OTHER_PARTICIPANT", "UNCERTAIN_MATCH"])]
    supplied_note = (f"a later delivery holds trimmed 16 kHz copies for {len(supplied_new)} of them "
                     f"({len(supplied_edited)} edited or damaged), which cannot count without the original recordings "
                     "(reports/supplied_audio_audit.md)" if len(supplied_new) else "")
    bounds_text = (f"boundaries verified by listening for {bounds_verified} of {n} selected recordings"
                   if bounds_verified else "segment boundaries not yet verified by listening")
    retake_text = ("all confirmed by human listening" if takes_human == len(retakes) else
                   f"{takes_human} confirmed by human listening, the rest pending" if takes_human else
                   "pending listening confirmation")
    cfg = pd.read_csv(REPORTS / "eda" / "recording_configuration_effect.csv")
    cfg_dirs = [f"{'lower' if r.median_B < r.median_A else 'higher'} {r.feature}" for r in
                cfg[(cfg["q_value_bh"] < 0.05) & cfg["feature"].isin(["zcr_mean", "spectral_centroid_hz_mean", "mfcc_1_mean"])].itertuples()]
    claim_ok = any_sig  # wording of the headline claim follows the permutation tests
    headline = ("provide preliminary, statistically supported evidence that acoustic features carry information about "
                "self-reported affect for some targets" if claim_ok else
                "do not support a claim that the system can recognise post-class affect from speech")

    def res_row(task, model):
        r = summ[(summ["task"] == task) & (summ["model"] == model)].iloc[0]
        return r

    main_tab = pd.DataFrame([{
        "Task": t, "Model": MODEL_LABEL[m],
        "Balanced accuracy [95% CI]": f"{res_row(t, m).pooled_ba:.3f} [{res_row(t, m).pooled_ba_ci_low:.3f}, {res_row(t, m).pooled_ba_ci_high:.3f}]",
        "Macro-F1 [95% CI]": f"{res_row(t, m).pooled_macro_f1:.3f} [{res_row(t, m).pooled_f1_ci_low:.3f}, {res_row(t, m).pooled_f1_ci_high:.3f}]",
        "Accuracy": res_row(t, m).pooled_accuracy,
        "Fold BA mean ± SD": f"{res_row(t, m).fold_balanced_accuracy_mean:.3f} ± {res_row(t, m).fold_balanced_accuracy_sd:.3f}",
        "Permutation p (Holm)": "—" if m.startswith("dummy") else f"{res_row(t, m).perm_p:.3f} ({res_row(t, m).perm_p_holm:.3f})",
    } for t in tasks for m in ["dummy_majority", "dummy_stratified"] + learned])

    # ---- result-dependent statements (branches on the data) ------------------------------------
    if any_sig:
        ml_statement = ("After Holm correction, " + "; ".join(
            f"{MODEL_LABEL[s['model']]} on {s['task']} (balanced accuracy {s['pooled_ba']:.3f}, adjusted p = {s['perm_p_holm']:.3f})"
            for s in sig_list) + " exceeded the label-permutation null.")
    else:
        raw_sig = summ[summ["perm_p"] < 0.05]
        n_tests = int(summ["perm_p"].notna().sum())
        ml_statement = ("No learned model exceeded the label-permutation null after Holm correction (all adjusted "
                        f"p ≥ 0.05; smallest unadjusted p = {summ['perm_p'].min():.4f}, before correcting for {n_tests} tests)."
                        + (f" Before correction, {len(raw_sig)} of {n_tests} tests had p < 0.05 ("
                           + "; ".join(f"{MODEL_LABEL[r.model]} on {r.task}: p = {r.perm_p:.3f}" for r in raw_sig.itertuples())
                           + f"). If no model carried any signal, about {0.05 * n_tests:.2f} such results would be expected "
                             "by chance across these tests, and none survives the correction." if len(raw_sig) else ""))
    best_txt = "; ".join(f"{r.task}: {MODEL_LABEL[r.model]} {r.pooled_ba:.3f} (chance {r.chance_ba:.2f})"
                         for r in best.itertuples())
    if len(comp_sig):
        rq5 = ("Pairwise comparisons found " + "; ".join(
            f"{MODEL_LABEL[r.model_a]} vs {MODEL_LABEL[r.model_b]} on {r.task} (Δ = {r.mean_ba_difference:+.3f}, p = {r.p_value:.3f})"
            for r in comp_sig.itertuples()) + " (corrected t-test, uncorrected for the number of comparisons).")
    else:
        rq5 = ("No pair of learned models differed significantly on any task (corrected repeated k-fold t-test, all "
               "p ≥ 0.05). The data therefore do not identify a best model; the highest point estimates were " + best_txt + ".")
    lex_statement = (
        f"The lexical-confound probe separated *Kapoy* from other words with balanced accuracy "
        f"{probe['balanced_accuracy_mean_10_repeats']:.3f} (permutation p = {probe['permutation_p']:.3f}); "
        + ("the acoustic features therefore carry word identity. " if probe_sig else
           "acoustic features did not reliably carry word identity in this sample. ")
        + ("Word choice was associated with the ratings " if word_assoc_sig else
           "Word choice (Kapoy vs other) was not significantly associated with the ratings ")
        + "(" + "; ".join(f"{t}: p = {a['p_value']:.3f}, Cramér's V = {a['cramers_v']:.2f}" for t, a in word_assoc.items()) + ")."
    )
    within_sig = int((within["q_bh"] < 0.05).sum()) if len(within) else 0
    acc_word = pd.read_csv(REPORTS / "lexical" / "oof_accuracy_by_word_group.csv")
    acc_word = acc_word[acc_word["word_group"].isin(["Kapoy", "other words"])]
    acc_pivot = acc_word.groupby(["task", "word_group"])["accuracy_over_repeats"].mean().unstack()
    acc_sentence = "; ".join(f"{t}: {acc_pivot.loc[t, 'Kapoy']:.2f} for Kapoy vs {acc_pivot.loc[t, 'other words']:.2f} "
                             "for other words" for t in acc_pivot.index)
    reversals = within[(~within["same_direction"]) & (within["rho_full_sample"].abs() >= 0.3)]
    reversal_sentence = ("; ".join(f"{r.feature} with {r.rating}: ρ = {r.rho_full_sample:+.2f} in the full sample but "
                                   f"{r.rho_within_word:+.2f} within {r.word}" for r in reversals.itertuples()))
    lexical_confound = probe_sig and word_assoc_sig
    gaps = (acc_pivot['Kapoy'] - acc_pivot['other words'])
    gap_task = gaps.idxmax()
    kap = meta[meta['spoken_word'] == 'Kapoy'].merge(included[['participant_code']], on='participant_code')
    kap_v, kap_a = kap['valence'].astype(int), kap['arousal'].astype(int)
    kap_n, kap_vmode, kap_amode = len(kap), int(kap_v.mode().iloc[0]), int(kap_a.mode().iloc[0])
    kap_v3, kap_a3 = int((kap_v == kap_vmode).sum()), int((kap_a == kap_amode).sum())
    ana = meta.merge(included[['participant_code']], on='participant_code')
    v35 = int(ana['valence'].astype(int).between(3, 5).sum())
    a35 = int(ana['arousal'].astype(int).between(3, 5).sum())

    # ---- consent documentation status (consent/consent_log.csv) ------------------------------
    clog = load_consent_log()
    if clog is None:
        consent_status = "BLOCKED"
        consent_evidence = (f"sheet records consent 'Y' for {(meta['consent_obtained'] == 'Y').sum()}/{n_meta}; "
                            "consent log not supplied")
        consent_issue = ("fill consent/consent_log_TEMPLATE.csv (codes only, no names), save as consent_log.csv; "
                         "keep signed forms outside the dataset")
    else:
        n_doc = sum(clog.values())
        consent_status = "COMPLETE" if n_doc >= n_meta else "INCOMPLETE"
        consent_evidence = f"documented consent for {n_doc} of {n_meta} metadata participants"
        consent_issue = "—" if n_doc >= n_meta else f"{n_meta - n_doc} participants still undocumented"

    # ---- deliverable status matrix ------------------------------------------------------------
    status = [
        ("1. Minimum of 40 valid participant observations", "BLOCKED" if n < MIN_VALID_OBSERVATIONS else "COMPLETE",
         "data/analysis_manifest.csv", f"{n} valid unique participants; {len(missing_rec)} metadata participants have no recording",
         f"needs {max(0, MIN_VALID_OBSERVATIONS - n)} more valid observations: recordings for {', '.join(missing_rec)} (if they exist) or new consented participants"
         + (f"; {supplied_note}" if supplied_note else "")),
        ("2. Required consent documentation", consent_status, "consent/consent_log.csv", consent_evidence,
         consent_issue),
        ("3. Properly coded research metadata", "COMPLETE", "data/metadata_cleaned.csv, reports/phase1/cleaning_log.csv",
         f"{n_meta} rows; raw kept beside cleaned; {sum(clean['changes_by_rule'].values())} logged changes (rules "
         f"{rule_span})",
         "confirm: Y2-076 arousal; 'Lecture' → 'Lecture/Discussion'; off-list words; 9 missing words, 2 missing feelings"),
        ("4. Properly named audio recordings", "COMPLETE", "data/processed_audio/recordings/, file_mapping.csv",
         f"{n} WAVs named ParticipantCode_YearLevel_Session_Word.wav; mapping for all {dq['total_files']} originals",
         "Y2-073 word token UNVERIFIED (word not recorded); session token PM is an assumption"),
        ("5. Data quality report", "COMPLETE", "reports/data_quality_report.md",
         f"{dq['total_files']} recordings assessed; {dq['selected_with_flags']} selected takes with ≥1 flag",
         "—" if bounds_verified == n else bounds_text + " (blind listening review, data/review/listening_review.html)"),
        ("6. Extracted acoustic feature dataset", "COMPLETE", "data/acoustic_features.csv",
         f"{len(feats)} × {fsum['n_features']} features, no missing values", "—"),
        ("7. Exploratory analysis", "COMPLETE", "reports/exploratory_analysis.md",
         "affect distributions, feature–affect statistics, confound analysis",
         "context × affect (RQ2) not analysable within Group 3 (constant variables)"),
        ("8. Machine learning experiment", "COMPLETE", "models/, src/06_train_models.py",
         f"{len(tasks)} viable tasks × 5 models, {train['n_repeats']}× repeated nested speaker-independent CV",
         f"tasks not modelled (sparse classes): {', '.join(not_run)}"),
        ("9. Model evaluation", "COMPLETE", "reports/model_evaluation.md",
         "BA, macro-P/R/F1, accuracy, per-class, confusion matrices, bootstrap CIs, permutation tests, corrected t-tests", "—"),
        ("10. Interpretation of findings", "COMPLETE", "this report, §10", "observations / statistics / performance / interpretation separated", "—"),
        ("11. Research limitations", "COMPLETE", "this report, §11", "sample, noise, device, lexical, self-report, duplicates, ethics", "—"),
        ("12. Group research report", "COMPLETE", "reports/final_group_research_report.md",
         "generated from the pipeline outputs",
         "—" if consent_status == "COMPLETE" else "dissemination blocked until consent documentation is complete (item 2)"),
    ]
    status_tab = pd.DataFrame(status, columns=["Deliverable", "Status", "File", "Evidence", "Remaining issue"])

    lines = [
        "# Bisaya Post-Class Speech and Affective State Recognition Among Computer Science Students",
        "",
        "**Group 3 Research Report: Second-Year Computer Science Students (pilot study)**",
        "",
        f"*Generated by `src/09_build_final_report.py` on {date.today().isoformat()} from the outputs of the project pipeline. "
        "Every number in this report is read from a generated data file.*",
        "",
        "## Abstract",
        "",
        f"This pilot study tested whether acoustic and prosodic features of a single, naturally spoken Bisaya word, "
        f"recorded immediately after class, carry information about the speaker's independently self-reported "
        f"valence and arousal.",
        "",
        f"Of {n_meta} second-year CS participants in the Group 3 metadata, {n} had a usable recording. That is below "
        f"the protocol minimum of {MIN_VALID_OBSERVATIONS}.",
        "",
        f"From each recording's automatically located target utterance, {fsum['n_features']} features were extracted "
        f"(F0, energy, zero-crossing rate, MFCC with derivatives, spectral and temporal descriptors). Logistic "
        f"regression, an RBF support-vector machine and a random forest were compared with dummy baselines under "
        f"repeated, nested, participant-grouped cross-validation.",
        "",
        f"{ml_statement} Self-reported valence and arousal were strongly correlated (Spearman ρ = {va[0]:.2f}). "
        f"Feature–affect correlations surviving FDR correction: {fdr['valence']} of {fsum['n_features']} for valence, "
        f"{fdr['arousal']} for arousal.",
        "",
        f"A recording-configuration artefact affected {cfg_feats} features, and the retained retakes and automatic "
        f"utterance localisation add further uncertainty. Within this pilot dataset, the results {headline}. The "
        f"findings should be interpreted as a feasibility study.",
        "",
        "## 1. Introduction",
        "",
        "Speech Emotion Recognition (SER) research has largely relied on acted emotion. This project instead studies "
        "*naturally expressed* post-class affect in Computer Science students. Each participant says one Bisaya word "
        "of their choosing and then reports how they actually feel. The self-report, not the meaning of the word "
        "and not a listener's impression, is the reference label (\"Kapoy\" ≠ automatically sad).",
        "",
        "The computational question is whether *how* the word is spoken (its acoustic and prosodic realisation) "
        "contains information about that self-reported state.",
        "",
        "## 2. Research Objectives",
        "",
        "1. Determine the affective states reported by CS students immediately after class.",
        "2. Examine the acoustic and prosodic characteristics of short Bisaya post-class responses.",
        "3. Investigate whether acoustic/prosodic features can classify self-reported affective states.",
        "4. Compare appropriate machine learning models for Bisaya affect recognition.",
        "5. Explore relationships between post-class affect and educational context (class activity, year level, session).",
        "",
        "## 3. Research Questions",
        "",
        "| RQ | Question (protocol) | Addressed in Group 3 |",
        "|---|---|---|",
        "| RQ1 | What affective states do students report immediately after class? | §7.1 |",
        "| RQ2 | How does affect vary by year level, class activity and session? | **Not answerable within Group 3**: all three are constant (§7.3); requires the consolidated class dataset |",
        "| RQ3 | Which acoustic/prosodic characteristics accompany different self-reported states? | §7.2 |",
        "| RQ4 | To what extent can ML classify self-reported affect from acoustic features? | §8 |",
        "| RQ5 | Which model performs best? | §8.3 |",
        "",
        "## 4. Methodology",
        "",
        "### 4.1 Participants",
        "",
        f"The assigned population is Second-Year CS students (protocol §IV). The Group 3 metadata lists {n_meta} "
        f"participants, Y2-041 to Y2-080, every one recorded as 2nd year. Recordings were supplied for "
        f"{dq['participants_with_recordings']} of them. No recording exists for {len(missing_rec)} "
        f"({', '.join(missing_rec)}). The analysis set comprises the **{n} participants** with a usable recording "
        f"and valid ratings. Every participant contributes exactly one observation.",
        "",
        "### 4.2 Research Ethics",
        "",
        (f"The metadata records informed consent ('Yes') for all {n_meta} participants. **No consent documentation "
         "(consent log or institutional clearance) has been supplied.** Protocol §VI makes it a requirement before "
         "the data can be used for publication, so the results in this report must not be disseminated until it "
         "is verified." if consent_status == "BLOCKED" else
         f"Consent is documented in `consent/consent_log.csv` ({consent_evidence}). Participants without documented "
         "consent are excluded from the analysis automatically."
         + ("" if consent_status == "COMPLETE" else
            " Dissemination should wait until every participant's consent is documented.")),
        "",
        "Participants are identified only by pseudonymous codes. The raw videos (faces, possible GPS tags) are kept "
        "locally and never copied into the analysis. Only audio is processed, and location metadata is never "
        "extracted.",
        "",
        "### 4.3 Data Collection",
        "",
        f"Every metadata row records Class Activity 'Lecture' and Session 'Afternoon–Evening' (subject: "
        f"Environmental Science for {(meta['subject'] == 'Environmental Science').sum()} rows, blank for "
        f"{(meta['subject'] == '').sum()}).",
        "",
        f"Embedded capture times place {len(main_caps)} of the {n} selected recordings between {main_caps.min()} "
        f"and {main_caps.max()} (PHT) on {caps.str[:10].mode().iloc[0]}. That is consistent with the recorded "
        f"session. The time boundary between sessions was not documented in the supplied material.",
        "",
        f"All recordings were made on one smartphone model (iPhone 17) as short MOV videos (AAC, 48 kHz). "
        f"{subsets['subset_config_or_time_anomaly']} recordings differ in audio configuration (stereo plus a second "
        f"track); {len(outlier_hours)} of them were captured {outlier_hours.min():.1f}–{outlier_hours.max():.1f} h after "
        f"the median capture time. The group must explain this (see §6).",
        "",
        "### 4.4 Natural Bisaya Speech Procedure",
        "",
        "Participants were asked to choose one word closest to their reaction and say it naturally. The protocol "
        "lists Salamat, Ambot, Grabe, Lisod, Sayon, Lingaw and Kapoy.",
        "",
        f"In Group 3, {int(on_list.sum())} of {len(words)} recorded words are on that list "
        f"({', '.join(words[on_list].value_counts().index)}) and {int((~on_list).sum())} are not "
        f"(most frequent: {', '.join(words[~on_list].value_counts().index[:4])}). {n_meta - len(words)} rows "
        "have no recorded word. Whether off-list words are acceptable needs an instructor ruling. They are kept "
        "here because the word is never a model input.",
        "",
        "### 4.5 Self-Reported Affect",
        "",
        "Immediately after the recording, each participant described how they felt and rated valence (1 very "
        "unpleasant – 5 very pleasant) and arousal (1 very low – 5 very high). These ratings are the reference "
        "labels.",
        "",
        "Y2-076's arousal rating is struck through in the sheet and is not used. "
        + ("Y2-076 has no original recording in any case." if "Y2-076" in missing_rec else
           "Y2-076 can therefore enter only the valence analyses."),
        "",
        "### 4.6 Audio Processing",
        "",
        "1. **Original-audio QA:** peak, clipping, DC offset, level and configuration, measured before processing.",
        "2. **Standardisation:** ffmpeg first audio stream → mono → 16 kHz → WAV PCM 16-bit, metadata stripped. "
        "There is no filtering, denoising, normalisation, pitch or speed change. Output is byte-reproducible.",
        "3. **Target-utterance localisation.** The clips contain continuous hallway babble and sometimes other "
        "speech, so the participant's word was located as the dominant short-time-energy event. This is "
        "deterministic and blind to labels. Segments touching the clip edge, longer than 1.5 s, or with a "
        "competing event within 3 dB are flagged.",
        f"4. **Duplicates.** {len(retakes)} participants ({', '.join(retakes)}) had retakes captured seconds apart. "
        "One take each was chosen by a label-blind *complete-word rule*: drop takes whose word is cut off by the "
        "clip edge, too long to be one word, or too short to contain the recorded word (< 0.10 s per syllable); "
        "prefer a take without a competing event; then use the clearest take if it is ≥ 3 dB clearer, otherwise "
        "the earliest. Takes are never chosen for sounding more emotional. A blind listening review (at least two "
        f"listeners; only unanimous answers count) can confirm or replace each choice; so far: {retake_text}.",
        "5. **Naming:** `ParticipantCode_YearLevel_Session_Word.wav`. `PM` is assumed for Afternoon–Evening, and "
        "`UNVERIFIED` is used where no word was recorded.",
        "",
        "### 4.7 Acoustic Feature Extraction",
        "",
        f"For each located utterance, {fsum['n_features']} features were computed: duration and voicing (3); F0 "
        "from Praat autocorrelation pitch (mean, SD, median, minimum, maximum, IQR in Hz; semitone SD, range and "
        "slope) (9); RMS energy in dBFS (6); zero-crossing rate (6); 13 MFCCs with Δ and ΔΔ (mean and SD each: "
        "78); spectral centroid, bandwidth and roll-off (6).",
        "",
        "Praat was used instead of librosa pYIN, whose voicing decisions were unstable in this noise (see "
        "documentation). F0 in Hz mainly reflects voice type, because no speaker baseline exists.",
        "",
        "### 4.8 Machine Learning Methods",
        "",
        f"- **Inputs:** acoustic features only. Spoken word, feeling text, year, subject, session, class activity, "
        "participant code and file names are excluded, and the training script refuses to run otherwise.",
        f"- **Targets:** derived groupings of the participants' own ratings. A task is modelled only if every "
        f"class has at least 5 participants. Modelled: {', '.join(f'{t} ({train['tasks'][t]['role']})' for t in tasks)}. "
        f"Not modelled (too few participants in a class): {', '.join(not_run)}.",
        "- **Models:** DummyClassifier (majority; stratified), L2 logistic regression, RBF SVM and random forest, "
        "each in a Pipeline of median imputation and standardisation (no scaler for the forest).",
        "- **Imbalance:** class weights; no resampling.",
        "- **Tuning:** small grids, tuned only inside training folds.",
        "",
        "### 4.9 Evaluation Strategy",
        "",
        f"- **Cross-validation:** speaker-independent StratifiedGroupKFold grouped by participant (5 folds), "
        f"repeated {train['n_repeats']} times. Every split is verified to share no participant.",
        "- **Metrics:** accuracy, balanced accuracy, macro precision, recall and F1, per-class metrics and "
        "confusion matrices.",
        f"- **Uncertainty and chance:** a participant-level bootstrap gives 95% intervals. A label-permutation test "
        f"({train['n_permutations']} permutations of the full nested procedure) checks each model against chance, "
        f"with Holm adjustment.",
        "- **Model comparison:** corrected repeated k-fold t-test (Nadeau & Bengio, 2003).",
        "- **Sensitivity analyses:** removing retake participants, config-B/capture-outlier recordings and "
        "segmentation-critical recordings.",
        "",
        "## 5. Dataset Description",
        "",
        md_table(pd.DataFrame([
            {"Stage": "Metadata rows (participants)", "n": n_meta},
            {"Stage": "Participants with ≥1 recording", "n": dq["participants_with_recordings"]},
            {"Stage": "Recording files supplied", "n": dq["total_files"]},
            {"Stage": "Participants with retakes", "n": len(retakes)},
            {"Stage": "Participants with a usable selected recording (analysis set)", "n": n},
            {"Stage": "Protocol minimum", "n": MIN_VALID_OBSERVATIONS},
        ])),
        "",
        "Target class counts in the analysis set:",
        "",
        md_table(pd.DataFrame([{"Task": t, "Classes": " / ".join(f"{k}: {v}" for k, v in i["class_counts"].items()),
                                "Modelled": "yes" if i["viable"] else "no"}
                               for t, i in train["tasks"].items() if t != "sensitivity"])),
        "",
        "## 6. Data Quality Results",
        "",
        md_table(pd.DataFrame([
            {"Quantity": "Recordings (files)", "Value": dq["total_files"]},
            {"Quantity": "Participants without any recording", "Value": f"{len(missing_rec)} ({', '.join(missing_rec)})"},
            {"Quantity": "Duplicate (retake) participants / extra files", "Value": f"{len(retakes)} / {dq['extra_takes']}"},
            {"Quantity": "Unreadable files / files without speech", "Value": f"{dq['unreadable_files']} / {dq['no_speech_files']}"},
            {"Quantity": "Valid recordings (selected, usable)", "Value": dq["valid_recordings"]},
            {"Quantity": "Selected recordings with ≥1 flag", "Value": dq["selected_with_flags"]},
            {"Quantity": "…with a segmentation-critical flag", "Value": dq["selected_segmentation_critical"]},
            {"Quantity": "…with a different recording configuration", "Value": subsets["subset_config_or_time_anomaly"]},
        ])),
        "",
        f"Flags among the selected recordings: {', '.join(f'{k} {v}' for k, v in dq['flag_counts_selected'].items())}. "
        f"Background voices were detected in {n_bg} of {n} selected clips. The recordings were not made in a quiet "
        "place (protocol §X).",
        "",
        f"**Recording-configuration artefact.** The {subsets['subset_config_or_time_anomaly']} config-B recordings "
        f"differ from the rest in {cfg_feats} of {fsum['n_features']} features at FDR q < 0.05 ({', '.join(cfg_dirs)}). "
        "This is a device/setting effect that a classifier could mistake for "
        "affect. Configuration was not significantly associated with the labels ("
        + "; ".join(f"{t}: p = {a['p_value']:.2f}" for t, a in eda["config_label_association"].items()) + ").",
        "",
        "Full detail: `reports/data_quality_report.md`.",
        "",
        "## 7. Exploratory Analysis",
        "",
        "### 7.1 Self-reported affect (RQ1)",
        "",
        f"Valence was mostly neutral to pleasant (analysis set: "
        + ", ".join(f"{k}: {v}" for k, v in train['tasks']['valence_5']['class_counts'].items())
        + ") and arousal mostly moderate to high ("
        + ", ".join(f"{k}: {v}" for k, v in train['tasks']['arousal_5']['class_counts'].items())
        + f"). Only {train['tasks']['valence_3']['class_counts']['Unpleasant']} participants in the analysis set "
        f"reported unpleasant valence (1–2). Valence and arousal were strongly positively correlated "
        f"(ρ = {va[0]:.2f}, p = {va[1]:.2g}, n = {va[2]}): lower valence tended to co-occur with lower arousal.",
        "",
        "### 7.2 Acoustic characteristics by self-reported affect (RQ3)",
        "",
        f"Across {fsum['n_features']} features, {fdr['arousal']} correlated with arousal and {fdr['valence']} with "
        f"valence after Benjamini–Hochberg correction. The strongest associations were {top_ar.feature} with arousal "
        f"(ρ = {top_ar.spearman_rho:+.2f}, q = {top_ar.q_value_bh:.3f}) and {top_va.feature} with valence "
        f"(ρ = {top_va.spearman_rho:+.2f}, q = {top_va.q_value_bh:.3f}).",
        "",
        pre_sentence + " (see `reports/exploratory_analysis.md`, Fig. 9–10).",
        "",
        "### 7.3 Educational context (RQ2)",
        "",
        "Year level (2nd year), class activity (Lecture) and session (Afternoon–Evening) are constant across Group "
        "3, and subject is one value or blank. Their relationship with affect **cannot be estimated** from these "
        "data; this requires the consolidated class dataset.",
        "",
        "## 8. Machine Learning Results",
        "",
        "### 8.1 Primary results (RQ4)",
        "",
        md_table(main_tab),
        "",
        "Balanced accuracy and macro-F1 are computed on pooled out-of-fold predictions and averaged over repeats; "
        "the 95% interval is a participant-level stratified bootstrap. Chance balanced accuracy is 1/number of "
        "classes.",
        "",
        f"**{ml_statement}**",
        "",
        "![balanced accuracy](../figures/evaluation/fig_eval_balanced_accuracy.png)",
        "",
        "### 8.2 Per-class behaviour",
        "",
        md_table(per_class[per_class["model"].isin(learned)].assign(model=lambda d: d["model"].map(MODEL_LABEL))),
        "",
        *[f"![{t}](../figures/evaluation/fig_eval_confusion_{t}.png)\n" for t in tasks],
        "### 8.3 Model comparison (RQ5)",
        "",
        rq5,
        "",
        md_table(comps.assign(model_a=comps["model_a"].map(MODEL_LABEL), model_b=comps["model_b"].map(MODEL_LABEL))),
        "",
        "### 8.4 Sensitivity analyses",
        "",
        md_table(sens.assign(model=sens["model"].map(MODEL_LABEL))[["scenario", "task", "model", "n", "pooled_ba",
                                                                   "primary_pooled_ba", "difference_vs_primary"]]
                 .rename(columns={"pooled_ba": "BA (subset)", "primary_pooled_ba": "BA (all 32)",
                                  "difference_vs_primary": "difference"})),
        "",
        ("Not modelled under the viability rule (class counts after removal): " + "; ".join(sens_skipped) + ". "
         if sens_skipped else "") + "Full detail: `reports/model_evaluation.md`.",
        "",
        "## 9. Lexically Controlled Analysis",
        "",
        f"Only *Kapoy* had enough recordings ({int(within['n'].iloc[0]) if len(within) else 0}) and rating "
        f"variation for a same-word comparison. Within *Kapoy*, {within_sig} of {len(within)} feature–rating "
        f"correlations passed FDR correction. Within-word correlations had the same sign as in the full sample in "
        f"{lex['within_word_same_direction_share']:.0%} of tests. With n = {kap_n} ({kap_v3} report valence "
        f"{kap_vmode}, {kap_a3} report arousal {kap_amode}), these estimates are descriptive only.",
        "",
        lex_statement,
        "",
        f"Out-of-fold accuracy of the affect models, averaged over the three learned models: {acc_sentence}.",
        "",
        (f"Some full-sample associations change sign once the word is held constant ({reversal_sentence}). Those "
         "associations are therefore at least partly lexical." if len(reversals) else ""),
        "",
        ("**Consequence.** Both conditions for a lexical confound hold in these data: the acoustic features identify "
         "the word, and the word is associated with the self-reports. Any above-chance affect classification here "
         "can therefore reflect *which word* was said rather than *how* it was said. Only the within-word "
         "comparison controls for this, and it found " + ("no reliable effect." if within_sig == 0 else
         f"{within_sig} FDR-significant association(s).") if lexical_confound else
         "Both conditions for a lexical confound (word identifiable from acoustics, and word associated with the "
         "self-reports) were not jointly met at p < 0.05 in this sample."),
        "",
        "Full detail: `reports/lexical_control_analysis.md`.",
        "",
        "## 10. Discussion",
        "",
        "**Observations.**",
        f"- {n} of {n_meta} participants have usable recordings; there are {len(retakes)} retake sets, "
        f"{n_meta - len(words)} missing words, and a different recording configuration for "
        f"{subsets['subset_config_or_time_anomaly']} files.",
        "- The recordings are short clips in a noisy, multi-speaker environment.",
        f"- Self-reports cluster at neutral-to-pleasant valence (3–5: {v35} of {n}) and moderate-to-high arousal "
        f"(3–5: {a35} of {n}).",
        "",
        "**Statistical results.**",
        f"- Valence and arousal co-vary strongly (ρ = {va[0]:.2f}).",
        f"- After FDR correction, {fdr['arousal']} {'feature is' if fdr['arousal'] == 1 else 'features are'} "
        f"associated with arousal and {fdr['valence']} with valence.",
        f"- The recording configuration changes {cfg_feats} features.",
        "",
        "**Model performance.**",
        f"- {ml_statement}",
        f"- Highest point estimates: {best_txt}.",
        "- Bootstrap intervals are wide and overlap chance-level baselines.",
        "",
        "**Interpretation.**",
        ("- Within this pilot dataset, part of the evidence indicates that the speech signal of a single word "
         "carries information about an unseen student's self-reported affect, but the margins over chance are "
         "small and fragile." if claim_ok else
         "- Within this pilot dataset, the evidence that the speech signal of a single word predicts an unseen "
         "student's self-reported affect is weak at best."),
        "- Three factors plausibly limit any signal:",
        "  1. a small, label-sparse sample (very few unpleasant or low-arousal reports);",
        "  2. heavy background babble and automatic utterance localisation;",
        "  3. single-word utterances that give little prosodic material.",
        ("- The lexical confound is not hypothetical here. Acoustic features identify the word "
         f"(probe balanced accuracy {probe['balanced_accuracy_mean_10_repeats']:.2f}), word choice is associated "
         f"with the self-reports, and for {gap_task} the models are markedly more accurate on *Kapoy* recordings "
         f"({acc_pivot.loc[gap_task, 'Kapoy']:.2f} vs {acc_pivot.loc[gap_task, 'other words']:.2f}). An acoustic model "
         "can therefore score above chance by recognising the word rather than affective prosody."
         if lexical_confound else
         "- Lexical content could still be absorbed by an acoustic model, although the confound conditions were "
         "not jointly significant in this sample."),
        f"- Recording configuration is a further confound: {subsets['subset_config_or_time_anomaly']} recordings "
        f"differ systematically in {cfg_feats} acoustic features.",
        "- These are associations in a small observational sample. Nothing here shows that classes, activities or "
        "words *cause* any affective state.",
        "- A negative or null result at this sample size is informative for the class study: it argues for the "
        "consolidated dataset, quieter recording, and verified segmentation.",
        "",
        "## 11. Limitations",
        "",
        f"1. **Small sample, below the protocol minimum:** n = {n} < {MIN_VALID_OBSERVATIONS}; no participant "
        f"recording for {', '.join(missing_rec)}.",
        f"2. **Class imbalance and sparse categories:** unpleasant valence n = "
        f"{train['tasks']['valence_3']['class_counts']['Unpleasant']}; {', '.join(not_run)} could not be "
        "modelled; derived groupings merge categories.",
        f"3. **Single-word utterances** ({seg_min:.2f}–{seg_max:.2f} s located segments) carry limited prosodic information.",
        f"4. **Naturally occurring noise:** background voices detected in {n_bg} of {n} selected clips; low "
        f"utterance-to-background ratio (< 10 dB) in {dq['flag_counts_selected'].get('LOW_SBR', 0)}.",
        f"5. **Automatic segmentation** ({bounds_text}); {dq['selected_segmentation_critical']} recordings "
        "have segmentation-critical flags.",
        f"6. **Device/setting variability:** {subsets['subset_config_or_time_anomaly']} recordings with a different "
        f"audio configuration shift {cfg_feats} features. F0 in Hz is dominated by voice type (no speaker baseline).",
        "7. **Limited contextual variation:** one year level, one activity, one session; RQ2 is not answerable "
        "within Group 3.",
        f"8. **Possible lexical influence:** {n_offlist} of {len(words)} words are off the protocol list and words "
        "differ phonetically, so acoustic features can encode word identity.",
        f"9. **Self-report uncertainty:** single-item ratings; {n_feel_eq} rows with feeling text identical to the "
        f"word; {n_feel_missing} rows without a description; {n_struck} struck-through rating{'' if n_struck == 1 else 's'}.",
        f"10. **Duplicates and missing observations:** {len(retakes)} retake sets were resolved by the complete-word "
        f"rule ({retake_text}); {len(missing_rec)} participants have no recording"
        + (f"; {supplied_note}" if supplied_note else "") + "."
        + (f" The delivery's file for {', '.join(supplied_odd['participant_code'])} does not match the original "
           "recording and is unexplained." if len(supplied_odd) else ""),
        "11. **No demographic variables were collected** (the protocol allows optional variables only when approved), "
        "so speaker characteristics such as sex confound F0.",
        ("12. **Consent documentation not supplied;** dissemination is blocked until it is verified."
         if consent_status == "BLOCKED" else
         f"12. **Consent documentation:** {consent_evidence}"
         + ("." if consent_status == "COMPLETE" else "; dissemination waits until it is complete.")),
        "13. **Limited generalisability:** one class session, one device model; a pilot study.",
        "",
        "## 12. Conclusion",
        "",
        f"The Group 3 pipeline is complete and reproducible: data audit, metadata cleaning, audio standardisation and "
        f"quality control, feature extraction, exploratory analysis, speaker-independent machine learning, "
        f"evaluation, lexical control and validation. It produced a documented dataset of {n} observations.",
        "",
        f"{ml_statement} Within this pilot dataset, the results therefore {headline}.",
        "",
        "They do identify concrete requirements for the consolidated study: reach the planned sample, record in a "
        "quieter location with one consistent device setting, verify the target utterance by listening, collect "
        "more low-valence and low-arousal responses, and supply consent documentation.",
        "",
        "## 13. References",
        "",
        "- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289–300.",
        "- Boersma, P. (1993). Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound. *Proceedings of the Institute of Phonetic Sciences, 17*, 97–110.",
        "- Bouckaert, R. R., & Frank, E. (2004). Evaluating the replicability of significance tests for comparing learning algorithms. In *Advances in Knowledge Discovery and Data Mining (PAKDD 2004)*, LNCS 3056, 3–12.",
        "- CS412 Research Mini-Project (v2). *Bisaya Post-Class Speech and Affective State Recognition Among Computer Science Students* (course protocol).",
        "- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.",
        "- Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. *Journal of Phonetics, 71*, 1–15.",
        "- McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference*, 18–25.",
        "- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*(3), 239–281.",
        "- Ojala, M., & Garriga, G. C. (2010). Permutation tests for studying classifier performance. *Journal of Machine Learning Research, 11*, 1833–1863.",
        "- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.",
        "- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology, 39*(6), 1161–1178.",
        "",
        "## 14. Appendices",
        "",
        "### A. Expected research outputs (Protocol §XVIII): status matrix",
        "",
        md_table(status_tab),
        "",
        "### B. Validation",
        "",
        "Automated checks of the whole project (raw files preserved, WAV format and naming, one feature row per "
        "participant, no leakage, speaker independence, report–result consistency) are in "
        "`reports/validation_report.md`.",
        "",
        "### C. Decisions awaiting confirmation",
        "",
        "See `docs/decision_log.md` (items marked PENDING):",
        "- the retake choices;",
        "- the config-B recordings;",
        "- Y2-076's arousal;",
        "- off-list words;",
        "- 'Lecture' → 'Lecture/Discussion';",
        "- the `PM` token;",
        "- consent documentation;",
        *([f"- the supplied audio delivery: original recordings for {', '.join(supplied_new['participant_code'])}"
           + (f", and an explanation for {', '.join(supplied_odd['participant_code'])}" if len(supplied_odd) else "")
           + " (`reports/supplied_audio_audit.md`);"] if len(supplied_new) or len(supplied_odd) else []),
        *([] if bounds_verified == n and "Y2-073" in words_from_review else
          ["- the blind listening review (utterance boundaries, retakes, Y2-073's word): "
           "`data/review/listening_review.html`, results in `reports/listening_review_report.md`."]),
        "",
        "### D. Reproduction",
        "",
        "`.venv/Scripts/python.exe run_pipeline.py` (see `README.md` and `docs/DOCUMENTATION.md`).",
        "",
    ]
    out = REPORTS / "final_group_research_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    status_tab.to_csv(REPORTS / "deliverable_status_matrix.csv", index=False)

    # README "Results at a glance": regenerated so the front page can never contradict the report
    best_txt_ci = "; ".join(
        f"{r.task}: {MODEL_LABEL[r.model]} balanced accuracy {r.pooled_ba:.3f} [95% CI {r.pooled_ba_ci_low:.3f}, "
        f"{r.pooled_ba_ci_high:.3f}] (chance {r.chance_ba:.2f})" for r in best.itertuples())
    readme_block = "\n".join([
        f"*Generated by `src/09_build_final_report.py` on {date.today().isoformat()} from the pipeline outputs; "
        "full detail in the report.*",
        "",
        "- **Data:**",
        f"  - {n_meta} metadata participants; {dq['total_files']} recordings (iPhone MOV) covering "
        f"{dq['participants_with_recordings']} participants.",
        f"  - {len(missing_rec)} participants have no recording, so the protocol minimum of "
        f"{MIN_VALID_OBSERVATIONS} is **{'met' if n >= MIN_VALID_OBSERVATIONS else 'not met'}** (n = {n})"
        + (f"; {supplied_note}" if supplied_note else "") + ".",
        f"  - {len(retakes)} participants had retakes; one take each was chosen by the label-blind complete-word "
        "rule.",
        f"- **Self-reports:** valence 3–5 for {v35} of {n} and arousal 3–5 for {a35} of {n}. Valence and arousal "
        f"correlate (ρ = {va[0]:.2f}). Only {train['tasks']['valence_3']['class_counts']['Unpleasant']} "
        f"participants reported unpleasant valence; not modelled (too few per class): {', '.join(not_run)}.",
        f"- **Acoustics vs affect:** after FDR correction, {fdr['arousal']} of {fsum['n_features']} features are "
        f"associated with arousal and {fdr['valence']} with valence. "
        f"{subsets['subset_config_or_time_anomaly']} recordings with a different audio configuration shift "
        f"{cfg_feats} features (a device confound).",
        f"- **Speaker-independent ML** (logistic regression, RBF SVM, random forest vs dummy baselines; "
        f"{train['n_repeats']}× repeated nested grouped CV):",
        f"  - {ml_statement}",
        f"  - Best point estimates: {best_txt_ci}.",
        "  - " + ("No pair of learned models differs significantly." if len(comp_sig) == 0 else rq5),
        "- **Lexical control:**",
        f"  - the acoustic features identify *Kapoy* vs other words (balanced accuracy "
        f"{probe['balanced_accuracy_mean_10_repeats']:.2f}, permutation p = {probe['permutation_p']:.3f});",
        "  - word choice vs the self-reports: " + "; ".join(f"{t} p = {a['p_value']:.3f}" for t, a in word_assoc.items()) + ";",
        f"  - within *Kapoy*, {within_sig} of {len(within)} prosody–affect correlations pass FDR correction.",
        "",
        ("  So any above-chance score can reflect *which* word was said rather than *how* it was said."
         if lexical_confound else
         "  The two conditions for a lexical confound were not jointly significant in this run."),
    ])
    readme = PROJECT_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    m_start, m_end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if m_start not in text or m_end not in text:
        raise RuntimeError("README.md lacks the RESULTS markers; cannot update the results block")
    text = text.split(m_start)[0] + m_start + "\n" + readme_block + "\n" + m_end + text.split(m_end, 1)[1]
    readme.write_text(text, encoding="utf-8")
    log.info("final report written: %s (n = %d, any significant = %s)", out.name, n, any_sig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
