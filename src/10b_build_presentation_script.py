"""
10b_build_presentation_script.py -- presenter script and defense guide for the results presentation.

Writes reports/presentation/presentation_script.md and a printable, self-contained
reports/presentation/presentation_script.html. The document follows the slides of G3_results_presentation.pptx:
what to say on each slide, every idea explained in plain words, the questions most likely to be asked with
honest answers, and the numbers to remember. slide_script() is also what 09c_build_presentation.py uses for the
slide titles and the speaker notes, so the deck and this script cannot disagree.

Every number is read from the pipeline's result files, so the script cannot disagree with the slides or the
report. Only the standard library is used, so it runs even where the project environment is not installed:
    python src/10b_build_presentation_script.py
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT_MD = REPORTS / "presentation" / "presentation_script.md"
OUT_HTML = REPORTS / "presentation" / "presentation_script.html"
LEARNED = ("logistic_regression", "svm_rbf", "random_forest")
MODEL = {"logistic_regression": "logistic regression", "svm_rbf": "SVM", "random_forest": "random forest"}
TASK = {"arousal_3": "three-level arousal (low, moderate, high)", "valence_2": "valence (pleasant vs not pleasant)",
        "arousal_2": "two-level arousal (high vs not high)"}
PRESPECIFIED = {"f0_median": "pitch level (F0 median)", "f0_range_st": "pitch range", "rms_db_mean": "loudness (RMS)",
                "utterance_duration_s": "duration", "spectral_centroid_hz_mean": "brightness (spectral centroid)",
                "voiced_ratio": "voiced ratio"}
SCENARIO = {"no_unverified_source": "without the files that have no verified original",
            "no_edited_or_clipped": "without the edited or clipped files",
            "no_config_B": "without the copies of recordings made with a different setting",
            "high_confidence_segmentation": "without the recordings whose word location is uncertain",
            "no_retakes": "without participants who were recorded more than once"}
# The analysis before decision D33 (original recordings only, commit 15df591). History, so it is fixed here; it is
# quoted only in the answer to "why did your numbers change".
EARLIER = {"n": 32, "fdr_arousal": 3}
# How the supplied files were made, as reported by the group on 2026-10-12 (decision D34). The pipeline cannot
# measure this; what it can measure about the conversion comes from the audit of the files against the originals.
PREPARED_WITH = "ChatGPT"


def table(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def facts() -> dict:
    """Everything the script quotes, read from the result files."""
    f: dict = {}
    manifest = table(ROOT / "data" / "analysis_manifest.csv")
    included = [m for m in manifest if m["include_primary"] == "True"]
    codes = {m["participant_code"] for m in included}
    meta = [m for m in table(ROOT / "data" / "metadata_cleaned.csv") if m["participant_code"] in codes]
    dq = load(REPORTS / "data_quality_summary.json")
    eda = load(REPORTS / "eda" / "eda_summary.json")
    lex = load(REPORTS / "lexical" / "lexical_summary.json")
    train = load(ROOT / "models" / "training_summary.json")
    feats = load(REPORTS / "features_summary.json")
    clean = load(REPORTS / "phase1" / "cleaning_summary.json")
    valid_file = REPORTS / "validation_results.json"  # written by step 10; absent on a first run
    valid = load(valid_file) if valid_file.exists() else []
    summary = table(REPORTS / "evaluation" / "summary_metrics.csv")
    sens = table(REPORTS / "evaluation" / "sensitivity_summary.csv")
    comp = table(REPORTS / "evaluation" / "model_comparison.csv")
    spear = table(REPORTS / "eda" / "spearman_feature_vs_rating.csv")
    groups = table(REPORTS / "lexical" / "oof_accuracy_by_word_group.csv")
    status = table(REPORTS / "deliverable_status_matrix.csv")

    f["n_meta"], f["n"] = len(manifest), len(included)
    f["supplied"] = dq.get("audio_source", "original") != "original"
    f["delivery"] = dq.get("audio_source", "")
    pick = lambda col: [m["participant_code"] for m in included if m.get(col) == "True"]  # noqa: E731
    f["no_orig"], f["edited"], f["cfg"] = pick("subset_no_original"), pick("subset_edited_or_clipped"), pick("config_b_origin")
    f["n_copies"] = sum(m.get("audio_source", "").endswith("COPY_OF_ORIGINAL") for m in included)
    f["unverified"] = [m["participant_code"] for m in included if m["word_in_filename"] == "UNVERIFIED"]
    flags = dq["flag_counts_selected"]
    f["bg"], f["low_sbr"], f["silence"] = flags.get("BACKGROUND_VOICES", 0), flags.get("LOW_SBR", 0), flags.get("DIGITAL_SILENCE", 0)
    f["seg_critical"] = dq["subsets"]["subset_segmentation_critical"]
    f["n_features"], f["feature_groups"] = feats["n_features"], feats["features_by_group"]
    f["words_missing"] = clean["flags"].get("SPOKEN_WORD_MISSING", 0)
    f["words_offlist"] = clean["flags"].get("WORD_NOT_IN_PROTOCOL_LIST", 0)
    words = [m["spoken_word"] for m in meta if m["spoken_word"]]
    f["words_with"], f["words_on_list"] = len(words), len(words) - sum(m["spoken_word_in_protocol_list"] == "False" for m in meta if m["spoken_word"])
    val = [int(m["valence"]) for m in meta if m["valence"]]
    aro = [int(m["arousal"]) for m in meta if m["arousal"]]
    f["v35"], f["n_val"], f["a35"], f["n_aro"] = sum(3 <= v <= 5 for v in val), len(val), sum(3 <= a <= 5 for a in aro), len(aro)
    f["no_arousal"] = [m["participant_code"] for m in meta if not m["arousal"]]
    f["va_rho"], f["va_n"] = eda["valence_arousal_spearman"]["analysis"][0], eda["valence_arousal_spearman"]["analysis"][2]
    f["targets"] = {d["task"]: d for d in eda["derived_targets"]}
    f["fdr"] = eda["significant_features_fdr05"]
    f["cfg_feats"] = eda["config_effect_features_fdr05"]
    f["n_repeats"], f["n_perm"] = train["n_repeats"], train["n_permutations"]

    pre = [r for r in spear if r["feature"] in PRESPECIFIED]
    f["pre_top"] = {k: max((r for r in pre if r["rating"] == k), key=lambda r: abs(float(r["spearman_rho"])))
                    for k in ("arousal", "valence")}
    f["pre_sig"] = sum(float(r["q_value_bh"]) < 0.05 for r in pre)

    z = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)  # two-sided alpha .05, power .80
    f["rho_min"] = math.tanh(z * math.sqrt(1.06 / (f["n"] - 3)))     # Spearman: Fieller variance factor 1.06
    f["n_for_03"] = math.ceil(3 + 1.06 * (z / math.atanh(0.3)) ** 2)

    tasks = []
    for task in dict.fromkeys(r["task"] for r in summary):
        rows = [r for r in summary if r["task"] == task]
        best = max((r for r in rows if r["model"] in LEARNED), key=lambda r: float(r["pooled_ba"]))
        tasks.append({"task": task, "n": int(best["n"]), "chance": float(best["chance_ba"]), "model": best["model"],
                      "ba": float(best["pooled_ba"]), "lo": float(best["pooled_ba_ci_low"]),
                      "hi": float(best["pooled_ba_ci_high"]), "p": float(best["perm_p"]), "holm": float(best["perm_p_holm"]),
                      "acc": float(best["pooled_accuracy"])})
    f["tasks"] = tasks
    tests = [r for r in summary if r["perm_p"]]
    f["n_tests"] = len(tests)
    f["raw_sig"] = [r for r in tests if float(r["perm_p"]) < 0.05]
    f["smallest"] = min(tests, key=lambda r: float(r["perm_p"]))
    f["any_holm"] = any(float(r["perm_p_holm"]) < 0.05 for r in tests)
    f["pairs_sig"] = [r for r in comp if r["model_a"] in LEARNED and r["model_b"] in LEARNED and float(r["p_value"]) < 0.05]

    lead = tasks[0]  # the first primary task: how its best model moves in each sensitivity scenario
    f["sens"] = {t["task"]: {r["scenario"]: (float(r["pooled_ba"]), int(r["n"])) for r in sens
                             if r["task"] == t["task"] and r["model"] == t["model"]} for t in tasks}
    f["lead"] = lead
    probe, assoc = lex["probe"], lex["word_affect_association"]
    f["probe"], f["assoc_v"], f["assoc_a"] = probe, assoc["valence_2"], assoc.get("arousal_3")
    f["within_sig"], f["within_tests"] = lex["within_word_significant_q05"], lex["within_word_tests"]
    by_group = {}
    for r in groups:
        if r["task"] == "arousal_2" and r["model"] in LEARNED:
            by_group.setdefault(r["word_group"], []).append(float(r["accuracy_over_repeats"]))
    f["kapoy_acc"] = {g: sum(v) / len(v) for g, v in by_group.items()}
    f["valid"] = {s: sum(1 for x in valid if x["status"] == s) for s in ("PASS", "BLOCKED", "WARNING", "FAIL")}
    f["blocked_items"] = [r["Deliverable"] for r in status if r["Status"] != "COMPLETE"]
    f["complete"], f["n_deliverables"] = sum(r["Status"] == "COMPLETE" for r in status), len(status)
    f["flags_all"], f["sheet_flags"], f["changes_by_rule"] = flags, clean["flags"], clean.get("changes_by_rule", {})
    f["train_minutes"] = round(train.get("runtime_s", 0) / 60)
    cfg_file = REPORTS / "eda" / "recording_configuration_effect.csv"  # config A vs B, one row per feature
    cfg_rows = table(cfg_file) if cfg_file.exists() else []
    f["cfg_raw"] = sum(float(r["p_value"]) < 0.05 for r in cfg_rows)
    f["cfg_min_q"] = min((float(r["q_value_bh"]) for r in cfg_rows), default=None)
    learned = [r for r in summary if r["model"] in LEARNED]
    f["n_learned"] = len(learned)
    f["crossing"] = sum(float(r["pooled_ba_ci_low"]) <= float(r["chance_ba"]) for r in learned)
    f["lazy_acc"] = {r["task"]: float(r["pooled_accuracy"]) for r in summary if r["model"] == "dummy_majority"}
    audit = REPORTS / "supplied_audio_audit.csv"
    rows = ([r for r in table(audit) if r["delivery"] == f["delivery"]] if f["supplied"] and audit.exists() else [])
    had = {r["participant_code"] for r in rows if r["verdict"] != "NEW_RECORDING"}
    f["with_original"] = len(had) or None  # participants for whom an original recording exists
    copies = [r for r in rows if r["verdict"] == "COPY_OF_ORIGINAL"]
    col = lambda k: [float(r[k]) for r in copies if r[k] not in ("", "nan")]  # noqa: E731
    f["conversion"] = ({"n": len(copies), "ncc_min": min(col("best_ncc")), "dur": (min(col("duration_s")), max(col("duration_s"))),
                        "level_unchanged": all(abs(g - 1.0) < 0.01 for g in col("gain"))}
                       if copies else None)  # what the audit shows the conversion did to the verified copies
    pending = REPORTS / "sheet_update_pending.json"  # written by step 00b while a newer sheet waits to be analysed
    f["pending_sheet"] = load(pending) if pending.exists() else None
    applied = REPORTS / "sheet_update_applied.json"  # step 00b: the changes of a completed sheet that is now in use
    f["applied_sheet"] = load(applied) if applied.exists() else None
    f["probe_sig"] = probe["permutation_p"] < 0.05  # can the audio alone tell Kapoy from the other words?
    f["assoc_sig"] = f["assoc_v"]["p_value"] < 0.05
    others = Counter(w.strip().lower() for w in words if w.strip().lower() != "kapoy")
    f["max_other_word"] = max(others.values(), default=0)
    return f


def pending_text(f: dict) -> dict | None:
    """Counts and phrases about a newer sheet that is not analysed yet (step 00b), or None when there is none."""
    p = f.get("pending_sheet")
    if not p:
        return None
    day = date.fromisoformat(p["received_on"]).strftime("%d %B").lstrip("0") if p.get("received_on") else "a later date"
    return {"day": day, "words": len(p["words_filled"]), "word_changes": len(p["words_changed"]),
            "ratings": len(p["ratings_changed"]), "codes": ", ".join(p["ratings_changed"]),
            "words_source": p.get("words_source", ""), "ratings_source": p.get("ratings_source", "")}


def applied_text(f: dict) -> dict | None:
    """The same for a completed sheet that is now the sheet in use, with the results the sheet before it gave."""
    p = f.get("applied_sheet")
    if not p:
        return None
    out = pending_text({"pending_sheet": p})
    out["before"] = p.get("results_before", {})
    out["all_unverified"] = bool(p["ratings_changed"]) and set(p["ratings_changed"]) <= set(f["no_orig"])
    return out


def codes(xs) -> str:
    return ", ".join(xs) if xs else "none"


def headline(any_supported: bool) -> str:
    """The conclusion in one sentence. The deck (09c) and this script both use it."""
    return ("For some targets, the voice features carry information about the ratings that holds after correction"
            if any_supported else "Our data do not show that post-class feelings can be recognised from the voice")


def derive(f: dict) -> SimpleNamespace:
    """Values that both the slide script and the rest of the guide quote."""
    by_task = {t["task"]: t for t in f["tasks"]}
    lead_sens = f["sens"][f["lead"]["task"]]
    sm = f["smallest"]  # the learned model with the smallest p-value: the result closest to a positive finding
    top = {"task": sm["task"], "model": sm["model"], "n": int(sm["n"]), "ba": float(sm["pooled_ba"]),
           "lo": float(sm["pooled_ba_ci_low"]), "hi": float(sm["pooled_ba_ci_high"]), "chance": float(sm["chance_ba"]),
           "p": float(sm["perm_p"]), "holm": float(sm["perm_p_holm"])}
    is_task_best = by_task.get(sm["task"], {}).get("model") == sm["model"]  # the sensitivity runs use the task's best model
    return SimpleNamespace(
        top=top, top_no_src=f["sens"].get(sm["task"], {}).get("no_unverified_source") if is_task_best else None,
        near=not f["any_holm"] and 0.05 <= top["holm"] < 0.10,
        n=f["n"], a3=by_task.get("arousal_3"), v2=by_task.get("valence_2"), sm=f["smallest"],
        exp_false=0.05 * f["n_tests"], kapoy=f["assoc_v"]["table"]["Kapoy"], other=f["assoc_v"]["table"]["other"],
        probe=f["probe"], lead=f["lead"], lead_sens=lead_sens, val_sens=f["sens"].get("valence_2", {}),
        no_src=lead_sens.get("no_unverified_source"), fg=f["feature_groups"], headline=headline(f["any_holm"]))


def slide_script(f: dict) -> list[dict]:
    """One entry per slide of the deck, in order: key, title, seconds, say, point, plain, asked.

    09c_build_presentation.py takes every slide title and its speaker notes from here, so the deck and this
    script cannot disagree."""
    d = derive(f)
    n, a3, sm, exp_false, kapoy, other, probe, no_src, fg = (d.n, d.a3, d.sm, d.exp_false, d.kapoy, d.other, d.probe,
                                                             d.no_src, d.fg)
    S: list[dict] = []
    pt, at = pending_text(f), applied_text(f)
    top, top_no_src = d.top, d.top_no_src

    def add(key, title, seconds, say, point, plain=None, asked=()):
        S.append({"key": key, "title": title, "seconds": seconds, "say": say, "point": point, "plain": plain,
                  "asked": list(asked)})

    add("title", "Title", 20, [
        "Good day. We are Group 3. Our project asks one simple question: can a single spoken word, said right after "
        "class, tell us how a student feels?",
        "We studied second-year Computer Science students. We will show what we collected, how we analysed it, what we "
        "found, and what we could not show."],
        "Nothing yet. Look at the audience.")
    add("question", "What we asked", 60, [
        "Right after class, each participant said one Bisaya word that matched their reaction, for example Kapoy, "
        "Salamat or Lingaw. Then they rated how they felt on two scales from 1 to 5.",
        "Valence is how pleasant the feeling is. Arousal is how much energy it has: calm or tired at the low end, "
        "excited or tense at the high end.",
        f"Our question: can the sound of the word, not its meaning, tell us those ratings? From each recording the "
        f"computer took {f['n_features']} voice measurements.",
        "And we made the test strict. The computer is always tested on people it has never heard, because a real "
        "system would meet new students."],
        f"The three tiles on the right: 1 word, 2 ratings, {f['n_features']} measurements.",
        "The correct answer is the participant's own rating. We never decide the feeling from the meaning of the word. "
        "Someone can say *Kapoy* and still report feeling fine. The technical name for the strict test is "
        "*speaker-independent*.",
        [("Why use self-report as the correct answer?",
          "The protocol defines it that way. The feeling belongs to the participant. A listener's impression, or the "
          "meaning of the word, would be our own assumption.")])
    add("data", "The data we had", 75, [
        f"The sheet lists {f['n_meta']} participants and we analysed " + (f"all {n}" if n == f["n_meta"] else str(n))
        + ", which " + ("meets" if n >= 40 else "is below") + " the protocol minimum of 40.",
        (f"We want to be open about the audio. We recorded phone videos, and a group member converted them to short "
         f"16 kHz WAV files. Those files are what we analysed. We checked every file against the original recordings. "
         f"{f['n_copies']} match an original. {len(f['no_orig'])} could not be checked, and {len(f['edited'])}"
         + (" of those" if set(f["edited"]) <= set(f["no_orig"]) else "")
         + " show signs of editing, such as stretches of exact silence. We kept them, marked them, and ran the models "
         "again without them. You will see that result later."
         if f["supplied"] else
         "The analysis uses the original phone recordings."),
        (f"{f['words_missing']} spoken words are blank in the sheet. We never guessed a word from the feeling text; "
         "those files carry the label UNVERIFIED." if f["words_missing"] else
         "Every participant has a spoken word in the sheet. No word was guessed from the feeling text."),
        *([f"The group completed the sheet on {at['day']}: {at['words']} blank words were filled in {at['words_source']}, "
           f"and the ratings of {at['ratings']} participants were corrected {at['ratings_source']}. The results we show "
           "use the completed sheet. The earlier sheet and its results are kept, so the two can be compared."]
          if at else []),
        *([f"One update. On {pt['day']} the group completed the sheet: {pt['words']} blank words are now filled in "
           f"{pt['words_source']}, and the ratings of {pt['ratings']} participants were corrected "
           f"{pt['ratings_source']}. The results we show today use the earlier sheet. Running the analysis again "
           "with the completed sheet is our first next step."] if pt else []),
        f"The recordings were made in a hallway right after class, so they are noisy: we detected background voices in "
        f"{f['bg']} of {n}.",
        "And consent: the sheet says Yes for everyone, but the log of the signed forms still has to be completed."],
        "The four tiles, then the first bullet about the audio.",
        "*Could not be checked* means we have no original recording to compare with, so we cannot prove when the file "
        "was recorded or that it is unedited. *Exact silence* is a stretch of perfect zeros. A real phone recording "
        "always contains a little noise, so perfect zeros mean the file was edited.",
        [("Why did you analyse copies and not the original recordings?",
          (f"We had original recordings for only {f['with_original']} of the {f['n_meta']} participants. "
           if f["with_original"] else "The original recordings covered fewer participants. ")
          + f"The version-2 set covers all {n}, so the group decided to use it for every requirement. We recorded that "
          "decision, kept the originals untouched, audited every file against them, and report the results with and "
          "without the uncertain files."),
         ("Did you use the denoised files?",
          "No. The protocol forbids denoising, and denoising changes exactly the properties we measure. We analysed "
          "only the non-denoised set, and the pipeline passes those files through sample for sample.")])
    add("pipeline", "How we analysed it", 70, [
        "Everything runs with one command, in eight steps.",
        "Check: every file and every sheet cell is checked, and nothing is changed. Clean: fixes are made by fixed "
        "rules only, and each fix is logged. Find the word: the computer finds the spoken word inside each clip "
        f"without ever seeing the ratings. Measure: {f['n_features']} numbers for each voice.",
        "Then we explore the data, train the models, test whether the scores are better than luck, and run the word "
        "check, which asks whether the model hears the feeling or only the word.",
        "The three rules at the bottom keep the test fair. No participant is in both the training and the test data. "
        "The model's settings are learned from the training data only. And the ratings are never used to cut or "
        "choose audio."],
        "Walk along the eight boxes with your hand, then point at the three rules.",
        "The three rules prevent *data leakage*: the model must not see the answers before the exam. If we prepared "
        "the measurements using all the data, a little information about the test participants would leak into "
        "training.",
        [(f"What are the {f['n_features']} features?",
          "; ".join(f"{k} ({v})" for k, v in fg.items()) + ". Pitch is how high the voice is, energy is loudness, and "
          "the MFCCs describe the shape of the sound spectrum, which is what makes voices and vowels sound different.")])
    add("ratings", "What the participants reported", 45, [
        f"Most participants reported neutral to pleasant feelings and moderate to high energy. The two ratings rise "
        f"together: the correlation is {f['va_rho']:.2f}.",
        "Some scores were chosen by very few people, so we cannot model all five scores. We grouped them, and we fixed "
        "the groups before modelling: arousal into low, moderate and high; valence into pleasant and not pleasant; "
        "and arousal into high and not high.",
        "We modelled a grouping only if every group has at least five people."],
        "The two bar charts, then the groupings on the right.",
        f"Valence 3–5: {f['v35']} of {f['n_val']}. Arousal 3–5: {f['a35']} of {f['n_aro']}. Group sizes: "
        + "; ".join(f"{t}: {f['targets'][t]['classes']}" for t in ("arousal_3", "valence_2", "arousal_2") if t in f["targets"])
        + (f". Arousal has one person fewer because {codes(f['no_arousal'])}'s arousal rating is crossed out in the "
           "sheet, so we did not use it." if f["no_arousal"] else ".")
        + (" In the chart, the blue and orange bars are the same height because every participant in the sheet is "
           "analysed." if n == f["n_meta"] else ""),
        [("Why not predict all 25 combinations of valence and arousal?",
          f"With {n} people, most of the 25 cells would be empty or hold one person. A model cannot learn or be tested "
          "on a group with one example."),
         ("Did you invent these groups?",
          "No. They are groupings of the participants' own 1-to-5 ratings. The original ratings stay in the data, and "
          "the groupings are marked as derived.")])
    pa, pv = f["pre_top"]["arousal"], f["pre_top"]["valence"]
    no_link = not f["pre_sig"] and not f["fdr"]["arousal"] and not f["fdr"]["valence"]
    add("features", "Simple voice measures show no clear link" if no_link else "Voice measures and the ratings", 60, [
        "Before any machine learning, we checked simple links. We chose six well-known voice features in advance: "
        "pitch level, pitch range, loudness, duration, brightness and the share of voiced sound.",
        f"{'None of them is' if not f['pre_sig'] else str(f['pre_sig']) + ' of them are'} significantly linked to the "
        f"ratings. The strongest are {PRESPECIFIED[pa['feature']]} with arousal at {float(pa['spearman_rho']):+.2f} and "
        f"{PRESPECIFIED[pv['feature']]} with valence at {float(pv['spearman_rho']):+.2f}. Those are weak.",
        f"Across all {f['n_features']} features, {f['fdr']['arousal']} pass the correction for testing many features at "
        f"once for arousal, and {f['fdr']['valence']} for valence.",
        f"With {n} people, only links of about {f['rho_min']:.2f} or stronger can be found reliably. A moderate link of "
        f"0.3 would need about {f['n_for_03']} participants."],
        "The table, then the last bullet about sample size.",
        "The symbol ρ (rho) is the Spearman correlation: +1 means the two rise together perfectly, 0 means no link, "
        "−1 means one rises as the other falls. We use Spearman because the ratings are ranks from 1 to 5.",
        [("Why choose six features in advance?",
          "If you test many features and report only the best ones, you will always find something by luck. Choosing "
          "in advance keeps us honest. We still report all of them, with a correction for testing many features.")])
    add("reading", "How to read our results", 50, [
        "Before the results, five ideas.",
        "Chance is the score you get by pure guessing: "
        + ", ".join(f"{1 / k:.2f} with {k} groups"
                    for k in sorted({round(1 / t["chance"]) for t in f["tasks"]}, reverse=True)) + ".",
        "Balanced accuracy is the average of the scores for each group. A model cannot look good by always giving the "
        "most common answer.",
        f"The 95% interval is the range where the true score probably lies. Ours are wide, because {n} people is few.",
        "The p-value tells us how often shuffled ratings, which mean nothing, score as high as the real ones. A small "
        "p-value means the score is unlikely to be luck.",
        f"And the correction. We ran {f['n_tests']} tests, and with {f['n_tests']} tries a lucky result is expected. The "
        "Holm correction accounts for that."],
        "Each row, as you explain it.",
        None,
        [("Why is the limit 0.05?",
          "It is the usual limit in research. It means we accept a 5% risk of calling luck a real effect.")])
    task_lines = "; ".join(f"{t['ba']:.2f} for {TASK.get(t['task'], t['task'])}, where chance is {t['chance']:.2f}"
                           for t in f["tasks"])
    same_best = len({t["model"] for t in f["tasks"]}) == 1
    interval = (f"The 95% interval runs from {top['lo']:.2f} to {top['hi']:.2f}"
                + (", so the low end is at the chance level." if top["lo"] <= top["chance"] else ", which is wide."))
    add("models", "No model beats chance after correction" if not f["any_holm"]
        else "Some models beat chance after correction", 70, [
        "Now the main test. For each target we show the best of three models.",
        (f"The {MODEL[f['tasks'][0]['model']]} was best for every target: " if same_best else "The best scores: ")
        + f"{task_lines}.",
        f"Is that better than luck? We shuffled the ratings {f['n_perm']} times and ran the whole procedure again each "
        f"time. {len(f['raw_sig'])} of {f['n_tests']} tests "
        + ("has" if len(f["raw_sig"]) == 1 else "have") + f" a p-value below 0.05: the smallest is "
        f"{top['p']:.3f}, for the {MODEL[top['model']]} on {TASK.get(top['task'], top['task'])}.",
        f"But we ran {f['n_tests']} tests. With {f['n_tests']} tries, about {exp_false:.2f} lucky results are expected. "
        f"After the Holm correction that p-value becomes {top['holm']:.3f}, which is "
        + ("still significant." if top["holm"] < 0.05 else "not significant.")
        + (" So no model stays significant after the correction." if not f["any_holm"] else ""),
        *(["That is close to 0.05, and we say so. But the limit was fixed before we looked at the results"
           + (f", and this score rests on files we could not check: without them it is {top_no_src[0]:.2f}."
              if top_no_src else ".")] if d.near else [])],
        "The p-value column, then the column *p after correction*.",
        "The *Chance* column is what a model scores if it always gives the same answer. "
        f"Holm in one line: multiply the smallest p-value by the number of tests. "
        f"{top['p']:.4f} × {f['n_tests']} = {top['p'] * f['n_tests']:.3f}.",
        [(f"Isn't {top['ba']:.2f} against {top['chance']:.2f} a good result?",
          f"It is our most promising number, but three things stop us from claiming it. {interval} It does not pass the "
          f"correction for {f['n_tests']} tests: its p-value is {top['holm']:.3f} after it."
          + (f" And without the {len(f['no_orig'])} files that could not be checked, the score is {top_no_src[0]:.2f}."
             if top_no_src else ""))]
        + ([(f"That interval starts at {top['lo']:.2f}, which is above {top['chance']:.2f}. Doesn't that mean it beats "
             "chance?",
             f"For this one test, yes: the interval and the p-value before correction ({top['p']:.3f}) say the same "
             f"thing. But it is one of {f['n_tests']} tests, and with that many tests about {exp_false:.2f} lucky results "
             f"are expected. After the correction the p-value is {top['holm']:.3f}, which is above 0.05"
             + (f", and the score is {top_no_src[0]:.2f} when the unchecked files are left out." if top_no_src else "."))]
           if top["lo"] > top["chance"] and not f["any_holm"] else [])
        + ([(f"{top['holm']:.3f} is almost 0.05. Why not call it significant?",
             "Because the limit was set before we saw the results. Moving it afterwards to fit a result is exactly what "
             "the correction is there to prevent. We report the number as it is: close, but not significant"
             + (f". It also rests on files we could not check: without them the score is {top_no_src[0]:.2f}."
                if top_no_src else "."))] if d.near else []) + [
         ("Why three models? Which is best?",
          f"A linear model (logistic regression), a kernel model (SVM) and a tree ensemble (random forest) make different "
          f"assumptions, so together they are a fair test. The {MODEL[top['model']]} gave our best result, but "
          + (f"only {len(f['pairs_sig'])} of the pairwise comparisons has p below 0.05, uncorrected, so we do not claim one "
             "model is truly better." if f["pairs_sig"] else "no pair of models differs significantly."))])
    clear = f["n_learned"] - f["crossing"]
    add("figure", "The same result as a picture", 40, [
        "This figure shows the same result as a picture, with one panel for each target. Each dot is a model's "
        "balanced accuracy. The horizontal bar is its 95% interval. The vertical line is chance.",
        "The grey rows are baselines that never listen to the audio. The blue rows are our models.",
        f"The bars are wide. {f['crossing']} of the {f['n_learned']} blue bars cross the chance line."
        + (f" {clear} " + ("bar clears" if clear == 1 else "bars clear") + " it, but none of those results passes the "
           "correction." if clear and not f["any_holm"] else "")
        + " This is what an inconclusive result looks like."],
        "One blue bar that crosses the vertical line.",
        "A wide bar means the score is uncertain. If the bar crosses the chance line, the true score could be no "
        "better than guessing.")
    add("word", "Is it the voice, or just the word?", 75, [
        "This is our most important finding. Is the model hearing the feeling, or only recognising the word?",
        "First test: can the computer tell Kapoy from the other words by sound alone? "
        + (f"Yes. Balanced accuracy {probe['balanced_accuracy_mean_10_repeats']:.2f}, with p = "
           f"{probe['permutation_p']:.3f}." if f["probe_sig"] else
           f"Only weakly. Balanced accuracy {probe['balanced_accuracy_mean_10_repeats']:.2f}, with p = "
           f"{probe['permutation_p']:.3f}, which is not significant."),
        "Second test: is the word linked to the feeling? " + ("Yes. " if f["assoc_sig"] else "")
        + f"Of {sum(kapoy.values())} Kapoy speakers, {kapoy['NotPleasant']} felt not pleasant. Of "
        f"{sum(other.values())} speakers of other words, {other['NotPleasant']} did.",
        ("So a model can score above chance by recognising Kapoy and answering *not pleasant*, without hearing any "
         "feeling at all." if f["probe_sig"] else
         "So the word and the feeling go together. A model that picks up the word even partly could score above "
         "chance without hearing the feeling, and with this data we cannot separate the two."),
        f"When we look at Kapoy speakers alone, the same word with different feelings, {f['within_sig']} of "
        f"{f['within_tests']} links between voice and rating are significant."],
        f"The small table: Kapoy {kapoy['NotPleasant']} and {kapoy['Pleasant']}, other words {other['NotPleasant']} and "
        f"{other['Pleasant']}.",
        "This is called a *confound*: a hidden third thing that explains the link. Here the word explains both the "
        "sound and the feeling. It is like concluding that umbrellas cause rain because they appear together.",
        [("How would you fix the word problem?",
          "Ask every participant to say the same word, or collect enough recordings of each word to analyse each word "
          "separately. Both are in our plan for a larger study."),
         ("Why only Kapoy?",
          f"It is the only word said by enough people ({probe['n_kapoy']}). Every other word was said by "
          f"{f['max_other_word']} people or fewer, which is too few to analyse.")])
    add("limits", "Limits of this study", 55, [
        f"We want to be clear about the limits. The sample is small. With {n} people only strong links can be found.",
        (f"The audio is converted copies, not the original recordings: {len(f['no_orig'])} could not be checked and "
         f"{len(f['edited'])} show signs of editing." if f["supplied"] else "Some recordings were retaken."),
        f"The recordings are noisy: we detected background voices in {f['bg']} of {n}, in a hallway right after class."
        + (f" And {len(f['cfg'])} files come from recordings made with a different audio setting"
           + ((". Some features look different for those files, but with so few files the difference does not "
               "pass the correction. We also ran the models without them." if f["cfg_raw"] else
               ", and we found no difference for them.") if not f["cfg_feats"] else ".")
           if f["cfg"] else ""),
        "Each person said only one word, so we cannot separate a person's normal voice from their feeling at that "
        "moment.",
        "Everyone came from the same year level, class activity and session, so we cannot compare contexts. And we "
        "measured links, not causes."],
        "The first three bullets.",
        f"*Power* is the chance of finding an effect that really exists. With {n} participants our study can only find "
        f"links of about {f['rho_min']:.2f} or more."
        + (f" *A different audio setting* ({len(f['cfg'])} files) means the original was saved in stereo with a second "
           f"audio track, unlike the others. {f['cfg_raw']} of {f['n_features']} features differ between the two "
           f"settings before the correction and {f['cfg_feats']} after it"
           + (f"; the closest corrected p-value is {f['cfg_min_q']:.3f}." if f["cfg_min_q"] is not None else ".")
           if f["cfg"] else ""))
    add("conclusion", "Conclusion", 45, [
        f"Our conclusion: {d.headline[0].lower() + d.headline[1:]}.",
        "That is a real result, not a failure. A careful analysis that finds no reliable signal is more useful than an "
        "impressive number that would not hold up.",
        "We contribute three things. A complete, tested pipeline that repeats the whole analysis with one command. The "
        "finding that the chosen word is linked to the feeling and must be controlled. And a clear statement of how "
        f"much data a real study needs: about {f['n_for_03']} participants."],
        "The blue sentence.")
    add("next", "Next steps", 30, [
        "Our next steps are concrete. " + ("First, run the analysis again with the completed sheet. " if pt else "")
        + "Complete the consent log from the signed forms. Have two group members listen to "
        "every clip without seeing the ratings. "
        + ("Find the original recordings for the files we could not check. " if f["supplied"] else "")
        + "Record more participants with one device, one setting and one take. Follow the analysis plan we wrote "
        "before seeing new data. And confirm the open protocol questions with our instructor.",
        "Thank you. We are happy to take your questions."],
        "The audience.")
    return S


def build(f: dict) -> str:
    d = derive(f)
    n, a3, v2, sm, exp_false, kapoy, other, probe = d.n, d.a3, d.v2, d.sm, d.exp_false, d.kapoy, d.other, d.probe
    lead, lead_sens, val_sens, no_src, fg = d.lead, d.lead_sens, d.val_sens, d.no_src, d.fg
    headline = d.headline[0].lower() + d.headline[1:]
    slides = slide_script(f)
    num = {s["key"]: i for i, s in enumerate(slides, 1)}
    last = len(slides)
    pt, at = pending_text(f), applied_text(f)
    top, top_no_src = d.top, d.top_no_src

    total = sum(s["seconds"] for s in slides)
    L = ["# Presentation script and defense guide", "",
         f"*Group 3 · Bisaya post-class speech and self-reported affect · {n} participants"
         + (f" · audio: {f['delivery']}" if f["supplied"] else "") + ". Built by `src/10b_build_presentation_script.py` "
         "from the result files, so every number matches the slides and the report.*", "",
         "## How to use this", "",
         f"- **Part 1** is what to say, slide by slide. It takes about {round(total / 60)} minutes.",
         "- **Part 2** explains every idea in plain words. Read it once before you present.",
         "- **Part 3** has the questions you are most likely to get, with honest answers.",
         "- **Part 4** is the short list of numbers to remember.",
         "- **Part 5** lists what not to say, and what to say instead.",
         "- Don't read the script word for word. Learn the first sentence of each slide and the number you point at.",
         f"- For four speakers, split the slides 1–{num['data']}, {num['pipeline']}–{num['reading']}, "
         f"{num['models']}–{num['word']} and {num['limits']}–{last}.",
         f"- If you are given only five minutes, present slides {num['question']}, {num['data']}, {num['models']}, "
         f"{num['word']} and {num['conclusion']} and skip the rest.",
         "- The speaker notes under each slide in PowerPoint are the same words as Part 1.", "",
         "## The whole story in one minute", "",
         f"> We asked {n} second-year CS students to say one Bisaya word right after class and to rate how they felt. We "
         f"measured {f['n_features']} properties of each voice clip and trained three kinds of models to guess the feeling "
         "from the voice alone, always testing on students the model had never heard.",
         ">",
         f"> The models did not beat guessing once we corrected for running {f['n_tests']} tests. We also found a trap: "
         "the word people choose is itself linked to how they feel. "
         + ("The computer can tell which word was said, so a model can look smart by recognising the word, not the "
            "emotion." if f["probe_sig"] else
            "So even a model that picks up the word only partly could look smart without hearing the emotion."),
         ">",
         f"> Our honest conclusion: with {n} noisy one-word recordings, {headline}. What we built is a complete, tested "
         "pipeline that re-runs everything with one command when better data arrives.", "",
         "## Part 1 · What to say, slide by slide", ""]
    for i, s in enumerate(slides, 1):
        say, point, plain, asked = s["say"], s["point"], s["plain"], s["asked"]
        L += [f"### Slide {i} · {s['title']}", "", f"**Say** (about {s['seconds']} seconds)", ""]
        for i, para in enumerate(say):
            L += ([">"] if i else []) + [f"> {para}"]
        L += ["", f"**Point at:** {point}", ""]
        if plain:
            L += [f"**In plain words:** {plain}", ""]
        for q, a in asked:
            L += [f"**If asked: “{q}”** {a}", ""]

    L += ["## Part 2 · Every idea in plain words", "",
          "### Valence and arousal", "",
          "- **What it is:** two ways to describe a feeling. Valence is unpleasant to pleasant. Arousal is low energy to "
          "high energy.",
          "- **Examples:** tired is low arousal; excited is high arousal and pleasant; stressed is high arousal and "
          "unpleasant.",
          f"- **In our project:** each participant rated both from 1 to 5. The two ratings rise together (ρ = {f['va_rho']:.2f}).", "",
          "### Groups (derived classes)", "",
          "- **What it is:** groups made from the 1-to-5 ratings, because some single scores have too few people. The "
          "slides say *group*; the technical word is *class*.",
          "- **The rule:** a grouping is modelled only if every group has at least 5 people.",
          "- **In our project:** " + "; ".join(f"`{t}` ({f['targets'][t]['classes']})" for t in f["targets"] if f["targets"][t]["viable"])
          + ". Not modelled: " + ", ".join(f"`{t}`" for t in f["targets"] if not f["targets"][t]["viable"]) + ".", "",
          "### Acoustic features", "",
          "- **What it is:** numbers that describe a sound, measured by the computer.",
          "- **Pitch (F0):** how high or low the voice is.",
          "- **Energy (RMS):** how loud it is.",
          "- **Zero-crossing rate and spectral centroid:** how noisy or bright the sound is.",
          "- **MFCCs:** 13 numbers that describe the shape of the sound spectrum. The *delta* versions describe how fast "
          "they change. For each we keep the average and the spread over the word, so 13 × 2 = 26 per group.",
          f"- **In our project:** {f['n_features']} features per recording, measured only on the located word: "
          + " + ".join(str(v) for v in fg.values()) + f" = {sum(fg.values())}.",
          "- **A caution about pitch:** pitch in Hz mostly shows whether a person's voice is naturally high or low. With "
          "one recording per person we cannot subtract their normal voice.",
          "- **Tools:** Python, with Praat (through parselmouth) for pitch, librosa for the spectrum and MFCCs, and "
          "scikit-learn for the models.", "",
          "### Finding the word in the clip", "",
          "- **What it is:** each clip also contains hallway noise, so the computer looks for the loudest short event and "
          "treats that as the word.",
          "- **Why it matters:** it never uses the labels, so it cannot cheat.",
          f"- **In our project:** {f['seg_critical']} of {n} recordings are flagged as uncertain, for example because the "
          "word touches the edge of the file. Nobody has yet checked them by listening.", "",
          "### Speaker-independent testing and cross-validation", "",
          "- **What it is:** we split the participants into 5 groups, train on 4 and test on the fifth, and rotate so "
          "everyone is tested once.",
          "- **Like:** practising with four chapters and being examined on the fifth, five times.",
          f"- **Repeats:** we repeat this {f['n_repeats']} times with different splits, so one lucky split cannot decide the result.",
          "- **Nested:** inside each training set there is another small split used only to choose the model's settings. "
          "The test group is never used for that.",
          "- **Speaker-independent:** nobody is in both the training and the test group. Here each person has one "
          "recording, so it holds automatically, and the code checks it anyway.", "",
          "### Data leakage", "",
          "- **What it is:** information about the test data slipping into training. It makes results look better than "
          "they are.",
          "- **How we prevent it:** only acoustic features go into the model; scaling and filling of missing values are "
          "fitted on the training folds only; the code stops if a non-acoustic column appears.", "",
          "### The models and the baselines", "",
          "- **Logistic regression:** draws straight boundaries between classes. The simplest of the three.",
          "- **SVM:** can draw curved boundaries.",
          "- **Random forest:** 300 decision trees that vote.",
          "- **Dummy baselines:** models that never look at the audio. One always answers the most common class; one "
          "guesses at random in the same proportions as the classes. A real model must beat them.",
          "- **Class weights:** each class counts equally even when it has fewer people. We did not create artificial "
          "samples (no SMOTE).", "",
          "### Balanced accuracy", "",
          "- **What it is:** the average of the scores for each class.",
          f"- **Why not plain accuracy:** in our {TASK.get(lead['task'], lead['task']).split(' (')[0]} task, always answering "
          f"the most common class is right {f['lazy_acc'].get(lead['task'], 0):.0%} of the time without learning anything. "
          f"Balanced accuracy gives that lazy model exactly the chance level, {lead['chance']:.2f}.",
          "- **Chance level:** 1 ÷ number of classes. 0.33 for three classes, 0.50 for two.", "",
          "### Confidence interval", "",
          "- **What it is:** the range where the true value probably lies.",
          "- **How we get it:** we re-draw the participants at random, with replacement, 2000 times, and recompute the "
          "score each time. The middle 95% of those scores is the interval. This is called a bootstrap.",
          f"- **In our project:** our best result has an interval from {top['lo']:.2f} to {top['hi']:.2f}. That is wide, "
          f"because {top['n']} people is few.", "",
          "### p-value and the permutation test", "",
          "- **What it is:** we shuffle the labels so they no longer belong to the right recordings, then run the whole "
          f"procedure again. We do that {f['n_perm']} times.",
          "- **The p-value:** the share of shuffles that score as high as the real data. A small p means the real score "
          "would be rare by luck.",
          f"- **In our project:** the smallest p is {float(sm['perm_p']):.3f}: about {round(float(sm['perm_p']) * 100)} in "
          "100 shuffles did as well.", "",
          "### Multiple testing: Holm and FDR", "",
          "- **The problem:** if you run many tests, some will look significant by luck. Roll a die many times and a six "
          "is no surprise.",
          f"- **Holm:** the correction we use for the {f['n_tests']} model tests. The smallest p is multiplied by "
          f"{f['n_tests']}: {float(sm['perm_p']):.4f} × {f['n_tests']} = {float(sm['perm_p_holm']):.3f}.",
          f"- **FDR (Benjamini–Hochberg):** the correction we use for the {f['n_features']} feature correlations.",
          f"- **Expected false alarms:** 0.05 × {f['n_tests']} = {exp_false:.2f} tests would pass by luck even if nothing were there.", "",
          "### The lexical confound", "",
          "- **What it is:** the word explains both the sound and the feeling, so the model may learn the word.",
          f"- **Evidence 1:** the audio identifies Kapoy with balanced accuracy {probe['balanced_accuracy_mean_10_repeats']:.2f} "
          f"(p = {probe['permutation_p']:.3f})."
          + ("" if f["probe_sig"] else " That is not significant: in this data the sound alone does not reliably give "
             "the word away."),
          f"- **Evidence 2:** Kapoy speakers felt not pleasant {kapoy['NotPleasant']} times out of {sum(kapoy.values())}; "
          f"others {other['NotPleasant']} out of {sum(other.values())} (Fisher p = {f['assoc_v']['p_value']:.3f}).",
          (f"- **Evidence 3:** on two-level arousal the models are right {f['kapoy_acc'].get('Kapoy', 0):.0%} of the time on "
           f"Kapoy recordings and {f['kapoy_acc'].get('other words', 0):.0%} on other words." if f["kapoy_acc"] else ""), "",
          "### Sensitivity analysis", "",
          "- **What it is:** repeating the analysis without a doubtful part of the data, to see whether the result holds.",
          f"- **In our project, {TASK.get(lead['task'], lead['task'])}, {MODEL[lead['model']]}** (all data: {lead['ba']:.2f}):"]
    for scen, (ba, nn) in lead_sens.items():
        L.append(f"  - {SCENARIO.get(scen, scen)}: {ba:.2f} (n = {nn})")
    if val_sens and v2:
        lo, hi = min(v[0] for v in val_sens.values()), max(v[0] for v in val_sens.values())
        L.append(f"- **Valence, {MODEL[v2['model']]}** (all data: {v2['ba']:.2f}): between {lo:.2f} and {hi:.2f} across the same scenarios.")
    L += ["- **What it tells us:** the scores move up and down depending on which files are included. That is a sign the "
          "sample is too small for a stable answer.", "",
          "### Statistical power", "",
          "- **What it is:** the chance that a study detects an effect that really exists.",
          f"- **In our project:** with {n} people, only correlations of about {f['rho_min']:.2f} or more are detectable "
          f"80% of the time. A moderate 0.3 needs about {f['n_for_03']} people.", "",
          "### Why raw audio, not denoised", "",
          "- The protocol forbids denoising.",
          "- Denoising changes loudness, the spectrum and the MFCCs, which are the things we measure.",
          "- It changes each file by a different amount, so a model could learn *how much a file was cleaned* instead of "
          "the feeling.", "",
          "### Reproducibility", "",
          "- One command runs every step, with fixed random seeds.",
          "- Automated tests check the rules that decide the data.",
          f"- A validation step re-checks the whole project: {f['valid']['PASS']} checks pass, {f['valid']['BLOCKED']} blocked"
          + (f", {f['valid']['FAIL']} failed" if f["valid"]["FAIL"] else "") + ".",
          f"- {f['complete']} of {f['n_deliverables']} required outputs are complete. Not complete: "
          + (codes([b.split('. ', 1)[-1] for b in f["blocked_items"]]) if f["blocked_items"] else "none") + ".", ""]

    qa = [("So your model does not work. What is the point of the project?",
           "The point was to find out whether it works, and to find out properly. We built a pipeline that answers the "
           f"question honestly, and the answer is: not with {n} noisy one-word recordings. We also found out why a naive "
           "analysis would be misleading, namely the word confound, and how much data a real study needs."),
          ("Your best p-value is below 0.05. Why do you say nothing is significant?",
           f"Because we ran {f['n_tests']} tests. If you run {f['n_tests']} tests on pure noise, about {exp_false:.2f} of them "
           f"pass by luck. After the Holm correction, {float(sm['perm_p']):.3f} becomes {float(sm['perm_p_holm']):.3f}."),
          ("Why did you not use deep learning?",
           f"Deep networks need thousands of examples. We have {n}. With so little data a large model memorises the "
           "training set, so simple models with cross-validation are the right tools."),
          ("Why did you not use SMOTE or oversampling?",
           "With this few people, artificial samples would be copies of the same few voices, and they leak information "
           "if created before the split. We used class weights instead, which need no artificial data."),
          ("Why are some spoken words UNVERIFIED?",
           f"{f['words_missing']} words are blank in the data sheet. The protocol forbids guessing the word from the "
           "feeling text, so we left them blank. The word is not a model input, so the models are not affected. Only "
           "the word analysis leaves those participants out."),
          ("Some of your files were edited. Is your data valid?",
           (f"We do not hide that. {len(f['edited'])} files contain inserted digital silence and {len(f['no_orig'])} have no "
            f"verified original. We flagged each one and re-ran the models without them"
            + (f": our best score moves from {top['ba']:.2f} to {top_no_src[0]:.2f} without the unverified files"
               if top_no_src else "")
            + ". Our conclusion does not depend on those files. The clean fix is to find the original recordings.")
           if f["supplied"] else "The pipeline never edits a recording; it only converts the format."),
          ("How do you know there is no data leakage?",
           "Three checks. Only acoustic columns may enter the model, and the code stops otherwise. Scaling and filling "
           "of missing values happen inside each training fold. And the code verifies that no participant is in both the "
           "training and the test fold."),
          ("How did you choose which part of the audio is the word?",
           "Automatically, from loudness alone: the dominant short event in the clip. The rule never sees the labels. "
           f"{f['seg_critical']} recordings are flagged as uncertain, and a blind listening review by two group members "
           "is the next step."),
          ("Is consent documented?",
           "The sheet records Yes for every participant, but the log of the signed forms is not complete yet. That is "
           "why the report marks the consent requirement as blocked, and marks its own dissemination as blocked until "
           "the log is complete. Completing it is the first of our next steps."),
          ("Did you account for different voices, for example male and female?",
           "We could not. The sheet has no sex or gender column, and each person gave one recording, so there is no "
           "personal baseline to subtract. Pitch in Hz therefore mostly reflects the natural voice. That is the "
           f"*one word per person* limit on slide {num['limits']}, and it is one reason we do not over-read the pitch features.")]
    if at and at["ratings"]:
        moves = "; ".join(f"{TASK.get(t['task'], t['task']).split(' (')[0]} went from {at['before'][t['task']]['ba']:.2f} to "
                          f"{t['ba']:.2f}" for t in f["tasks"] if t["task"] in at["before"])
        none_before = all(r["holm"] >= 0.05 for r in at["before"].values())
        qa.append((f"The ratings of {at['ratings']} participants were changed after the first analysis. Did you change the "
                   "data to get a better result?",
                   f"No. On {at['day']} the group corrected the sheet {at['ratings_source']}, for {at['codes']}. We kept "
                   "the earlier sheet and the earlier results, so anyone can compare the two. "
                   + (f"With the corrected sheet, the best scores changed like this: {moves}. " if moves else "")
                   + ("No model passes the correction in either analysis. " if none_before and not f["any_holm"] else "")
                   + ("Those participants are all among the files we could not check, so the scores without those files "
                      "are exactly the same before and after the correction"
                      + (f" ({top_no_src[0]:.2f} for our best result)" if top_no_src else "") + ". "
                      if at["all_unverified"] else "")
                   + "Bring the answer forms: they are the evidence for the corrected ratings."))
    if pt:
        qa.append(("The sheet now has all the words, and some ratings are different. Why do your slides not match it?",
                   f"The group completed the sheet on {pt['day']}, after this analysis. {pt['words']} blank words were "
                   f"filled in {pt['words_source']}"
                   + (f", {pt['word_changes']} word was corrected" if pt["word_changes"] == 1 else
                      f", {pt['word_changes']} words were corrected" if pt["word_changes"] else "")
                   + f", and the ratings of {pt['codes']} were corrected {pt['ratings_source']}. A rating is what the "
                   "models learn from, so the whole analysis must be run again, and that could not be done and "
                   "checked before today. We did not mix the two: the results use the earlier sheet, the new sheet "
                   "is stored beside it, and every changed cell is listed in `reports/sheet_update_pending.md`. "
                   "Bring the answer forms of those participants: they are the evidence for the corrected ratings."))
    if f["cfg"] and f["cfg_raw"] and not f["cfg_feats"]:
        without = lead_sens.get("no_config_B")
        qa.append(("The figure of the two recording settings shows a clear difference. Why do you say it is not "
                   "significant?",
                   f"Because of the correction. We compared {f['n_features']} features. {f['cfg_raw']} differ before "
                   f"the correction, and the closest corrected p-value is {f['cfg_min_q']:.3f}, just above 0.05. With only "
                   f"{len(f['cfg'])} files the test is weak, so we do not claim a difference and we do not rule one out."
                   + (f" That is why we also ran the models without those files: the main arousal score moves from "
                      f"{lead['ba']:.2f} to {without[0]:.2f}." if without else "")))
    if f["supplied"]:
        cv = f["conversion"]
        qa.append(("Who prepared the version-2 audio files, and how?",
                   f"A group member converted the recordings using {PREPARED_WITH}: each clip was cut to the part around "
                   "the word and saved as a 16 kHz WAV file. We did not take that conversion on trust. "
                   + (f"For the {cv['n']} files where we have the original, our audit shows that the copy matches the "
                      f"original (similarity {cv['ncc_min']:.2f} or higher out of 1), that it is a cut of "
                      f"{cv['dur'][0]:.1f} to {cv['dur'][1]:.1f} seconds from it"
                      + (", and that the loudness was not changed. " if cv["level_unchanged"] else ". ") if cv else "")
                   + f"For the other {len(f['no_orig'])} we have nothing to compare with, so they are flagged and the models "
                   "were re-run without them. Before you present, open that conversation and note the steps or code it "
                   "used, so you can describe them in your own words. The project has no record of them."))
        qa.append(("Your earlier submission had different numbers. Why?",
                   f"The earlier analysis used the original phone recordings, which covered {EARLIER['n']} participants. "
                   f"This one uses the version-2 audio for all {n}. Different audio and more people give different "
                   "numbers. The conclusion is the same in both: no model beats chance after correction, and the word "
                   "confound is present. One detail did change: "
                   + (f"{EARLIER['fdr_arousal']} features passed the correction for arousal before, and none do now."
                      if not f["fdr"]["arousal"] else
                      f"{EARLIER['fdr_arousal']} features passed the correction for arousal before, and "
                      f"{f['fdr']['arousal']} do now.")))
    qa += [
          ("What would you do with more time?",
           f"Collect about {f['n_for_03']} participants with one device and one setting, have everyone say the same word "
           "or enough repetitions of each word, verify the word boundaries by listening, and follow the analysis plan "
           "we wrote before seeing new data."),
          ("How was the code written? Do you understand it?",
           "Answer this truthfully. The pipeline was implemented with an AI coding assistant under the group's "
           "direction; the commit history shows it. Say which decisions were yours (the data, the audio set, what to "
           "report), and explain any step using Part 2. Check your course's rule on AI tools before you present.")]
    L += ["## Part 3 · Likely questions and honest answers", ""]
    for q, a in qa:
        L += [f"**“{q}”**", "", a, ""]

    L += ["## Part 4 · Numbers to remember", "",
          "| What | Number |", "|---|---|",
          f"| Participants analysed | {n} (protocol minimum 40) |",
          f"| Features per recording | {f['n_features']} |",
          f"| Models | 3, plus 2 dummy baselines |",
          f"| Cross-validation | 5 folds × {f['n_repeats']} repeats, nested |",
          f"| Permutations per test | {f['n_perm']} |"]
    for t in f["tasks"]:
        L.append(f"| Best score, {TASK.get(t['task'], t['task'])} | {t['ba']:.2f} (chance {t['chance']:.2f}), "
                 f"interval {t['lo']:.2f}–{t['hi']:.2f} |")
    L += [f"| Smallest p-value | {float(sm['perm_p']):.3f} → {float(sm['perm_p_holm']):.3f} after Holm ({f['n_tests']} tests) |",
          f"| Features passing correction | {f['fdr']['arousal']} for arousal, {f['fdr']['valence']} for valence |",
          f"| Kapoy recognised from audio | {probe['balanced_accuracy_mean_10_repeats']:.2f} balanced accuracy, p = {probe['permutation_p']:.3f} |",
          f"| Kapoy and feeling | {kapoy['NotPleasant']} of {sum(kapoy.values())} not pleasant; others {other['NotPleasant']} of {sum(other.values())} |"]
    if f["supplied"]:
        L += [f"| Files without a verified original | {len(f['no_orig'])} ({codes(f['no_orig'])}) |",
              f"| Edited or clipped files | {len(f['edited'])} ({codes(f['edited'])}) |"]
    L += [f"| Words missing in the sheet | {f['words_missing']} |",
          *([f"| Completed sheet, not analysed yet | received {pt['day']}: {pt['words']} words filled in, ratings of "
             f"{pt['ratings']} participants corrected |"] if pt else []),
          f"| Smallest detectable correlation | about {f['rho_min']:.2f}; 0.3 needs about {f['n_for_03']} people |",
          f"| Validation | {f['valid']['PASS']} pass, {f['valid']['BLOCKED']} blocked (consent) |", "",
          "## Part 5 · What not to say", "",
          "| Don't say | Say instead |", "|---|---|",
          "| “Our model detects emotions.” | “Our models did not beat chance after correction.” |",
          f"| “We got {top['ba']:.0%} accuracy.” | “Balanced accuracy {top['ba']:.2f} where chance is {top['chance']:.2f}, "
          "with a wide interval.” |",
          ("| “The result is significant.” | “No result passes the correction for " + str(f["n_tests"]) + " tests.” |"
           if not f["any_holm"] else "| “Everything is significant.” | “Only the results that pass the correction count.” |"),
          "| “Kapoy means the student is sad.” | “Kapoy speakers more often reported not-pleasant feelings.” |",
          "| “The class caused the feeling.” | “These are associations. The design cannot show causes.” |",
          "| “The data is clean.” | “We flagged every doubtful file and tested the results without them.” |",
          "| “The project failed.” | “The answer to our research question is no, with this data.” |", ""]
    return "\n".join(line for line in L if line is not None)


# ---- a small Markdown -> HTML renderer for exactly the constructs used above -----------------------------
def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", text)


def to_html(md: str, title: str = "Presentation script") -> str:
    out, lines, i = [], md.split("\n"), 0
    toc = []
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
            if level == 2:
                toc.append((slug, text))
            out.append(f'<h{level} id="{slug}">{inline(text)}</h{level}>')
            i += 1
        elif line.startswith(">"):
            paras, cur = [], []
            while i < len(lines) and lines[i].startswith(">"):
                body = lines[i][1:].strip()
                if body:
                    cur.append(body)
                elif cur:
                    paras.append(" ".join(cur))
                    cur = []
                i += 1
            if cur:
                paras.append(" ".join(cur))
            out.append("<blockquote>" + "".join(f"<p>{inline(p)}</p>" for p in paras) + "</blockquote>")
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], rows[2:]
            out.append('<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{inline(c)}</th>" for c in head)
                       + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                                                          for r in body) + "</tbody></table></div>")
        elif line.lstrip().startswith("- "):
            out.append("<ul>")
            open_sub = False
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                nested = lines[i].startswith("  ")
                item = inline(lines[i].lstrip()[2:])
                if nested and not open_sub:
                    out[-1] = out[-1][:-5] + "<ul>"  # reopen the previous <li> to hold the sub-list
                    open_sub = True
                elif not nested and open_sub:
                    out.append("</ul></li>")
                    open_sub = False
                out.append(f"<li>{item}</li>")
                i += 1
            out.append("</ul></li></ul>" if open_sub else "</ul>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", ">", "|")) \
                    and not lines[i].lstrip().startswith("- "):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{inline(' '.join(para))}</p>")
    nav = "".join(f'<a href="#{slug}">{html.escape(text)}</a>' for slug, text in toc)
    return PAGE.replace("__TITLE__", html.escape(title)).replace("__NAV__", nav).replace("__BODY__", "\n".join(out))


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg: #f7f7f4; --surface: #ffffff; --ink: #1c1c1a; --ink-2: #4f4f4a; --line: #e2e2dc; --accent: #2563c9;
  --say: #f1f5fc; --code: #efefea; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #141412; --surface: #1e1e1b;
  --ink: #ecece6; --ink-2: #bdbdb4; --line: #33332e; --accent: #7ea6ff; --say: #1b2433; --code: #2a2a26; } }
:root[data-theme="dark"] { --bg: #141412; --surface: #1e1e1b; --ink: #ecece6; --ink-2: #bdbdb4; --line: #33332e;
  --accent: #7ea6ff; --say: #1b2433; --code: #2a2a26; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }
nav { position: sticky; top: 0; background: var(--surface); border-bottom: 1px solid var(--line); padding: 8px 16px;
  display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 14px; z-index: 2; }
nav a { color: var(--ink-2); text-decoration: none; white-space: nowrap; }
nav a:hover { color: var(--accent); }
main { max-width: 820px; margin: 0 auto; padding: 24px 16px 72px; }
h1 { font-size: 1.7rem; line-height: 1.25; margin: 8px 0 8px; }
h2 { font-size: 1.3rem; margin: 44px 0 12px; padding-top: 14px; border-top: 2px solid var(--line); }
h3 { font-size: 1.08rem; margin: 30px 0 8px; }
p, li { max-width: 70ch; }
em { color: var(--ink-2); }
li { margin: 4px 0; }
blockquote { margin: 10px 0 14px; padding: 10px 16px; background: var(--say); border-left: 4px solid var(--accent);
  border-radius: 0 8px 8px 0; font-size: 1.05rem; }
blockquote p { margin: 8px 0; }
code { background: var(--code); padding: 1px 5px; border-radius: 4px; font-size: .9em; overflow-wrap: anywhere; }
.table-wrap { overflow-x: auto; margin: 12px 0; border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 15px; background: var(--surface); }
th, td { padding: 7px 12px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); }
th { background: var(--code); }
tr:last-child td { border-bottom: none; }
@media print { nav { display: none; } body { background: #fff; color: #000; font-size: 11.5pt; }
  h2 { break-before: page; } h3, blockquote, tr { break-inside: avoid; } blockquote { background: #f3f3f3; } }
</style>
</head>
<body>
<nav>__NAV__</nav>
<main>
__BODY__
</main>
</body>
</html>
"""


def main() -> int:
    md = build(facts())
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md + "\n", encoding="utf-8")
    OUT_HTML.write_text(to_html(md), encoding="utf-8")
    print(f"presentation script -> {OUT_MD.relative_to(ROOT).as_posix()} and {OUT_HTML.name} "
          f"({len(md.split())} words)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
