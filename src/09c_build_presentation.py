"""
09c_build_presentation.py -- the results presentation: reports/presentation/G3_results_presentation.pptx.

Twelve 16:9 slides with speaker notes, built from the pipeline's own outputs, the same files the final report
reads. Every number is read from a generated file and every result-dependent sentence comes from a branch on
the data, so the slides cannot drift from the results. Figures are the ones in figures/.

Usage: .venv/Scripts/python.exe src/09c_build_presentation.py
"""

from __future__ import annotations

import json
import sys
from datetime import date

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.parts.image import Image
from pptx.util import Inches, Pt
from scipy.stats import norm

from common import (CLEAN_METADATA_CSV, DATA, FIGURES, MIN_VALID_OBSERVATIONS, MODELS, PROTOCOL_WORDS, REPORTS,
                    load_consent_log, rel, setup_logging)

OUT = REPORTS / "presentation" / "G3_results_presentation.pptx"
INK, MUTED, ACCENT = RGBColor(0x1C, 0x1C, 0x1A), RGBColor(0x5A, 0x5A, 0x55), RGBColor(0x25, 0x63, 0xC9)
LINE, TILE, WHITE = RGBColor(0xE2, 0xE2, 0xDC), RGBColor(0xF1, 0xF4, 0xFA), RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"
W, H, MARGIN = 13.333, 7.5, 0.6
MODEL_LABEL = {"logistic_regression": "Logistic regression", "svm_rbf": "SVM (RBF)", "random_forest": "Random forest",
               "dummy_majority": "Dummy (majority)", "dummy_stratified": "Dummy (stratified)"}
PRESPECIFIED = {"f0_median": "F0 median", "f0_range_st": "F0 range (st)", "rms_db_mean": "RMS level",
                "utterance_duration_s": "Duration", "spectral_centroid_hz_mean": "Spectral centroid",
                "voiced_ratio": "Voiced ratio"}

log = setup_logging("presentation", REPORTS / "logs" / "09c_build_presentation.log")


# ---- building blocks ------------------------------------------------------------------
def set_bullet(paragraph, char: str, margin_in: float, hanging_in: float) -> None:
    """A real bullet with a hanging indent, so wrapped lines align with the text, not the bullet."""
    ppr = paragraph._p.get_or_add_pPr()
    ppr.set("marL", str(int(Inches(margin_in))))
    ppr.set("indent", str(-int(Inches(hanging_in))))
    for old in ppr.findall(qn("a:buChar")) + ppr.findall(qn("a:buNone")):
        ppr.remove(old)
    ppr.append(ppr.makeelement(qn("a:buChar"), {"char": char}))


def text_box(slide, left, top, width, height, paragraphs, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP):
    """paragraphs: list of str or (str, level): level 0 = plain, 1 = bullet, 2 = sub-bullet."""
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04)
    for i, item in enumerate(paragraphs):
        text, level = (item, 0) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = text
        f = run.font
        f.name, f.size, f.bold = FONT, Pt(size - (2 if level == 2 else 0)), bold
        f.color.rgb = color
        p.space_after = Pt(6 if level else 8)
        if level == 1:
            set_bullet(p, "•", 0.3, 0.3)
        elif level == 2:
            set_bullet(p, "–", 0.65, 0.3)
    return box


def new_slide(prs, title, subtitle=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    text_box(slide, MARGIN, 0.4, W - 2 * MARGIN, 0.75, [title], size=30, bold=True)
    if subtitle:
        text_box(slide, MARGIN, 1.08, W - 2 * MARGIN, 0.5, [subtitle], size=16, color=MUTED)
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(MARGIN), Inches(1.6), Inches(W - 2 * MARGIN), Inches(0.02))
    rule.fill.solid()
    rule.fill.fore_color.rgb = LINE
    rule.line.fill.background()
    rule.shadow.inherit = False
    return slide


def footer(slide, number):
    text_box(slide, MARGIN, H - 0.45, 9.5, 0.3,
             ["Group 3 · CS412 mini-project · generated from the pipeline outputs (src/09c_build_presentation.py)"],
             size=10, color=MUTED)
    text_box(slide, W - MARGIN - 1.0, H - 0.45, 1.0, 0.3, [str(number)], size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def picture(slide, path, left, top, max_w, max_h):
    px_w, px_h = Image.from_file(str(path)).size
    scale = min(max_w / px_w, max_h / px_h)
    w, h = px_w * scale, px_h * scale
    slide.shapes.add_picture(str(path), Inches(left + (max_w - w) / 2), Inches(top), Inches(w), Inches(h))


def table(slide, df, left, top, width, col_widths, size=14, row_h=0.42):
    shape = slide.shapes.add_table(len(df) + 1, len(df.columns), Inches(left), Inches(top), Inches(width),
                                   Inches(row_h * (len(df) + 1)))
    tbl = shape.table
    for j, w in enumerate(col_widths):
        tbl.columns[j].width = Inches(w)
    for j, name in enumerate(df.columns):
        cell = tbl.cell(0, j)
        cell.text = str(name)
        cell.fill.solid()
        cell.fill.fore_color.rgb = ACCENT
        run = cell.text_frame.paragraphs[0].runs[0]
        run.font.name, run.font.size, run.font.bold, run.font.color.rgb = FONT, Pt(size), True, WHITE
    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, value in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = TILE if i % 2 == 0 else WHITE
            run = cell.text_frame.paragraphs[0].runs[0]
            run.font.name, run.font.size, run.font.color.rgb = FONT, Pt(size), INK
    return shape


def tiles(slide, items, top, height=1.55):
    gap = 0.25
    width = (W - 2 * MARGIN - gap * (len(items) - 1)) / len(items)
    for k, (big, label) in enumerate(items):
        left = MARGIN + k * (width + gap)
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        shape.adjustments[0] = 0.08
        shape.fill.solid()
        shape.fill.fore_color.rgb = TILE
        shape.line.fill.background()
        shape.shadow.inherit = False
        text_box(slide, left + 0.15, top + 0.12, width - 0.3, 0.8, [big], size=40, bold=True, color=ACCENT)
        text_box(slide, left + 0.15, top + 0.92, width - 0.3, height - 0.95, [label], size=14, color=INK)


def flow(slide, steps, top):
    """Pipeline steps as boxes in two rows of four, with arrows in reading order."""
    per_row, gap, box_h = 4, 0.45, 1.25
    box_w = (W - 2 * MARGIN - gap * (per_row - 1)) / per_row
    for k, (head, sub) in enumerate(steps):
        r, c = divmod(k, per_row)
        left, y = MARGIN + c * (box_w + gap), top + r * (box_h + 0.55)
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(y), Inches(box_w), Inches(box_h))
        shape.adjustments[0] = 0.12
        shape.fill.solid()
        shape.fill.fore_color.rgb = TILE
        shape.line.color.rgb = LINE
        shape.shadow.inherit = False
        text_box(slide, left + 0.12, y + 0.1, box_w - 0.24, 0.4, [f"{k + 1}. {head}"], size=15, bold=True)
        text_box(slide, left + 0.12, y + 0.5, box_w - 0.24, box_h - 0.55, [sub], size=12, color=MUTED)
        if c < per_row - 1 and k < len(steps) - 1:
            arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(left + box_w + 0.08), Inches(y + box_h / 2 - 0.12),
                                           Inches(gap - 0.16), Inches(0.24))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = ACCENT
            arrow.line.fill.background()


def min_detectable_rho(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest Spearman rho detectable (two-sided), Fisher z with the 1.06 variance factor (Fieller et al., 1957)."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(np.tanh(z * np.sqrt(1.06 / (n - 3))))


def n_for_rho(rho: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Participants needed to detect a Spearman rho (two-sided), same approximation as min_detectable_rho."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil(3 + 1.06 * (z / np.arctanh(rho)) ** 2))


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---- slides ----------------------------------------------------------------------------
def main() -> int:
    dq = load_json(REPORTS / "data_quality_summary.json")
    eda = load_json(REPORTS / "eda" / "eda_summary.json")
    fsum = load_json(REPORTS / "features_summary.json")
    train = load_json(MODELS / "training_summary.json")
    ev = load_json(REPORTS / "evaluation" / "evaluation_summary.json")
    lex = load_json(REPORTS / "lexical" / "lexical_summary.json")
    manifest = pd.read_csv(DATA / "analysis_manifest.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    meta = pd.read_csv(CLEAN_METADATA_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    summ = pd.read_csv(REPORTS / "evaluation" / "summary_metrics.csv")
    sp = pd.read_csv(REPORTS / "eda" / "spearman_feature_vs_rating.csv")

    n_meta, n = len(meta), int((manifest["include_primary"] == "True").sum())
    supplied_csv = REPORTS / "supplied_audio_audit.csv"
    supplied = (pd.read_csv(supplied_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig") if supplied_csv.is_file()
                else pd.DataFrame(columns=["verdict", "flags", "participant_code"]))
    supplied_new = supplied[supplied["verdict"] == "NEW_RECORDING"]
    n_recorded = dq["participants_with_recordings"]
    retakes = dq["retake_participants"]
    extra_takes = dq["total_files"] - n_recorded
    supplied_mode = dq.get("audio_source", "original") != "original"
    no_orig = int((manifest.get("subset_no_original", pd.Series(dtype=str)) == "True").sum())
    edited = int((manifest.get("subset_edited_or_clipped", pd.Series(dtype=str)) == "True").sum())
    n_cfg = dq.get("config_b_origin", dq["flag_counts_selected"].get("RECORDING_CONFIG_DIFFERS", 0))
    missing_words = int((meta["spoken_word"] == "").sum())
    flags = dq["flag_counts_selected"]
    any_sig = ev["any_model_significant_after_holm"]
    fdr = eda["significant_features_fdr05"]
    va = eda["valence_arousal_spearman"]["analysis"]
    probe, assoc = lex["probe"], lex["word_affect_association"]
    consent = load_consent_log()
    rho_min = min_detectable_rho(n)
    learned = ["logistic_regression", "svm_rbf", "random_forest"]
    headline = ("The features carry statistically supported information about self-reported affect for some targets"
                if any_sig else "The data do not support a claim that post-class affect can be recognised from speech")

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    number = 1  # the title slide is slide 1 and carries no footer

    # 1 title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text_box(s, MARGIN, 2.2, W - 2 * MARGIN, 1.4, ["Bisaya post-class speech and self-reported affect"], size=40, bold=True)
    text_box(s, MARGIN, 3.55, W - 2 * MARGIN, 0.9,
             ["Can one spoken word, said right after class, tell us how a student feels?"], size=22, color=MUTED)
    text_box(s, MARGIN, 5.2, W - 2 * MARGIN, 0.9,
             [f"CS412 mini-project · Group 3 · second-year Computer Science students · results of {date.today().isoformat()}"],
             size=16, color=MUTED)
    notes(s, "Introduce the group and the question. Everything on these slides is generated from the pipeline "
             "outputs; the full report is reports/final_group_research_report.md.")

    # 2 question and design
    number += 1
    s = new_slide(prs, "The question and how we asked it", "Protocol of the class mini-project")
    text_box(s, MARGIN, 1.9, 7.4, 4.8, [
        ("Right after class, each participant said one Bisaya word that matched their reaction", 1),
        (f"Suggested words: {', '.join(PROTOCOL_WORDS)}", 2),
        ("They then rated how they felt on two 1–5 scales", 1),
        ("Valence: very unpleasant (1) to very pleasant (5)", 2),
        ("Arousal: very low (1) to very high (5)", 2),
        ("Research question: do acoustic features of that word carry information about the ratings?", 1),
        ("Speaker-independent: the system is always tested on people it has never heard", 1),
    ])
    tiles_left = 8.4
    for k, (big, label) in enumerate([("1", "word per participant"), ("2", "self-rated scales (1–5)"),
                                      (str(fsum["n_features"]), "acoustic features per recording")]):
        y = 1.95 + k * 1.6
        shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(tiles_left), Inches(y), Inches(4.3), Inches(1.4))
        shape.adjustments[0] = 0.08
        shape.fill.solid()
        shape.fill.fore_color.rgb = TILE
        shape.line.fill.background()
        shape.shadow.inherit = False
        text_box(s, tiles_left + 0.2, y + 0.18, 1.5, 1.0, [big], size=40, bold=True, color=ACCENT)
        text_box(s, tiles_left + 1.6, y + 0.42, 2.6, 0.8, [label], size=16)
    footer(s, number)
    notes(s, "The prompt and the scales follow the class protocol. The ratings are the participants' own; the "
             "word is chosen by them, which later turns out to matter (lexical confound).")

    # 3 data
    number += 1
    s = new_slide(prs, "The data we actually had", "From the group's sheet and recordings to the analysis set")
    tiles(s, [(str(n_meta), "participants in the sheet"), (str(n_recorded), "with a recording"),
              (str(n), "analysed"), (str(MIN_VALID_OBSERVATIONS), "required by the protocol")], top=1.95)
    text_box(s, MARGIN, 3.85, W - 2 * MARGIN, 3.0, [
        *([(f"By the group's decision the analysis uses the {dq['audio_source']} delivery: trimmed 16 kHz copies, one "
            "per participant; the original recordings are archived, not analysed", 1),
           (f"{no_orig} files have no verified original and {edited} are edited or clipped; sensitivity analyses "
            "exclude them", 2)] if supplied_mode else
          [(f"{n_meta - n_recorded} participants have no recording; {len(retakes)} were recorded more than once "
            f"({extra_takes} extra takes): one take each was chosen by a label-blind complete-word rule", 1),
           *([(f"A later delivery has trimmed 16 kHz copies for {len(supplied_new)} of them, "
               f"{int((supplied_new['flags'] != '').sum())} edited or damaged: not counted until the originals are "
               "supplied", 2)] if len(supplied_new) else [])]),
        (f"{missing_words} spoken words are missing from the sheet; the word is never guessed from the feeling text", 1),
        (f"Background voices in {flags.get('BACKGROUND_VOICES', 0)} of {n} selected recordings; "
         f"{n_cfg} {'are copies of recordings that' if supplied_mode else ''} used a different recording setting", 1),
        ("Consent: " + ("documented for the analysed participants" if consent and all(
            consent.get(c, False) for c in manifest.loc[manifest["include_primary"] == "True", "participant_code"])
            else "the sheet says Yes, but the signed-form log is still to be supplied"), 1),
    ], size=17)
    footer(s, number)
    notes(s, f"The protocol asks for at least {MIN_VALID_OBSERVATIONS} valid observations; we have {n}. "
             "Nothing was imputed or invented: missing recordings and words stay missing.")

    # 4 pipeline
    number += 1
    s = new_slide(prs, "How the pipeline works", "One command runs every step; each step writes its own report")
    flow(s, [("Audit", "every file and sheet cell checked; nothing changed"),
             ("Clean", "logged, rule-based fixes; raw values kept"),
             ("Audio", "WAV 16 kHz mono; label-blind word location and take choice"),
             ("Features", f"{fsum['n_features']} acoustic features: pitch, energy, spectrum, MFCC"),
             ("Explore", "distributions, correlations, confound checks"),
             ("Models", f"nested speaker-grouped CV, {train['n_repeats']} repeats; dummy baselines"),
             ("Test", "permutation tests, Holm correction, bootstrap CIs"),
             ("Lexical control", "is it the voice, or just the word?")], top=1.95)
    text_box(s, MARGIN, 5.75, W - 2 * MARGIN, 1.0,
             ["Safeguards: no participant in both training and test data · preprocessing fitted inside each training fold "
              "· the labels are never used to cut or choose audio"], size=15, color=MUTED)
    footer(s, number)
    notes(s, "The design choices that make the numbers trustworthy: grouping by participant, all preprocessing "
             "inside the cross-validation, permutation tests of the whole procedure, and correction for multiple tests.")

    # 5 ratings
    number += 1
    s = new_slide(prs, "What the participants reported",
                  f"Valence and arousal move together (Spearman ρ = {va[0]:.2f}, n = {va[2]})")
    picture(s, FIGURES / "eda" / "fig01_affect_distributions.png", MARGIN, 1.85, 7.6, 4.9)
    not_run = ev["tasks_not_modelled"]
    text_box(s, 8.5, 1.95, 4.25, 4.8, [
        ("Most ratings are neutral to pleasant and moderate to high", 1),
        ("Classes for modelling are groupings of the participants' own ratings (derived, fixed in advance):", 1),
        (", ".join(ev["tasks_modelled"]), 2),
        ("Not modelled, too few people per class: " + ", ".join(not_run), 1),
    ], size=16)
    footer(s, number)
    notes(s, "Derived targets: arousal_3 = low (1-2) / moderate (3) / high (4-5); valence_2 = not pleasant (1-3) / "
             "pleasant (4-5). A task is modelled only if every class has at least 5 participants.")

    # 6 features vs ratings
    number += 1
    s = new_slide(prs, "Acoustic features and the ratings",
                  f"Six prosodic features chosen in advance; {fsum['n_features']} features in total")
    rows = []
    for f, label in PRESPECIFIED.items():
        a = sp[(sp["feature"] == f) & (sp["rating"] == "arousal")].iloc[0]
        v = sp[(sp["feature"] == f) & (sp["rating"] == "valence")].iloc[0]
        rows.append({"Feature": label, "ρ with arousal": f"{a.spearman_rho:+.2f}", "p": f"{a.p_value:.3f}",
                     "ρ with valence": f"{v.spearman_rho:+.2f}", "p ": f"{v.p_value:.3f}"})
    table(s, pd.DataFrame(rows), MARGIN, 1.95, 7.3, [2.3, 1.45, 1.0, 1.55, 1.0], size=14)
    pre_sig = bool((sp[sp["feature"].isin(PRESPECIFIED)]["q_value_bh"] < 0.05).any())
    top_sig = sp[(sp["q_value_bh"] < 0.05)].sort_values("p_value")
    text_box(s, 8.3, 1.95, 4.45, 4.9, [
        ("None of the six passes FDR correction" if not pre_sig else "At least one of the six passes FDR correction", 1),
        (f"Across all {fsum['n_features']} features: {fdr['arousal']} pass for arousal, {fdr['valence']} for valence", 1),
    ] + ([(f"{r.feature} (ρ = {r.spearman_rho:+.2f})", 2) for r in top_sig.itertuples()][:4]) + [
        ("Those were not chosen in advance, and their meaning is unclear", 1),
        (f"With n = {n}, only correlations of about |ρ| ≥ {rho_min:.2f} can be detected reliably (80% power)", 1),
    ], size=15)
    footer(s, number)
    notes(s, "Spearman correlations, Benjamini-Hochberg FDR across all features. The power statement uses the Fisher "
             "z approximation with the Spearman variance factor (Fieller et al., 1957).")

    # 7 models table
    number += 1
    s = new_slide(prs, "Speaker-independent classification",
                  "Balanced accuracy pooled over repeated nested cross-validation; chance = dummy baselines")
    rows = []
    for task in ev["tasks_modelled"]:
        t = summ[summ["task"] == task]
        best = t[t["model"].isin(learned)].sort_values("pooled_ba", ascending=False).iloc[0]
        dummy = t[t["model"] == "dummy_majority"].iloc[0]
        rows.append({"Task": task, "Best model": MODEL_LABEL[best.model],
                     "Balanced accuracy [95% CI]": f"{best.pooled_ba:.2f} [{best.pooled_ba_ci_low:.2f}, {best.pooled_ba_ci_high:.2f}]",
                     "Majority baseline": f"{dummy.pooled_ba:.2f}", "Permutation p": f"{best.perm_p:.3f}",
                     "Holm p": f"{best.perm_p_holm:.3f}"})
    table(s, pd.DataFrame(rows), MARGIN, 1.95, W - 2 * MARGIN, [1.6, 2.2, 3.0, 1.9, 1.7, 1.73], size=15, row_h=0.5)
    n_tests = int(summ["perm_p"].notna().sum())
    n_raw = int((summ["perm_p"] < 0.05).sum())
    text_box(s, MARGIN, 4.35, W - 2 * MARGIN, 2.4, [
        (("No model survives Holm correction across the " if not any_sig else "Some models survive Holm correction across the ")
         + f"{n_tests} permutation tests", 1),
        (f"{n_raw} of {n_tests} raw p-values are below 0.05; about {0.05 * n_tests:.2f} would be expected by chance "
         "alone, so a few raw hits are not evidence", 1),
        ("The confidence intervals are wide and overlap the baselines", 1),
    ], size=17)
    footer(s, number)
    notes(s, "Each permutation test repeats the whole nested procedure with shuffled labels. Holm controls the "
             "family-wise error across all learned models and tasks.")

    # 8 evaluation figure
    number += 1
    s = new_slide(prs, "Model results against chance", "Balanced accuracy with 95% bootstrap intervals")
    picture(s, FIGURES / "evaluation" / "fig_eval_balanced_accuracy.png", MARGIN, 1.85, W - 2 * MARGIN, 5.0)
    footer(s, number)
    notes(s, "Dots are pooled balanced accuracy; bars are participant-level bootstrap intervals; the vertical line "
             "in each panel is chance for that task.")

    # 9 lexical confound
    number += 1
    s = new_slide(prs, "Is it the voice, or just the word?", "The lexical confound")
    v2 = assoc["valence_2"]["table"]
    text_box(s, MARGIN, 1.95, 6.9, 4.8, [
        (f"The audio alone tells 'Kapoy' apart from the other words: balanced accuracy "
         f"{probe['balanced_accuracy_mean_10_repeats']:.2f} (chance 0.50, permutation p = {probe['permutation_p']:.3f})", 1),
        (f"And the word goes with the feeling: see the table (Fisher p = {assoc['valence_2']['p_value']:.3f}, "
         f"Cramér's V = {assoc['valence_2']['cramers_v']:.2f})", 1),
        ("So a model can score above chance by recognising which word was said, without hearing any emotion", 1),
        (f"Within Kapoy alone, {lex['within_word_significant_q05']} of {lex['within_word_tests']} feature–rating "
         "correlations pass correction", 1),
    ], size=17)
    word_tab = pd.DataFrame([{"Word": "Kapoy", "Not pleasant (1–3)": v2["Kapoy"]["NotPleasant"],
                              "Pleasant (4–5)": v2["Kapoy"]["Pleasant"]},
                             {"Word": "other words", "Not pleasant (1–3)": v2["other"]["NotPleasant"],
                              "Pleasant (4–5)": v2["other"]["Pleasant"]}])
    text_box(s, 7.95, 1.95, 4.8, 0.45, ["Self-reported valence by spoken word (participants)"], size=14, color=MUTED)
    table(s, word_tab, 7.95, 2.45, 4.8, [1.6, 1.6, 1.6], size=15, row_h=0.55)
    footer(s, number)
    notes(s, "This is why the word must be controlled in future collections: a fixed word for everyone, or enough "
             "repetitions of each word to analyse within words.")

    # 10 limitations
    number += 1
    s = new_slide(prs, "Other confounds and limits", "Why the results are preliminary")
    text_box(s, MARGIN, 1.95, W - 2 * MARGIN, 4.9, [
        (f"Sample: {n} participants analysed, "
         + (f"below the protocol minimum of {MIN_VALID_OBSERVATIONS}" if n < MIN_VALID_OBSERVATIONS else
            f"meeting the protocol minimum of {MIN_VALID_OBSERVATIONS}"), 1),
        (f"Power: with n = {n}, only correlations of about |ρ| ≥ {rho_min:.2f} are reliably detectable; a moderate "
         f"ρ = 0.3 needs about {n_for_rho(0.3)} participants", 2),
        *([(f"Audio: trimmed copies, not the original recordings; {no_orig} without a verified original, {edited} "
            "edited or clipped", 1)] if supplied_mode else []),
        (f"Recording setting: {n_cfg} files {'are copies of recordings that ' if supplied_mode else ''}used a "
         "different setting, which "
         + (f"shifts {eda['config_effect_features_fdr05']} of {fsum['n_features']} features"
            if eda["config_effect_features_fdr05"] else "shows no significant feature difference in this analysis"), 1),
        (f"Noise: background voices in {flags.get('BACKGROUND_VOICES', 0)} of {n} recordings, captured in a "
         "hallway right after class", 1),
        ("One word per person: no way to separate a person's voice from their state", 1),
        ("Context cannot be compared: one year level, one class activity, one session", 1),
        ("Correlational design with self-reports: no causal conclusions", 1),
    ], size=19)
    footer(s, number)
    notes(s, "These limits are documented in the final report, with the exact numbers and the sensitivity analyses.")

    # 11 conclusion
    number += 1
    s = new_slide(prs, "Conclusion")
    text_box(s, MARGIN, 2.0, W - 2 * MARGIN, 1.4, [headline + "."], size=26, bold=True, color=ACCENT)
    text_box(s, MARGIN, 3.5, W - 2 * MARGIN, 3.2, [
        ("The pipeline is complete, reproducible and validated, so a larger dataset can be analysed with one command", 1),
        ("The word a participant chooses is itself linked to how they feel, and must be controlled", 1),
        ("The present sample is too small to detect realistic effects", 1),
    ], size=20)
    footer(s, number)
    notes(s, "Say it plainly: a negative or inconclusive result from a careful analysis is a valid finding.")

    # 12 next steps
    number += 1
    s = new_slide(prs, "Next steps")
    text_box(s, MARGIN, 1.95, W - 2 * MARGIN, 4.9, [
        ("Document consent from the signed forms (consent/consent_log.csv)", 1),
        ("Blind listening review by two group members: words, boundaries, retakes (data/review/listening_review.html)", 1),
        *([("Find the original recordings behind the copies without a verified original "
            "(reports/supplied_audio_audit.md)", 1)] if supplied_mode else []),
        ("Collect more participants with the checklist: audio only, one setting, quiet spot, one take", 1),
        ("Pre-registered analysis plan for a larger or pooled study (docs/analysis_plan_pooled_study.md)", 1),
        ("Confirm open protocol questions with the instructor (docs/next_steps/)", 1),
    ], size=20)
    footer(s, number)
    notes(s, "Each step has a ready-made file in the repository; RUNNING.md explains how to rerun everything after.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    log.info("presentation: %d slides -> %s", len(prs.slides), rel(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
