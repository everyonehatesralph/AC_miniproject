"""
09c_build_presentation.py -- the results presentation: reports/presentation/G3_results_presentation.pptx.

16:9 slides with speaker notes, built from the pipeline's own outputs, the same files the final report reads.
Every number is read from a generated file and every result-dependent sentence comes from a branch on the
data, so the slides cannot drift from the results. Figures are the ones in figures/.

The slide titles and the speaker notes come from 10b_build_presentation_script.py, which also writes the
presenter script, so the deck and the script always say the same thing. The wording is plain English: each
title states the slide's message, and each technical word is explained where it first appears.

Usage: .venv/Scripts/python.exe src/09c_build_presentation.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

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

from common import (CLEAN_METADATA_CSV, DATA, FIGURES, MIN_CLASS_SIZE, MIN_VALID_OBSERVATIONS, MODELS, PROTOCOL_WORDS, REPORTS,
                    load_consent_log, rel, setup_logging)

OUT = REPORTS / "presentation" / "G3_results_presentation.pptx"
INK, MUTED, ACCENT = RGBColor(0x1C, 0x1C, 0x1A), RGBColor(0x5A, 0x5A, 0x55), RGBColor(0x25, 0x63, 0xC9)
LINE, TILE, WHITE = RGBColor(0xE2, 0xE2, 0xDC), RGBColor(0xF1, 0xF4, 0xFA), RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"
W, H, MARGIN = 13.333, 7.5, 0.6
MODEL_LABEL = {"logistic_regression": "Logistic regression", "svm_rbf": "SVM", "random_forest": "Random forest"}
PRESPECIFIED = {"f0_median": "Pitch level", "f0_range_st": "Pitch range", "rms_db_mean": "Loudness",
                "utterance_duration_s": "Word duration", "spectral_centroid_hz_mean": "Brightness of the sound",
                "voiced_ratio": "Share of voiced sound"}
# the derived targets in plain words: a table label, the grouping with its scores, and a short name
TASK_LABEL = {"arousal_3": "Arousal: low / moderate / high", "valence_2": "Valence: not pleasant / pleasant",
              "arousal_2": "Arousal: not high / high", "valence_3": "Valence: unpleasant / neutral / pleasant"}
TASK_GROUPS = {"arousal_3": "Arousal: low (1–2), moderate (3), high (4–5)",
               "valence_2": "Valence: not pleasant (1–3), pleasant (4–5)",
               "arousal_2": "Arousal: not high (1–3), high (4–5)",
               "valence_3": "Valence: unpleasant (1–2), neutral (3), pleasant (4–5)"}
TASK_SHORT = {"valence_5": "valence with all five scores", "arousal_5": "arousal with all five scores",
              "valence_3": "valence in three groups", "arousal_3": "arousal in three groups",
              "valence_2": "valence in two groups", "arousal_2": "arousal in two groups"}

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
             ["Group 3 · CS412 mini-project · every number comes from the analysis output files"],
             size=10, color=MUTED)
    text_box(slide, W - MARGIN - 1.0, H - 0.45, 1.0, 0.3, [str(number)], size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def picture(slide, path, left, top, max_w, max_h) -> float:
    px_w, px_h = Image.from_file(str(path)).size
    scale = min(max_w / px_w, max_h / px_h)
    w, h = px_w * scale, px_h * scale
    slide.shapes.add_picture(str(path), Inches(left + (max_w - w) / 2), Inches(top), Inches(w), Inches(h))
    return top + h  # the bottom edge, in inches


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


def signed(value: float) -> str:
    """+0.26 / −0.03, with a real minus sign."""
    return f"{value:+.2f}".replace("-", "−")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---- slides ----------------------------------------------------------------------------
def presenter_script():
    """The module that owns the slide titles and the speaker notes (it needs only the standard library)."""
    path = Path(__file__).with_name("10b_build_presentation_script.py")
    spec = importlib.util.spec_from_file_location("presentation_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    n_features = fsum["n_features"]

    ps = presenter_script()
    facts = ps.facts()
    script = ps.slide_script(facts)
    by_key = {entry["key"]: entry for entry in script}
    headline = ps.headline(any_sig)

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    built = []  # slide keys in the order they are added; the footer number is the position in this list

    def start(key, subtitle=None):
        """A content slide whose title and speaker notes are the presenter script's entry for `key`."""
        entry = by_key[key]
        slide = new_slide(prs, entry["title"], subtitle)
        notes(slide, "\n\n".join(entry["say"]).replace("*", ""))
        built.append(key)
        footer(slide, len(built))
        return slide

    # title (no footer)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text_box(s, MARGIN, 2.2, W - 2 * MARGIN, 1.4, ["Bisaya post-class speech and self-reported affect"], size=40, bold=True)
    text_box(s, MARGIN, 3.55, W - 2 * MARGIN, 0.9,
             ["Can one spoken word, said right after class, tell us how a student feels?"], size=22, color=MUTED)
    text_box(s, MARGIN, 5.2, W - 2 * MARGIN, 0.9,
             [f"CS412 mini-project · Group 3 · second-year Computer Science students · results of {date.today().isoformat()}"],
             size=16, color=MUTED)
    notes(s, "\n\n".join(by_key["title"]["say"]).replace("*", ""))
    built.append("title")

    # the question
    s = start("question", "One word, two ratings, and a strict test")
    text_box(s, MARGIN, 1.9, 7.4, 4.8, [
        ("Right after class, each participant said one Bisaya word that matched their reaction", 1),
        (f"Suggested words: {', '.join(PROTOCOL_WORDS)}", 2),
        ("Then they rated their feeling on two scales from 1 to 5", 1),
        ("Valence: how pleasant the feeling is", 2),
        ("Arousal: how much energy the feeling has", 2),
        ("Our question: can the sound of the word tell us those ratings?", 1),
        ("The test is strict: the computer is always tested on people it has never heard", 1),
    ])
    tiles_left = 8.4
    for k, (big, label) in enumerate([("1", "word per participant"), ("2", "ratings from 1 to 5"),
                                      (str(n_features), "voice measurements per recording")]):
        y = 1.95 + k * 1.6
        shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(tiles_left), Inches(y), Inches(4.3), Inches(1.4))
        shape.adjustments[0] = 0.08
        shape.fill.solid()
        shape.fill.fore_color.rgb = TILE
        shape.line.fill.background()
        shape.shadow.inherit = False
        text_box(s, tiles_left + 0.2, y + 0.18, 1.5, 1.0, [big], size=40, bold=True, color=ACCENT)
        text_box(s, tiles_left + 1.6, y + 0.42, 2.6, 0.8, [label], size=16)

    # the data
    s = start("data", "What went into the analysis")
    tiles(s, [(str(n_meta), "participants in the sheet"), (str(n_recorded), "with a recording"),
              (str(n), "analysed"), (str(MIN_VALID_OBSERVATIONS), "required by the protocol")], top=1.95)
    text_box(s, MARGIN, 3.85, W - 2 * MARGIN, 3.0, [
        *([("Audio: one short 16 kHz WAV file per participant, converted by the group from the phone recordings", 1),
           (f"{facts['n_copies']} files match an original recording. {no_orig} could not be checked, and {edited} are "
            "edited or clipped", 2),
           ("We ran the models again without those files", 2)] if supplied_mode else
          [(f"{n_meta - n_recorded} participants have no recording. {len(retakes)} were recorded more than once "
            f"({extra_takes} extra takes); one take each was kept, chosen by a fixed rule that never looks at the "
            "ratings", 1),
           *([(f"A later delivery has converted copies for {len(supplied_new)} of them, "
               f"{int((supplied_new['flags'] != '').sum())} edited or damaged. They are not counted until the originals "
               "are supplied", 2)] if len(supplied_new) else [])]),
        (f"{missing_words} spoken words are blank in the sheet. We never guess a word from the feeling text", 1),
        (f"Noise: background voices in {flags.get('BACKGROUND_VOICES', 0)} of {n} recordings", 1),
        ("Consent: " + ("documented for all analysed participants" if consent and all(
            consent.get(c, False) for c in manifest.loc[manifest["include_primary"] == "True", "participant_code"])
            else "the sheet says Yes for everyone, but the log of signed forms is not complete yet"), 1),
    ], size=17)

    # the pipeline
    s = start("pipeline", "One command runs eight steps. Each step writes its own report")
    flow(s, [("Check", "every file and every sheet cell is checked; nothing is changed"),
             ("Clean", "fixes by fixed rules only; every fix is logged"),
             ("Find the word", "the computer finds the spoken word in each clip, without seeing the ratings"),
             ("Measure", f"{n_features} numbers per voice: pitch, loudness, sound shape"),
             ("Explore", "charts, and simple links between voice and ratings"),
             ("Train", "three models learn from some participants and are tested on the others, "
                       f"{train['n_repeats']} times"),
             ("Test", f"is the score better than luck? {train['n_permutations']} shuffles of the ratings, then a "
                      "correction"),
             ("Word check", "does the model hear the feeling, or only the word?")], top=1.95)
    text_box(s, MARGIN, 5.75, W - 2 * MARGIN, 1.0,
             ["Three rules keep the test fair: no participant is in both the training and the test data · the model's "
              "settings are learned from the training data only · the ratings are never used to cut or choose audio"],
             size=15, color=MUTED)

    # the ratings
    s = start("ratings", f"Pleasant feelings came with higher energy (Spearman correlation {va[0]:.2f}, n = {va[2]})"
              if va[0] > 0 else f"Valence and arousal: Spearman correlation {va[0]:.2f}, n = {va[2]}")
    bottom = picture(s, FIGURES / "eda" / "fig01_affect_distributions.png", MARGIN, 1.85, 7.6, 4.9)
    text_box(s, MARGIN, bottom + 0.1, 7.6, 0.6,
             ["Blue = everyone in the sheet · orange = the participants we analysed"
              + (" (here they are the same people)" if n == n_meta else "")], size=13, color=MUTED)
    not_run = ev["tasks_not_modelled"]
    text_box(s, 8.5, 1.95, 4.25, 4.8, [
        ("Most participants felt neutral to pleasant, with moderate to high energy", 1),
        ("Some scores were chosen by very few people, so we grouped the scores before modelling:", 1),
        *[(TASK_GROUPS.get(t, t), 2) for t in ev["tasks_modelled"]],
        (f"Rule: every group needs at least {MIN_CLASS_SIZE} people"
         + (". Not modelled: " + "; ".join(TASK_SHORT.get(t, t) for t in not_run) if not_run else ""), 1),
    ], size=16)

    # features and ratings
    s = start("features", f"Six voice features chosen in advance, out of {n_features} measured")
    rows = []
    for f, label in PRESPECIFIED.items():
        a = sp[(sp["feature"] == f) & (sp["rating"] == "arousal")].iloc[0]
        v = sp[(sp["feature"] == f) & (sp["rating"] == "valence")].iloc[0]
        rows.append({"Voice feature": label, "Link with arousal (ρ)": signed(a.spearman_rho), "p": f"{a.p_value:.3f}",
                     "Link with valence (ρ)": signed(v.spearman_rho), "p ": f"{v.p_value:.3f}"})
    table(s, pd.DataFrame(rows), MARGIN, 1.95, 7.3, [2.6, 1.55, 0.8, 1.55, 0.8], size=14)
    six = sp[sp["feature"].isin(PRESPECIFIED)]
    strongest = float(six.loc[six["spearman_rho"].abs().idxmax(), "spearman_rho"])
    pre_sig = bool((six["q_value_bh"] < 0.05).any())
    top_sig = sp[(sp["q_value_bh"] < 0.05)].sort_values("p_value")
    text_box(s, 8.3, 1.95, 4.45, 4.9, [
        ("ρ is a correlation: 0 means no link, +1 or −1 means a perfect link", 1),
        (f"The strongest link among the six is {signed(strongest)}" + (", which is weak" if abs(strongest) < 0.3 else ""), 1),
        ("None of the six is significant after the correction for testing many features" if not pre_sig
         else "At least one of the six is significant after the correction for testing many features", 1),
        (f"All {n_features} features: {fdr['arousal']} significant for arousal, {fdr['valence']} for valence", 1),
    ] + ([(f"{r.feature} (ρ = {signed(r.spearman_rho)})", 2) for r in top_sig.itertuples()][:4]) + (
        [("Those were not chosen in advance, and their meaning is unclear", 1)] if len(top_sig) else []) + [
        (f"With {n} participants, only links of about {rho_min:.2f} or stronger can be found reliably", 1),
    ], size=15)

    # how to read the results
    n_tests = int(summ["perm_p"].notna().sum())
    groups = sorted({int(round(1 / summ[(summ["task"] == t) & (summ["model"] == "dummy_majority")].iloc[0].chance_ba))
                     for t in ev["tasks_modelled"]}, reverse=True)
    s = start("reading", "Five ideas you need for the next slides")
    table(s, pd.DataFrame([
        {"Idea": "Chance", "What it means": "The score you get by pure guessing",
         "In our study": " · ".join(f"{1 / k:.2f} with {k} groups" for k in groups)},
        {"Idea": "Balanced accuracy",
         "What it means": "The average of the scores for each group, so always giving the most common answer does not help",
         "In our study": "From 0 to 1. Higher is better"},
        {"Idea": "95% interval", "What it means": "The range where the true score probably lies",
         "In our study": f"Wide, because {n} participants are few"},
        {"Idea": "p-value",
         "What it means": "How often shuffled ratings, which mean nothing, score as high as the real ratings",
         "In our study": "Below 0.05 means it is unlikely to be luck"},
        {"Idea": "Correction (Holm)",
         "What it means": "With many tests, some look good by luck. The correction raises the p-values to account for that",
         "In our study": f"{n_tests} tests. A result counts only if p stays below 0.05"},
    ]), MARGIN, 1.95, W - 2 * MARGIN, [2.5, 5.9, 3.73], size=15, row_h=0.8)

    # the models
    s = start("models", "The best of three models for each target, always tested on people the model never heard")
    rows = []
    for task in ev["tasks_modelled"]:
        t = summ[summ["task"] == task]
        best = t[t["model"].isin(learned)].sort_values("pooled_ba", ascending=False).iloc[0]
        dummy = t[t["model"] == "dummy_majority"].iloc[0]
        rows.append({"Target": TASK_LABEL.get(task, task), "Best model": MODEL_LABEL[best.model],
                     "Balanced accuracy [95% interval]":
                         f"{best.pooled_ba:.2f} [{best.pooled_ba_ci_low:.2f}, {best.pooled_ba_ci_high:.2f}]",
                     "Chance": f"{dummy.pooled_ba:.2f}", "p-value": f"{best.perm_p:.3f}",
                     "p after correction": f"{best.perm_p_holm:.3f}"})
    table(s, pd.DataFrame(rows), MARGIN, 1.95, W - 2 * MARGIN, [3.3, 1.9, 3.1, 1.0, 1.1, 1.73], size=15, row_h=0.5)
    n_raw = int((summ["perm_p"] < 0.05).sum())
    text_box(s, MARGIN, 4.35, W - 2 * MARGIN, 2.4, [
        (("No model stays below 0.05 after the correction for " if not any_sig
          else "Some models stay below 0.05 after the correction for ") + f"{n_tests} tests", 1),
        (f"{n_raw} of {n_tests} p-values {'is' if n_raw == 1 else 'are'} below 0.05 before the correction. With "
         f"{n_tests} tests, about {0.05 * n_tests:.2f} lucky results are expected, so that is not evidence", 1),
        (f"The intervals are wide: {facts['crossing']} of the {facts['n_learned']} model intervals include the chance "
         "level", 1),
    ], size=17)

    # the same result as a figure
    s = start("figure", "Dot = score · bar = 95% interval · line = chance · grey = baselines that ignore the "
                        "audio · blue = our models")
    bottom = picture(s, FIGURES / "evaluation" / "fig_eval_balanced_accuracy.png", MARGIN, 1.85, W - 2 * MARGIN, 3.9)
    text_box(s, MARGIN, bottom + 0.2, W - 2 * MARGIN, 1.5, [
        ("Names in the figure: " + " · ".join(f"{t} = {TASK_SHORT.get(t, t)}" for t in ev["tasks_modelled"]), 1),
        ("“Dummy” rows are the baselines: they answer without listening to the audio", 1),
        (f"{facts['crossing']} of the {facts['n_learned']} blue bars cross the chance line", 1),
    ], size=16)

    # the word confound
    s = start("word", "The chosen word is linked to the feeling. This is called a lexical confound")
    v2 = assoc["valence_2"]["table"]
    kapoy_n, other_n = sum(v2["Kapoy"].values()), sum(v2["other"].values())
    text_box(s, MARGIN, 1.95, 6.9, 4.8, [
        (f"The computer can tell “Kapoy” from the other words by sound alone: balanced accuracy "
         f"{probe['balanced_accuracy_mean_10_repeats']:.2f} (chance 0.50, p = {probe['permutation_p']:.3f})", 1),
        (f"Not pleasant: {v2['Kapoy']['NotPleasant']} of {kapoy_n} Kapoy speakers, against {v2['other']['NotPleasant']} "
         f"of {other_n} speakers of other words (Fisher test, p = {assoc['valence_2']['p_value']:.3f})", 1),
        ("So a model can score above chance by recognising the word, without hearing any feeling", 1),
        (f"Among Kapoy speakers alone, {lex['within_word_significant_q05']} of {lex['within_word_tests']} links between "
         "voice and rating are significant after correction", 1),
    ], size=17)
    word_tab = pd.DataFrame([{"Word": "Kapoy", "Not pleasant (1–3)": v2["Kapoy"]["NotPleasant"],
                              "Pleasant (4–5)": v2["Kapoy"]["Pleasant"]},
                             {"Word": "other words", "Not pleasant (1–3)": v2["other"]["NotPleasant"],
                              "Pleasant (4–5)": v2["other"]["Pleasant"]}])
    text_box(s, 7.95, 1.95, 4.8, 0.45, ["Number of participants, by word and feeling"], size=14, color=MUTED)
    table(s, word_tab, 7.95, 2.45, 4.8, [1.3, 1.95, 1.55], size=14, row_h=0.55)

    # limits
    s = start("limits", "Why we call the results preliminary")
    text_box(s, MARGIN, 1.95, W - 2 * MARGIN, 4.9, [
        (f"Small sample: {n} participants"
         + (f", below the protocol minimum of {MIN_VALID_OBSERVATIONS}" if n < MIN_VALID_OBSERVATIONS else ""), 1),
        (f"Only strong links (about {rho_min:.2f} or more) can be found reliably. A moderate link of 0.3 needs about "
         f"{n_for_rho(0.3)} participants", 2),
        *([(f"Audio: converted copies, not the original recordings. {no_orig} could not be checked and {edited} are "
            "edited or clipped", 1)] if supplied_mode else []),
        (f"Noise: background voices in {flags.get('BACKGROUND_VOICES', 0)} of {n} recordings, made in a hallway right "
         "after class", 1),
        *([(f"Recording setting: {n_cfg} files " + ("come from recordings made with" if supplied_mode else "used")
            + " a different audio setting"
            + (f", which changes {eda['config_effect_features_fdr05']} of {n_features} features"
               if eda["config_effect_features_fdr05"] else "; no significant difference was found"), 1)] if n_cfg else []),
        ("One word per person: we cannot separate a person's normal voice from their feeling", 1),
        ("One context: one year level, one class activity and one session, so contexts cannot be compared", 1),
        ("We measured links, not causes", 1),
    ], size=18)

    # conclusion
    s = start("conclusion")
    text_box(s, MARGIN, 2.0, W - 2 * MARGIN, 1.4, [headline + "."], size=26, bold=True, color=ACCENT)
    text_box(s, MARGIN, 3.5, W - 2 * MARGIN, 3.2, [
        ("The pipeline is complete, tested and repeatable: new data can be analysed with one command", 1),
        ("The chosen word is linked to the feeling, so a future study must control the word", 1),
        (f"The sample is too small: a real study needs about {n_for_rho(0.3)} participants", 1),
    ], size=20)

    # next steps
    s = start("next")
    text_box(s, MARGIN, 1.95, W - 2 * MARGIN, 4.9, [
        ("Complete the consent log from the signed forms", 1),
        ("Two group members listen to every clip, without seeing the ratings, to check the words and where they "
         "start and end", 1),
        *([(f"Find the original recordings for the {no_orig} files we could not check", 1)] if supplied_mode else []),
        ("Record more participants: one device, one setting, a quiet place, one take", 1),
        ("Follow the analysis plan we wrote before seeing new data", 1),
        ("Confirm the open protocol questions with our instructor", 1),
    ], size=20)

    if built != [entry["key"] for entry in script]:
        raise RuntimeError(f"the slides {built} do not match the presenter script (10b_build_presentation_script.py)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    log.info("presentation: %d slides -> %s", len(prs.slides), rel(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
