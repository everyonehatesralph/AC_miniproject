"""
10c_build_defense_documentation.py -- the project explained, for the defense.

Writes reports/presentation/defense_documentation.md and a printable, self-contained .html. Where the presenter
script (10b) says what to say, this document explains how the project works and why each choice was made:
the path from recording to result, every step with its reason and the questions it may raise, how to read each
table and figure, where each number on the slides comes from, and a glossary.

Every number is read from the pipeline's result files (through 10b's facts()), and the settings quoted from the
method are read from src/common.py, so the document cannot disagree with the results or the code. Only the
standard library is used:
    python src/10c_build_defense_documentation.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "reports" / "presentation" / "defense_documentation.md"
OUT_HTML = OUT_MD.with_suffix(".html")
COMMON = (ROOT / "src" / "common.py").read_text(encoding="utf-8")

RULES = {"R1": "remove stray spaces in the participant code", "R2": "remove stray spaces in text cells",
         "R3": "write the year level as a number",
         "R4": "map the class activity to the protocol's category (the instructor still has to confirm this mapping)",
         "R5": "use the protocol's spelling for the session", "R6": "write each rating as a whole number",
         "R7": "do not use a rating that is crossed out in the sheet", "R8": "write consent as Y or N"}
FLAGS = {  # recording flag -> plain meaning ({} is filled from the settings in common.py)
    "BACKGROUND_VOICES": "other voices are heard outside the word (at least {BACKGROUND_VOICED_S} s of voiced sound)",
    "LOW_SBR": "the word is less than {LOW_SBR_DB} dB louder than the background",
    "COMPETING_EVENT": "another sound almost as loud as the word (within {SEG_COMPETING_DB} dB) is elsewhere in the clip",
    "TOO_SHORT_FOR_WORD": "the located sound is shorter than the word should be",
    "EDGE_START": "the word touches the start of the file, so its beginning may be cut",
    "EDGE_END": "the word touches the end of the file, so its end may be cut",
    "LONG_SEGMENT": "the located sound is longer than {SEG_LONG_S} s, so it may hold more than one word",
    "DIGITAL_SILENCE": "at least {DIGITAL_SILENCE_S} s of exact zeros, which a phone recording never contains",
    "CLIPPING": "the sound was too loud for the recorder, so its peaks are cut off",
    "NO_CAPTURE_TIME": "the file does not carry the date and time of the recording",
}
FEATURE_GROUPS = {
    "MFCC": "13 numbers for the shape of the sound spectrum; we keep the average and the spread of each over the word",
    "delta MFCC": "how fast those 13 numbers change during the word",
    "delta-delta MFCC": "how fast that change itself changes",
    "F0 (pitch)": "how high the voice is: average, middle value, lowest, highest, spread, range and slope",
    "energy": "loudness: average, middle value, lowest, highest and spread",
    "zero-crossing rate": "how noisy or hissing the sound is",
    "spectral": "where the energy sits in the spectrum: brightness (centroid), width and roll-off",
    "duration/voicing": "how long the word is, and how much of it is voiced",
}
GROUPS = {"arousal_3": "arousal in three groups: low (1–2), moderate (3), high (4–5)",
          "valence_2": "valence in two groups: not pleasant (1–3), pleasant (4–5)",
          "arousal_2": "arousal in two groups: not high (1–3), high (4–5)",
          "valence_3": "valence in three groups: unpleasant (1–2), neutral (3), pleasant (4–5)",
          "valence_5": "the five valence scores one by one", "arousal_5": "the five arousal scores one by one"}
GLOSSARY = [
    ("Affect", "A feeling. Here it is described by two ratings: valence and arousal."),
    ("Valence", "How pleasant the feeling is, from 1 (very unpleasant) to 5 (very pleasant)."),
    ("Arousal", "How much energy the feeling has, from 1 (very low) to 5 (very high)."),
    ("Self-report", "What the participant says about their own feeling. It is our correct answer."),
    ("Acoustic feature", "A number that describes a sound, measured by the computer."),
    ("Pitch (F0)", "How high or low the voice is, in Hz."),
    ("Energy (RMS)", "How loud the sound is."),
    ("MFCC", "13 numbers that describe the shape of the sound spectrum. They are what makes voices and vowels sound different."),
    ("Zero-crossing rate", "How often the sound wave crosses zero. Noisy and hissing sounds have a high rate."),
    ("Spectral centroid", "The brightness of the sound: high when the sound has much high-frequency content."),
    ("Sampling rate", "How many sound values are stored per second. Ours is 16 000 (16 kHz), enough for speech."),
    ("SBR", "Speech-to-background ratio: how much louder the word is than the background, in dB."),
    ("Group (class)", "One of the answers a model can give, for example low, moderate or high arousal."),
    ("Model", "A program that learns a rule from examples and then gives an answer for a new example."),
    ("Training and test data", "The model learns from the training data. It is scored on the test data, which it has never seen."),
    ("Cross-validation", "Splitting the participants into parts, and letting each part be the test data once."),
    ("Nested cross-validation", "A second split inside the training data, used only to choose the model's settings."),
    ("Speaker-independent", "No person is in both the training and the test data."),
    ("Data leakage", "Information about the test data that reaches the model during training. It makes scores look too good."),
    ("Setting (hyperparameter)", "A choice that controls how a model learns, for example how many trees a forest has."),
    ("Baseline (dummy model)", "A model that never looks at the audio. A real model must beat it."),
    ("Class weight", "Making each group count equally, even when it has fewer people."),
    ("Balanced accuracy", "The average of the scores for each group. Always giving the most common answer scores only the chance level."),
    ("Chance level", "The balanced accuracy of pure guessing: 1 divided by the number of groups."),
    ("95% interval", "The range where the true value probably lies. A wide interval means an uncertain result."),
    ("Bootstrap", "Drawing the participants again at random, many times, to see how much a score changes."),
    ("p-value", "How often results as good as ours appear when there is really nothing to find. Small means unlikely to be luck."),
    ("Permutation test", "Shuffling the ratings so they mean nothing, running everything again, and counting how often the score is as high."),
    ("Multiple testing", "Running many tests. Some will look good by luck alone."),
    ("Holm correction", "A correction for multiple testing. It raises the p-values of the model tests."),
    ("FDR (Benjamini–Hochberg)", "A correction for multiple testing, used for the many feature correlations."),
    ("Spearman correlation (ρ)", "How strongly two things rise together, from −1 to +1. It uses ranks, so it suits 1-to-5 ratings."),
    ("Statistical power", "The chance of finding an effect that really exists. Small samples have low power."),
    ("Confound", "A hidden third thing that explains a link between two others."),
    ("Lexical confound", "The spoken word is linked to the feeling, so a model can use the word instead of the feeling."),
    ("Sensitivity analysis", "Repeating the analysis without a doubtful part of the data, to see whether the result holds."),
    ("Reproducible", "Anyone can run the same code on the same data and get the same result."),
    ("Pseudonymous code", "A code such as Y2-041 that stands for a participant, so no name is stored."),
]


def script_module():
    """10b_build_presentation_script.py: the facts, the slide script and the Markdown-to-HTML page."""
    path = Path(__file__).with_name("10b_build_presentation_script.py")
    spec = importlib.util.spec_from_file_location("presentation_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def const(name: str) -> str:
    """A numeric setting from src/common.py, as text (5 instead of 5.0)."""
    match = re.search(rf"^{name}\s*=\s*([-0-9.]+)", COMMON, re.M)
    if not match:
        raise KeyError(f"{name} is not a numeric setting in src/common.py")
    value = float(match.group(1))
    return str(int(value)) if value == int(value) else f"{value:g}"


def ms(name: str) -> str:
    """A setting given in seconds, as milliseconds."""
    return f"{float(const(name)) * 1000:g}"


def build(ps, f: dict) -> str:
    d = ps.derive(f)
    n, a3, v2, sm, probe, kapoy, other, lead = d.n, d.a3, d.v2, d.sm, d.probe, d.kapoy, d.other, d.lead
    slides = ps.slide_script(f)
    pt = ps.pending_text(f)
    num = {s["key"]: i for i, s in enumerate(slides, 1)}
    title = {s["key"]: s["title"] for s in slides}
    flag_text = {k: v.format(**{c: const(c) for c in re.findall(r"\{(\w+)\}", v)}) for k, v in FLAGS.items()}
    modelled = [t for t in f["targets"] if f["targets"][t]["viable"]]
    not_modelled = [t for t in f["targets"] if not f["targets"][t]["viable"]]
    perm_floor = 1 / (f["n_perm"] + 1)
    hits = lambda p: round(float(p) * (f["n_perm"] + 1)) - 1  # noqa: E731  shuffles that scored at least as high
    kapoy_n, other_n = sum(kapoy.values()), sum(other.values())
    best_line = (f"The best score was {a3['ba']:.2f}, where pure guessing gives {a3['chance']:.2f}. After the correction for "
                 f"{f['n_tests']} tests its p-value is {a3['holm']:.3f}, so it is not significant." if a3 else "")

    L = ["# Defense documentation: the project explained", "",
         f"*Group 3 · Bisaya post-class speech and self-reported affect · {n} participants. Built by "
         "`src/10c_build_defense_documentation.py` from the result files and the settings in the code, so every number "
         "matches the slides and the report.*", "",
         "## How to use this", "",
         "- This document explains **how the project works and why**. The presentation script "
         "(`presentation_script.html`) has **what to say** and the likely questions with answers.",
         "- Read sections 1 to 3 once, from start to end. Use sections 4 to 6 to look things up.",
         "- Each step in section 3 has three parts: what happens, why we did it this way, and what you may be asked.", "",
         "## 1. The project on one page", "",
         "- **The question.** Can the sound of one spoken Bisaya word, said right after class, tell us how a student "
         "feels?",
         f"- **The participants.** {n} second-year Computer Science students. Each one has a code such as Y2-041. No "
         "names are stored.",
         "- **What each participant gave.** One spoken word, and two ratings from 1 to 5: valence (how pleasant the "
         "feeling is) and arousal (how much energy it has).",
         f"- **What the computer measured.** {f['n_features']} numbers that describe the sound of the word.",
         "- **The test.** Three kinds of models tried to guess the rating group from those numbers. Each model was "
         "always tested on participants it had not learned from.",
         f"- **The result.** {d.headline}. {best_line}",
         f"- **The trap we found.** The word is linked to the feeling: {kapoy['NotPleasant']} of {kapoy_n} Kapoy speakers "
         f"felt not pleasant, against {other['NotPleasant']} of {other_n} speakers of other words. The computer can also "
         f"tell Kapoy from other words by sound (balanced accuracy {probe['balanced_accuracy_mean_10_repeats']:.2f}). So a "
         "model can score above chance by recognising the word, not the feeling.",
         f"- **Why the project is still useful.** The whole analysis is complete, tested and repeatable. We know the word "
         f"must be controlled in a future study. And we know how many participants such a study needs: about "
         f"{f['n_for_03']}.", "",
         "## 2. From recording to result", "",
         "| Step | What goes in | What happens | What comes out |", "|---|---|---|---|",
         "| Collect (before the pipeline) | participants | each one says a word on a phone video and gives two ratings | "
         "the videos and the sheet |"]
    if f["supplied"]:
        L.append(f"| Convert (before the pipeline) | the videos | a group member converted them to short 16 kHz WAV files "
                 f"using {ps.PREPARED_WITH} | {n} WAV files (`{f['delivery']}`) |")
    L += ["| 1. Check | the sheet and the audio | every cell and every file is checked; nothing is changed | "
          "`reports/phase0_data_audit.md` |",
          "| 2. Clean | the sheet | fixes by fixed rules only, each one logged | `data/metadata_cleaned.csv`, "
          "`reports/phase1/cleaning_log.csv` |",
          "| 3. Find the word | the audio | each file is checked, the word is located, doubtful files are marked | "
          "`data/analysis_manifest.csv`, `reports/data_quality_report.md` |",
          f"| 4. Measure | the located words | {f['n_features']} numbers per recording | `data/acoustic_features.csv` |",
          "| 5. Explore | the numbers and the ratings | charts and simple links | `reports/exploratory_analysis.md`, "
          "`figures/eda/` |",
          f"| 6. Train | the numbers and the rating groups | 3 models and 2 baselines, tested on unseen participants; "
          f"{f['n_perm']} shuffles per test | the `models/` folder |",
          "| 7. Evaluate | the models' answers | scores, intervals, p-values, corrections | `reports/model_evaluation.md`, "
          "`reports/evaluation/` |",
          "| 8. Word check | the numbers, the words, the answers | is it the voice or the word? | "
          "`reports/lexical_control_analysis.md` |",
          "| 9. Report | all results | the report, the web pages and the slides are written | "
          "`reports/final_group_research_report.md`, `reports/presentation/` |",
          f"| 10. Validate | the whole project | {sum(f['valid'].values())} automatic checks | `reports/validation_report.md` |", "",
          f"One command runs steps 1 to 10: `python run_pipeline.py`. Step 6 is the slow one: about {f['train_minutes']} "
          "minutes on our PC.", "",
          "### The tools", "",
          "| Tool | What we use it for |", "|---|---|",
          "| Python | the language of every step |",
          "| pandas, numpy | tables and numbers |",
          "| Praat (through parselmouth) | measuring pitch |",
          "| librosa | measuring loudness, the spectrum and the MFCCs |",
          "| scikit-learn | the models, the cross-validation and the baselines |",
          "| scipy | the statistical tests |",
          "| matplotlib | the figures |",
          "| python-pptx | writing the slides |",
          "| pytest | automatic tests of the code |",
          "| Git and GitHub | keeping every version of the code and the reports |", "",
          "## 3. Each step: what, why, and what you may be asked", "",
          "### 3.1 The correct answers (labels)", "",
          "- **What happens:** the correct answer for each recording is the participant's own rating. Because some scores "
          "were chosen by very few people, we group the scores: " + "; ".join(GROUPS.get(t, t) for t in modelled) + ".",
          f"- **The rule:** a grouping is used only if every group has at least {const('MIN_CLASS_SIZE')} people. Not used: "
          + ("; ".join(GROUPS.get(t, t) for t in not_modelled) if not_modelled else "none") + ".",
          "- **Why this way:** the protocol defines the feeling as what the participant reports. A listener's guess, or "
          "the meaning of the word, would be our opinion and not data.",
          f"- **Details to know:** {f['words_missing']} spoken words are blank in the sheet and stay blank (label "
          f"UNVERIFIED). {f['words_offlist']} words are not on the protocol's suggested list; we kept them as the "
          "participants said them."
          + (f" The arousal rating of {ps.codes(f['no_arousal'])} is crossed out in the sheet, so it is not used, and the "
             f"arousal analyses have {f['n_aro']} participants." if f["no_arousal"] else ""),
          "- **If asked “Did you change any answer?”** No. Ratings are only written as whole numbers. Nothing was "
          "changed because it looked unusual, and no word was guessed from the feeling text.", "",
          "### 3.2 Checking and cleaning the sheet", "",
          "- **What happens:** step 1 reads every cell and every file and lists the problems. It changes nothing. Step 2 "
          "then applies these fixed rules:"]
    for rule, text in RULES.items():
        k = f["changes_by_rule"].get(rule, 0)
        L.append(f"  - {rule}: {text} ({k} change{'' if k == 1 else 's'})")
    L += ["- **Two more rules for later:** the group can supply corrections in a file, and a blank word can be filled "
          "from the listening review, but only when two listeners agree and only with a protocol word. Neither has been "
          "used yet.",
          *([f"- **A newer sheet is waiting:** on {pt['day']} the group completed the sheet. {pt['words']} blank words "
             f"were filled in {pt['words_source']}, and the ratings of {pt['codes']} were corrected "
             f"{pt['ratings_source']}. It is stored beside the sheet in use and is not analysed yet. Every changed "
             "cell is listed in `reports/sheet_update_pending.md`. A rating is what the models learn from, so the "
             "whole analysis must be run again before any number may be quoted from the new sheet."] if pt else []),
          "- **Why this way:** no cell is edited by hand, so every change can be traced. The original value stays in the "
          "data beside the cleaned one, and each change is a line in `reports/phase1/cleaning_log.csv`.",
          "- **If asked “How do we know you did not edit the data?”** Every change is one of the rules above and is "
          "logged with the old and the new value.", "",
          "### 3.3 The audio and finding the word", "",
          "- **The format:** WAV, 16 kHz, one channel, 16 bits. The pipeline does not remove noise, change the loudness "
          "or filter the sound. The protocol forbids that, and it would change the things we measure.",
          f"- **How the word is found:** the computer measures the loudness every {ms('HOP_S')} ms. The background level "
          f"is the loudness that {const('SEG_FLOOR_PERCENTILE')}% of the clip stays below. The word is the stretch around "
          f"the loudest point that stays clearly above the background (at least {const('SEG_MIN_REL_DB')} dB above it, or "
          f"half-way to the peak if that is higher). Short dips of less than {const('SEG_MAX_GAP_S')} s inside a word are bridged, and "
          f"{const('SEG_PAD_S')} s is kept on each side.",
          "- **Why this way:** the rule is automatic, the same for every file, and it never sees the ratings. So it cannot "
          "favour any result.",
          "- **Doubtful files are marked, not removed.** The marks and how many of the "
          f"{n} recordings carry each:"]
    for flag, count in f["flags_all"].items():
        L.append(f"  - `{flag}` ({count}): {flag_text.get(flag, 'see the data dictionary')}")
    L += [f"- **Uncertain word location:** {f['seg_critical']} of {n} recordings have a mark that puts the location of the "
          "word in doubt. Nobody has checked them by listening yet; that is the listening review in our next steps.",
          (f"- **Where the files came from:** {f['n_copies']} match an original recording. {len(f['no_orig'])} could not be "
           f"checked ({ps.codes(f['no_orig'])}), and {len(f['edited'])} of the files are edited or clipped "
           f"({ps.codes(f['edited'])})." if f["supplied"] else
           "- **Where the files came from:** the original phone recordings."),
          "- **If asked “Is the automatic word location correct?”** We do not claim it is perfect. We mark the doubtful "
          "files and run the models again without them (section 3.9).", "",
          f"### 3.4 The {f['n_features']} features", "",
          f"- **What happens:** the located word is cut into windows of {ms('FRAME_S')} ms, one every {ms('HOP_S')} ms. "
          "The computer measures each window and then summarises the whole word.",
          "", "| Group | How many | What it describes |", "|---|---|---|"]
    for group, count in d.fg.items():
        L.append(f"| {group} | {count} | {FEATURE_GROUPS.get(group, '')} |")
    L += ["", f"- **Total:** " + " + ".join(str(v) for v in d.fg.values()) + f" = {sum(d.fg.values())}.",
          "- **Why these features:** the protocol lists them (Section XIV), and they are the usual features in speech "
          "research. The list was fixed before we saw any result.",
          "- **A caution about pitch:** pitch in Hz mostly shows whether a person's voice is naturally high or low. With "
          "one recording per person we cannot subtract the person's normal voice.",
          "- **If asked “Why not let a neural network learn the features?”** That needs thousands of recordings. We have "
          f"{n}.", "",
          "### 3.5 Exploring the data", "",
          "- **What happens:** we compute the Spearman correlation between every feature and each rating, then correct "
          f"for testing {f['n_features']} features (the FDR correction).",
          f"- **The result:** {f['fdr']['arousal']} features are significant for arousal and {f['fdr']['valence']} for "
          "valence. Of the six features chosen in advance, "
          + ("none is significant." if not f["pre_sig"] else f"{f['pre_sig']} are significant."),
          "- **Why Spearman:** the ratings are ordered scores from 1 to 5, not measurements. Spearman uses ranks, so it "
          "fits.",
          "- **Why choose six features in advance:** if you test many features and report only the best, you always find "
          "something by luck. Choosing first keeps us honest.",
          f"- **What the sample can show:** with {n} participants, only links of about {f['rho_min']:.2f} or stronger can be "
          f"found reliably (80% of the time). A moderate link of 0.3 needs about {f['n_for_03']} participants.", "",
          "### 3.6 Training and testing the models", "",
          "- **The three models:** logistic regression (straight boundaries between groups), SVM (curved boundaries) and "
          "random forest (300 decision trees that vote). They make different assumptions, so together they are a fair "
          "test.",
          "- **The two baselines:** one always answers the most common group; one guesses at random in the same "
          "proportions as the groups. Neither looks at the audio.",
          f"- **The test:** the participants are split into {const('N_OUTER_FOLDS')} parts. The model learns from "
          f"{int(const('N_OUTER_FOLDS')) - 1} parts and is tested on the last one, and each part is the test once. We "
          f"repeat this {f['n_repeats']} times with different splits, so one lucky split cannot decide the result.",
          "- **Choosing the settings:** inside each training set there is a second, smaller split. It is used only to "
          "choose the model's settings. The test part is never used for that. This is called nested cross-validation.",
          "- **Preparing the numbers:** a missing value is replaced by the middle value of the training data, and the "
          "features are put on the same scale using the training data only.",
          "- **Unequal groups:** each group counts equally (class weights). We did not create artificial samples.",
          "- **Why this way:** a real system would meet new students, so the model must be tested on people it has never "
          "heard. And the model must not see anything about the test participants while it learns.",
          "- **The code stops by itself** if a column that is not an acoustic feature enters the model, if a participant "
          "is in both the training and the test part, or if a test part lacks a group.",
          "- **If asked “Why not SMOTE?”** With so few people, artificial samples would be copies of the same few voices, "
          "and they leak information if they are made before the split.", "",
          "### 3.7 Scoring and statistics", "",
          "- **The score:** balanced accuracy, the average of the scores for each group. Pure guessing gives 1 divided by "
          "the number of groups.",
          "- **The interval:** we draw the participants again at random, with replacement, 2000 times and compute the "
          "score each time. The middle 95% of those scores is the 95% interval.",
          f"- **The p-value:** we shuffle the ratings so they no longer belong to the right recordings and run the whole "
          f"procedure again, {f['n_perm']} times. The p-value is (number of shuffles that score at least as high + 1) "
          f"divided by ({f['n_perm']} + 1). So the smallest possible p-value is {perm_floor:.3f}.",
          f"- **Our smallest p-value:** {float(sm['perm_p']):.3f}, for the {ps.MODEL[sm['model']]} on "
          f"{ps.TASK.get(sm['task'], sm['task'])}. It means {hits(sm['perm_p'])} of the {f['n_perm']} shuffles scored at "
          "least as high as the real data.",
          f"- **The correction:** we ran {f['n_tests']} tests ({len(ps.LEARNED)} models for each of {len(f['tasks'])} "
          f"targets). With {f['n_tests']} tests, about {0.05 * f['n_tests']:.2f} lucky results are expected. The Holm "
          f"correction multiplies the smallest p-value by {f['n_tests']}, the next by {f['n_tests'] - 1}, and so on. "
          f"{float(sm['perm_p']):.4f} × {f['n_tests']} = {float(sm['perm_p_holm']):.3f}.",
          "- **Comparing the models:** we use a t-test that is corrected for repeated cross-validation, because the "
          "same participants are tested again in every repeat. "
          + (f"{len(f['pairs_sig'])} comparison has p below 0.05 before any correction, so we do not claim that one model "
             "is better." if len(f["pairs_sig"]) == 1 else
             f"{len(f['pairs_sig'])} comparisons have p below 0.05 before any correction."),
          "- **If asked “Why is p below 0.05 not enough?”** Because it was one of "
          f"{f['n_tests']} tests. One lucky result among {f['n_tests']} is not evidence.", "",
          "### 3.8 The word check", "",
          "- **Why we did it:** each participant chose their own word. If the word is linked to the feeling, a model can "
          "use the word as a shortcut.",
          f"- **Check 1, which words can be analysed:** only Kapoy was said by enough people ({probe['n_kapoy']}).",
          f"- **Check 2, can the computer hear the word:** the same models tell Kapoy from other words with balanced "
          f"accuracy {probe['balanced_accuracy_mean_10_repeats']:.2f} (p = {probe['permutation_p']:.3f}"
          + (", the smallest possible value: no shuffle did as well" if abs(probe["permutation_p"] - perm_floor) < 1e-6 else "")
          + ").",
          f"- **Check 3, is the word linked to the feeling:** not pleasant for {kapoy['NotPleasant']} of {kapoy_n} Kapoy "
          f"speakers and for {other['NotPleasant']} of {other_n} speakers of other words (Fisher test, p = "
          f"{f['assoc_v']['p_value']:.3f}).",
          f"- **Check 4, the same word with different feelings:** among Kapoy speakers alone, {f['within_sig']} of "
          f"{f['within_tests']} links between voice and rating are significant.",
          "- **What it means:** a score above chance could come from the word and not from the feeling. A future study "
          "must use one fixed word, or enough recordings of each word.", "",
          "### 3.9 Running the models again without doubtful files", "",
          f"Scores of the {ps.MODEL[lead['model']]} when a doubtful part of the data is left out. With all data: "
          f"{lead['ba']:.2f} for {ps.TASK.get(lead['task'], lead['task'])}"
          + (f" and {v2['ba']:.2f} for valence." if v2 else "."), "",
          "| What is left out | Participants left | Arousal score | Valence score |", "|---|---|---|---|"]
    for scen, (ba, nn) in d.lead_sens.items():
        val = d.val_sens.get(scen)
        L.append(f"| {ps.SCENARIO.get(scen, scen).replace('without ', '')} | {nn} | {ba:.2f} | "
                 + (f"{val[0]:.2f}" if val else "–") + " |")
    L += ["", "- **How to read it:** the scores move up and down depending on which files are in. That shows the sample "
          "is too small for a stable answer.",
          "- **Why it matters for the defense:** our conclusion does not depend on the doubtful files. With or without "
          "them, no claim of recognition is supported.", "",
          "### 3.10 Checks, tests and repeatability", "",
          f"- **Validation:** step 10 checks the whole project: {f['valid']['PASS']} checks pass and {f['valid']['BLOCKED']} "
          "is blocked" + (" (the consent log)." if f["valid"]["BLOCKED"] else "."),
          "- **What is checked:** the number of participants, consent, no names in the data, one recording and one row "
          "of features per participant, the WAV format, no leakage, no participant in both training and test data, and "
          "that the report states the same numbers as the results.",
          "- **Code tests:** the rules that decide the data (which flags are set, how a word is found, how consent is "
          "read) have automatic tests that run on GitHub for every change.",
          f"- **Same result every time:** the random choices use a fixed seed ({const('RANDOM_STATE')}), and the versions of "
          "all tools are written down in `requirements.txt`.",
          f"- **What is not complete:** " + (ps.codes([b.split('. ', 1)[-1] for b in f["blocked_items"]])
                                           if f["blocked_items"] else "nothing") + ".", "",
          "## 4. How to read each table and figure", "",
          "### The tables on the slides", "",
          f"- **Slide {num['features']}, “{title['features']}”:** one row per voice feature. ρ is the Spearman correlation "
          "with the rating; p is its p-value before any correction. Look for ρ far from 0 and a small p. "
          + ("None of the six is significant after the correction." if not f["pre_sig"] else
             f"{f['pre_sig']} of the twelve links are significant after the correction."),
          f"- **Slide {num['models']}, “{title['models']}”:** one row per target, showing the best of the three models. "
          "Compare *Balanced accuracy* with *Chance*. The interval in brackets shows how uncertain the score is. "
          "*p-value* is before the correction, *p after correction* is the one that counts.",
          f"- **Slide {num['word']}, “{title['word']}”:** each cell is a number of participants. Read along a row: most "
          "Kapoy speakers are in *Not pleasant*, most speakers of other words are in *Pleasant*.", "",
          "### The figures", "",
          "| Figure | What it shows | How to read it |", "|---|---|---|",
          f"| `eda/fig01` | how many participants chose each score | {f['v35']} of {f['n_val']} valence ratings and "
          f"{f['a35']} of {f['n_aro']} arousal ratings are 3 or higher |",
          "| `eda/fig02` | each pair of valence and arousal scores, with the number of participants | numbers along the "
          f"diagonal mean the two ratings rise together (ρ = {f['va_rho']:.2f}) |",
          f"| `eda/fig03` | the size of each group in every grouping | the line marks {const('MIN_CLASS_SIZE')} people; a "
          "grouping with a bar below the line is not modelled |",
          f"| `eda/fig04` | how many participants said each word | Kapoy was said by {probe['n_kapoy']} participants; "
          f"{f['words_missing']} words are blank |",
          "| `eda/fig05` | the feeling words that participants wrote | descriptive only; it is not used by the models |",
          "| `eda/fig06` | how selected features are spread over the recordings | wide spreads mean the voices differ "
          "a lot from person to person |",
          "| `eda/fig07` | how strongly features are related to each other | strongly related features carry the same "
          "information |",
          f"| `eda/fig08` | the correlation of every feature with each rating | the panel titles say how many pass the "
          f"correction: {f['fdr']['arousal']} for arousal, {f['fdr']['valence']} for valence |",
          "| `eda/fig09`, `eda/fig10` | the six features chosen in advance, by arousal group and by valence group | each "
          "dot is one participant; where the groups overlap, the feature does not separate them |",
          "| `eda/fig11` | four features, for the two recording settings | "
          + (f"the {len(f['cfg'])} files of setting B look different, but {f['cfg_feats']} of {f['n_features']} features "
             f"differ after the correction (closest p = {f['cfg_min_q']:.3f}; {f['cfg_raw']} before it). If the title "
             "printed inside the figure says something else, the figure is older than this document: use these numbers"
             if f["cfg"] and f["cfg_min_q"] is not None else "the two settings are compared feature by feature") + " |",
          "| `eda/fig12` | the ratings of the participants, by the word they said | descriptive only; the test is in "
          "the word check (section 3.8) |",
          "| `eda/fig13` | how noisy the recordings are, by word | the black tick is the middle value; it shows "
          "whether some words were recorded in noisier conditions |",
          "| `eda/fig14`, `eda/fig15` | the average pitch and the average loudness of each participant, from lowest to "
          "highest | the colours are the rating groups; where they are mixed along the line, the feature does not "
          "separate the groups |",
          f"| `evaluation/fig_eval_balanced_accuracy` | the score of every model, with its 95% interval (slide "
          f"{num['figure']}) | {f['crossing']} of the {f['n_learned']} blue bars cross the chance line |",
          "| `evaluation/fig_eval_confusion_…` | for each model, which group it answered for each true group | rows are "
          "the true group, columns the answer; a good model has high numbers on the diagonal |",
          "| `evaluation/fig_eval_sensitivity` | the scores when doubtful files are left out (section 3.9) | the "
          "horizontal line is chance; compare each score with the line and with the score for all data |",
          f"| `lexical/fig_lexical_kapoy_features_vs_affect` | the six features against the ratings, for Kapoy speakers "
          f"only | {f['within_sig']} of {f['within_tests']} links are significant |", "",
          "## 5. Where each number on the slides comes from", "",
          "| Slide | Number | What it is | File |", "|---|---|---|---|",
          f"| {num['data']} | {f['n_meta']} and {n} | rows in the sheet; participants in the analysis | "
          "`data/metadata_cleaned.csv`, `data/analysis_manifest.csv` |"]
    if f["supplied"]:
        L.append(f"| {num['data']} | {f['n_copies']}, {len(f['no_orig'])}, {len(f['edited'])} | files that match an original; "
                 "files that could not be checked; files edited or clipped | `reports/supplied_audio_audit.csv` |")
    L += [f"| {num['data']} | {f['words_missing']} | blank spoken words | `reports/phase1/cleaning_summary.json` |",
          f"| {num['data']} | {f['bg']} of {n} | recordings with background voices | `reports/data_quality_summary.json` |",
          f"| {num['ratings']} | {f['va_rho']:.2f} | Spearman correlation of valence with arousal ({f['va_n']} participants) | "
          "`reports/eda/eda_summary.json` |",
          f"| {num['features']} | ρ and p in the table | correlation of each feature with each rating | "
          "`reports/eda/spearman_feature_vs_rating.csv` |",
          f"| {num['features']}, {num['limits']} | {f['rho_min']:.2f} and {f['n_for_03']} | the weakest link that {n} "
          "participants can show, and the participants needed for a link of 0.3 (5% significance, 80% power) | computed "
          "in `src/09c_build_presentation.py` |"]
    for t in f["tasks"]:
        L.append(f"| {num['models']} | {t['ba']:.2f} [{t['lo']:.2f}, {t['hi']:.2f}], p {t['p']:.3f} → {t['holm']:.3f} | "
                 f"{ps.TASK.get(t['task'], t['task'])}: score, 95% interval, p-value before and after correction | "
                 "`reports/evaluation/summary_metrics.csv` |")
    L += [f"| {num['models']}, {num['figure']} | {f['crossing']} of {f['n_learned']} | model intervals that include the chance "
          "level | `reports/evaluation/summary_metrics.csv` |",
          f"| {num['word']} | {probe['balanced_accuracy_mean_10_repeats']:.2f}, p = {probe['permutation_p']:.3f} | how well "
          "the audio tells Kapoy from other words | `reports/lexical/lexical_summary.json` |",
          f"| {num['word']} | {kapoy['NotPleasant']} of {kapoy_n}; {other['NotPleasant']} of {other_n}; p = "
          f"{f['assoc_v']['p_value']:.3f} | not-pleasant feelings by word, Fisher test | "
          "`reports/lexical/lexical_summary.json` |",
          f"| {num['word']} | {f['within_sig']} of {f['within_tests']} | significant links among Kapoy speakers alone | "
          "`reports/lexical/lexical_summary.json` |"]
    if f["cfg"] and f["cfg_min_q"] is not None:
        L.append(f"| {num['limits']} | {len(f['cfg'])} files; {f['cfg_raw']} of {f['n_features']}; {f['cfg_min_q']:.3f} | files "
                 "from the other recording setting; features that differ before the correction; closest corrected "
                 "p-value | `data/analysis_manifest.csv`, `reports/eda/recording_configuration_effect.csv` |")
    L += ["", "## 6. Glossary", "", "| Word | Meaning in plain words |", "|---|---|"]
    L += [f"| {word} | {meaning} |" for word, meaning in GLOSSARY]
    return "\n".join(L) + "\n"


def main() -> int:
    ps = script_module()
    md = build(ps, ps.facts())
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md, encoding="utf-8")
    OUT_HTML.write_text(ps.to_html(md, "Defense documentation"), encoding="utf-8")
    print(f"defense documentation -> {OUT_MD.relative_to(ROOT).as_posix()} and {OUT_HTML.name} ({len(md.split())} words)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
