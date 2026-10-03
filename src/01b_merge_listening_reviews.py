"""
01b_merge_listening_reviews.py -- blind listening review: answers -> agreement -> human-verified decisions.

Group members listen to every take in data/review/listening_review.html (built by 03b) WITHOUT seeing the
metadata word, the ratings or the automatic take choice, and save their answers as a CSV in
data/review/answers/. This script

  1. validates every answers file (one file per listener; codes, takes and values checked);
  2. measures agreement between listeners (percent agreement and Cohen's kappa for every pair);
  3. turns judgements that are UNANIMOUS among at least two listeners into decisions the pipeline uses:
       data/review/take_overrides_from_review.csv        the take that holds the whole word most clearly
       data/review/segment_overrides_from_review.csv     corrected word boundaries (listeners within 50 ms;
                                                          their mean is used)
       data/review/metadata_corrections_from_review.csv  a MISSING spoken word, when every listener heard
                                                          the same protocol word
     A row in the group's own take_overrides.csv / segment_overrides.csv / metadata_corrections.csv always
     wins (02 and 03 apply the files). Everything that is not unanimous is listed for a joint decision;
  4. writes reports/listening_review_report.md, reports/listening_review_agreement.csv and
     reports/listening_review_decisions.csv.

A heard word that differs from the metadata word is reported, never changed: the sheet is the
participant's record. Boundary judgements refer to the AUTOMATIC boundaries shown in the page; if the
automatic boundaries have changed since (new audio or a changed algorithm), those judgements are stale
and ignored until the take is listened to again.

Runs before 02 (run_pipeline.py step 01b). The take list comes from the previous run of 03
(reports/audio_quality_table.csv). Without answers files it writes empty decision files.

Usage: .venv/Scripts/python.exe src/01b_merge_listening_reviews.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import DATA, PROTOCOL_WORDS, RAW_METADATA_CSV, REPORTS, normalise_code, rel, setup_logging

REVIEW_DIR = DATA / "review"
ANSWERS_DIR = REVIEW_DIR / "answers"
QA_TABLE = REPORTS / "audio_quality_table.csv"
OUT_TAKES = REVIEW_DIR / "take_overrides_from_review.csv"
OUT_SEGMENTS = REVIEW_DIR / "segment_overrides_from_review.csv"
OUT_CORRECTIONS = REVIEW_DIR / "metadata_corrections_from_review.csv"
REPORT = REPORTS / "listening_review_report.md"
AGREEMENT_CSV = REPORTS / "listening_review_agreement.csv"
DECISIONS_CSV = REPORTS / "listening_review_decisions.csv"

ANSWER_COLUMNS = ["reviewer_initials", "participant_code", "take", "drive_file_id", "shown_start_s", "shown_end_s",
                  "heard_word", "whole_word_audible_Y_N", "boundaries_ok_Y_N", "corrected_start_s",
                  "corrected_end_s", "clarity_1_to_5", "other_voice_over_word_Y_N", "chosen_take_Y_N", "notes",
                  "page_version", "saved_at"]
YN_COLUMNS = ("whole_word_audible_Y_N", "boundaries_ok_Y_N", "other_voice_over_word_Y_N", "chosen_take_Y_N")
TIME_COLUMNS = ("shown_start_s", "shown_end_s", "corrected_start_s", "corrected_end_s")
TAKE_COLUMNS = ["participant_code", "drive_file_id", "reason", "decided_by"]
SEGMENT_COLUMNS = ["participant_code", "drive_file_id", "seg_start_s", "seg_end_s", "reviewer", "note"]
CORRECTION_COLUMNS = ["participant_code", "field", "new_value", "source", "verified_by", "date_verified"]
MIN_LISTENERS = 2       # a decision needs at least two listeners, all giving the same answer
BOUNDARY_TOL_S = 0.05   # corrected boundaries must agree within 50 ms (start and end separately)
SHOWN_TOL_S = 0.0005    # the boundaries shown in the page must still be the current automatic ones
NEEDS_DECISION = ("disagree", "MISMATCH", "group decision")

log = setup_logging("listening_review", REPORTS / "logs" / "01b_merge_listening_reviews.log")


def normalise_word(text: str) -> str:
    """'Kapoy!' / ' kapoy ' -> 'kapoy': letters only, case-folded, for comparing what listeners typed."""
    return "".join(ch for ch in str(text).casefold() if ch.isalpha())


def load_takes() -> pd.DataFrame:
    """Every take of the last run of 03, with the automatic (label-blind) word boundaries."""
    if not QA_TABLE.is_file():
        raise FileNotFoundError(f"{rel(QA_TABLE)} not found: run src/03_prepare_audio.py first")
    qa = pd.read_csv(QA_TABLE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "review_take" not in qa.columns:
        raise ValueError(f"{rel(QA_TABLE)} predates the listening review: re-run src/03_prepare_audio.py")
    takes = qa[["participant_code", "drive_file_id", "review_take", "selected", "auto_seg_start_s",
                "auto_seg_end_s"]].copy()
    takes["review_take"] = takes["review_take"].astype(int)
    takes["selected"] = takes["selected"] == "True"
    for col in ("auto_seg_start_s", "auto_seg_end_s"):
        takes[col] = takes[col].astype(float)
    return takes.sort_values(["participant_code", "review_take"]).reset_index(drop=True)


def load_answers(files: list[Path], takes: pd.DataFrame) -> pd.DataFrame:
    """All answers files, validated; one file per listener."""
    known = set(zip(takes["participant_code"], takes["drive_file_id"]))
    frames, owner = [], {}
    for path in files:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        missing = [c for c in ANSWER_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}; use the CSV saved from the review page")
        df = df[ANSWER_COLUMNS].apply(lambda s: s.str.strip())
        initials = set(df["reviewer_initials"].str.upper())
        if len(initials) != 1 or "" in initials:
            raise ValueError(f"{path.name}: one listener's initials expected, found {sorted(initials)}")
        who = initials.pop()
        if who in owner:
            raise ValueError(f"listener {who} has two answers files ({owner[who]}, {path.name}): keep only the latest")
        owner[who] = path.name
        df["reviewer_initials"] = who
        df["participant_code"] = df["participant_code"].map(lambda c: normalise_code(c) or c)
        unknown = [f"{c} [{i[:6]}]" for c, i in zip(df["participant_code"], df["drive_file_id"]) if (c, i) not in known]
        if unknown:
            raise ValueError(f"{path.name}: takes that are not in the current recordings: {unknown}")
        if df.duplicated(["participant_code", "drive_file_id"]).any():
            raise ValueError(f"{path.name}: the same take is answered twice")
        for col in YN_COLUMNS:
            df[col] = df[col].str.upper()
            bad = sorted(set(df[col]) - {"Y", "N", ""})
            if bad:
                raise ValueError(f"{path.name}: {col} must be Y, N or blank, found {bad}")
        bad = sorted(set(df["clarity_1_to_5"]) - {"", "1", "2", "3", "4", "5"})
        if bad:
            raise ValueError(f"{path.name}: clarity_1_to_5 must be 1-5 or blank, found {bad}")
        for col in TIME_COLUMNS:
            values = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")
            if (values.isna() & df[col].ne("")).any():
                raise ValueError(f"{path.name}: {col} must be a number of seconds")
            df[col] = values
        both = df["corrected_start_s"].notna() & df["corrected_end_s"].notna()
        if (both & (df["corrected_start_s"] >= df["corrected_end_s"])).any():
            raise ValueError(f"{path.name}: a corrected start is not before its end")
        chosen = df[df["chosen_take_Y_N"] == "Y"].groupby("participant_code").size()
        if (chosen > 1).any():
            raise ValueError(f"{path.name}: more than one chosen take for {chosen[chosen > 1].index.tolist()}")
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=ANSWER_COLUMNS + ["source_file", "word", "boundary_stale"])
    answers = pd.concat(frames, ignore_index=True)
    answers["word"] = answers["heard_word"].map(normalise_word)
    auto = takes.set_index(["participant_code", "drive_file_id"])[["auto_seg_start_s", "auto_seg_end_s"]]
    shown = answers.join(auto, on=["participant_code", "drive_file_id"])
    answers["boundary_stale"] = ((shown["shown_start_s"] - shown["auto_seg_start_s"]).abs().gt(SHOWN_TOL_S)
                                 | (shown["shown_end_s"] - shown["auto_seg_end_s"]).abs().gt(SHOWN_TOL_S)
                                 | shown["shown_start_s"].isna() | shown["shown_end_s"].isna())
    return answers


def chosen_take(rows: pd.DataFrame) -> str | None:
    """One listener's take choice for one participant: a drive_file_id, 'none' (no take holds the whole
    word), or None when the question was not answered."""
    vals = rows["chosen_take_Y_N"]
    if (vals == "Y").sum() == 1:
        return rows.loc[vals == "Y", "drive_file_id"].iloc[0]
    if len(vals) and (vals == "N").all():
        return "none"
    return None


def kappa(a: pd.Series, b: pd.Series, labels=None, weights=None) -> float:
    """Cohen's kappa; NaN when both listeners used one and the same category (kappa is undefined)."""
    if len(set(a) | set(b)) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b, labels=labels, weights=weights))


def agreement(answers: pd.DataFrame, takes: pd.DataFrame) -> pd.DataFrame:
    """Percent agreement and Cohen's kappa for every pair of listeners and every judgement."""
    per_take = answers.copy()
    per_take.loc[per_take["boundary_stale"], "boundaries_ok_Y_N"] = ""
    multi = set(takes.loc[takes.duplicated("participant_code", keep=False), "participant_code"])
    choices = pd.DataFrame([{"reviewer_initials": who, "participant_code": code, "choice": chosen_take(g) or ""}
                            for (who, code), g in answers.groupby(["reviewer_initials", "participant_code"])
                            if code in multi], columns=["reviewer_initials", "participant_code", "choice"])
    items = [("heard word (letters only, case ignored)", per_take, "word", ["participant_code", "drive_file_id"], {}),
             ("whole word audible", per_take, "whole_word_audible_Y_N", ["participant_code", "drive_file_id"], {}),
             ("automatic boundaries contain the whole word", per_take, "boundaries_ok_Y_N",
              ["participant_code", "drive_file_id"], {}),
             ("another voice over the word", per_take, "other_voice_over_word_Y_N",
              ["participant_code", "drive_file_id"], {}),
             ("clarity 1-5 (quadratic-weighted kappa)", per_take, "clarity_1_to_5", ["participant_code", "drive_file_id"],
              {"labels": ["1", "2", "3", "4", "5"], "weights": "quadratic"}),
             ("take choice (participants with retakes)", choices, "choice", ["participant_code"], {})]
    rows = []
    listeners = sorted(answers["reviewer_initials"].unique())
    for label, table, col, key, opts in items:
        pair_rows = []
        for a, b in itertools.combinations(listeners, 2):
            x = table[table["reviewer_initials"] == a].set_index(key)[col]
            y = table[table["reviewer_initials"] == b].set_index(key)[col]
            both = pd.concat([x, y], axis=1, join="inner", keys=["a", "b"])
            both = both[(both["a"] != "") & (both["b"] != "")]
            n = len(both)
            pair_rows.append({"judgement": label, "listeners": f"{a} vs {b}", "n_items": n,
                              "percent_agreement": round(100 * float((both["a"] == both["b"]).mean()), 1) if n else np.nan,
                              "cohen_kappa": round(kappa(both["a"], both["b"], **opts), 3) if n else np.nan})
        rows += pair_rows
        if len(pair_rows) > 1:
            rows.append({"judgement": label, "listeners": "mean over pairs (Light's kappa)",
                         "n_items": int(np.mean([r["n_items"] for r in pair_rows])),
                         "percent_agreement": round(float(np.nanmean([r["percent_agreement"] for r in pair_rows])), 1),
                         "cohen_kappa": round(float(np.nanmean([r["cohen_kappa"] for r in pair_rows])), 3)})
    return pd.DataFrame(rows, columns=["judgement", "listeners", "n_items", "percent_agreement", "cohen_kappa"])


def decide(answers: pd.DataFrame, takes: pd.DataFrame, sheet_words: dict[str, str]):
    """Unanimous judgements of >= MIN_LISTENERS listeners -> decision rows; everything else -> an outcome."""
    take_rows, seg_rows, corr_rows, outcomes = [], [], [], []
    protocol = {w.casefold(): w for w in PROTOCOL_WORDS}

    def outcome(code, take, item, text, given):
        outcomes.append({"participant_code": code, "take": take, "judgement": item, "outcome": text, "answers": given})

    def saved_date(who: list[str]) -> str:
        dates = answers.loc[answers["reviewer_initials"].isin(who), "saved_at"].str[:10]
        return max((d for d in dates if d), default="")

    for code, t in takes.groupby("participant_code", sort=True):
        a = answers[answers["participant_code"] == code]
        if a.empty:
            continue
        auto_id = t.loc[t["selected"], "drive_file_id"].iloc[0] if t["selected"].any() else ""
        take_no = dict(zip(t["drive_file_id"], t["review_take"]))

        if len(t) > 1:  # which take holds the whole word most clearly
            choices = {who: chosen_take(g) for who, g in a.groupby("reviewer_initials")}
            answered = {who: c for who, c in choices.items() if c is not None}
            given = "; ".join(f"{who}: {'none' if c == 'none' else f'take {take_no[c]}'}" for who, c in answered.items())
            if len(answered) < MIN_LISTENERS:
                outcome(code, "", "take choice", f"pending: {len(answered)} listener(s) answered", given)
            elif len(set(answered.values())) > 1:
                outcome(code, "", "take choice", "listeners disagree: joint listening decision needed", given)
            elif next(iter(answered.values())) == "none":
                outcome(code, "", "take choice", "every listener heard no complete word in any take: group decision "
                        "needed (nothing changed)", given)
            else:
                fid = next(iter(answered.values()))
                take_rows.append({"participant_code": code, "drive_file_id": fid,
                                  "reason": f"unanimous blind listening: take {take_no[fid]} holds the whole word most "
                                            f"clearly ({len(answered)} listeners)",
                                  "decided_by": "listening review: " + ";".join(sorted(answered))})
                same = "same as" if fid == auto_id else "differs from"
                outcome(code, "", "take choice", f"decided: take {take_no[fid]} ({same} the automatic choice)", given)

        for _, tk in t.iterrows():  # do the automatic boundaries contain the whole word?
            k = int(tk["review_take"])
            rows = a[a["drive_file_id"] == tk["drive_file_id"]]
            stale = int((rows["boundary_stale"] & rows["boundaries_ok_Y_N"].ne("")).sum())
            rows = rows[~rows["boundary_stale"] & rows["boundaries_ok_Y_N"].ne("")]
            given = "; ".join(f"{r.reviewer_initials}: {r.boundaries_ok_Y_N}"
                              + (f" {r.corrected_start_s:.3f}-{r.corrected_end_s:.3f} s"
                                 if pd.notna(r.corrected_start_s) and pd.notna(r.corrected_end_s) else "")
                              for r in rows.itertuples())
            note = f" ({stale} stale answer(s) ignored: boundaries changed since)" if stale else ""
            if len(rows) < MIN_LISTENERS:
                outcome(code, k, "boundaries", f"pending: {len(rows)} listener(s) answered{note}", given)
            elif (rows["boundaries_ok_Y_N"] == "Y").all():
                outcome(code, k, "boundaries", f"confirmed: the automatic boundaries contain the whole word{note}", given)
            elif ((rows["boundaries_ok_Y_N"] == "N").all() and rows["corrected_start_s"].notna().all()
                  and rows["corrected_end_s"].notna().all()
                  and np.ptp(rows["corrected_start_s"]) <= BOUNDARY_TOL_S + 1e-9
                  and np.ptp(rows["corrected_end_s"]) <= BOUNDARY_TOL_S + 1e-9):
                start, end = round(float(rows["corrected_start_s"].mean()), 3), round(float(rows["corrected_end_s"].mean()), 3)
                seg_rows.append({"participant_code": code, "drive_file_id": tk["drive_file_id"], "seg_start_s": start,
                                 "seg_end_s": end, "reviewer": "listening review: " + ";".join(sorted(rows["reviewer_initials"])),
                                 "note": f"take {k}: corrected by {len(rows)} listeners, start spread "
                                         f"{np.ptp(rows['corrected_start_s']):.3f} s, end spread "
                                         f"{np.ptp(rows['corrected_end_s']):.3f} s; mean used"})
                outcome(code, k, "boundaries", f"corrected: {start:.3f}-{end:.3f} s (automatic "
                        f"{tk['auto_seg_start_s']:.3f}-{tk['auto_seg_end_s']:.3f} s){note}", given)
            else:
                outcome(code, k, "boundaries", f"listeners disagree (or a corrected boundary is missing or "
                        f"differs by more than {BOUNDARY_TOL_S * 1000:.0f} ms): joint decision needed{note}", given)

        heard = a[a["word"] != ""]  # which word was said?
        words = sorted(set(heard["word"]))
        n_listeners = heard["reviewer_initials"].nunique()
        given = "; ".join(f"{r.reviewer_initials} take {take_no[r.drive_file_id]}: {r.heard_word}" for r in heard.itertuples())
        sheet = sheet_words.get(code, "").strip()
        if n_listeners < MIN_LISTENERS:
            outcome(code, "", "spoken word", f"pending: {n_listeners} listener(s) wrote a word", given)
        elif len(words) > 1:
            outcome(code, "", "spoken word", "listeners disagree (or the takes differ): joint decision needed", given)
        elif not sheet and words[0] in protocol:
            who = sorted(heard["reviewer_initials"].unique())
            corr_rows.append({"participant_code": code, "field": "spoken_word", "new_value": protocol[words[0]],
                              "source": f"blind listening review, unanimous ({len(who)} listeners)",
                              "verified_by": ";".join(who), "date_verified": saved_date(who)})
            outcome(code, "", "spoken word", f"filled: the sheet has no word; every listener heard '{protocol[words[0]]}'",
                    given)
        elif not sheet:
            outcome(code, "", "spoken word", f"the sheet has no word; listeners heard '{words[0]}', which is not a "
                    "protocol word: left missing, group decision needed", given)
        elif normalise_word(sheet) == words[0]:
            outcome(code, "", "spoken word", "confirmed: listeners heard the word in the sheet", given)
        else:
            outcome(code, "", "spoken word", f"MISMATCH: listeners heard '{words[0]}', the sheet says '{sheet}': "
                    "group to check (nothing changed)", given)
    return (pd.DataFrame(take_rows, columns=TAKE_COLUMNS), pd.DataFrame(seg_rows, columns=SEGMENT_COLUMNS),
            pd.DataFrame(corr_rows, columns=CORRECTION_COLUMNS),
            pd.DataFrame(outcomes, columns=["participant_code", "take", "judgement", "outcome", "answers"]))


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    cells = df.map(lambda v: "—" if pd.isna(v) or v == "" else str(v).replace("|", "\\|"))
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(r) + " |" for r in cells.itertuples(index=False)])


def write_report(answers, takes, agree, takes_out, segs_out, corr_out, outcomes) -> None:
    how = ("How to take part: run the pipeline once so the audio exists, open `data/review/listening_review.html` "
           "in a browser (double-click it), listen with headphones, answer every take, and save the CSV into "
           "`data/review/answers/`. At least two listeners are needed; do it independently and without looking at the "
           "sheet. Then re-run `run_pipeline.py --skip-fetch --from 01b`.")
    lines = ["# Blind listening review", "",
             f"_Generated by `src/01b_merge_listening_reviews.py`; rules in its docstring and in "
             f"`docs/DOCUMENTATION.md` §5.14._", ""]
    if answers.empty:
        lines += ["**No answers have been submitted yet**, so the pipeline still uses only the automatic, label-blind "
                  "rules for takes and word boundaries, and missing spoken words stay missing.", "", how, ""]
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        return
    listeners = (answers.groupby("reviewer_initials")
                 .agg(file=("source_file", "first"), takes_answered=("drive_file_id", "size"),
                      words_written=("word", lambda s: int((s != "").sum())),
                      saved=("saved_at", lambda s: max(s) if len(s) else ""))
                 .reset_index().rename(columns={"reviewer_initials": "listener"}))
    need = outcomes[outcomes["outcome"].str.contains("|".join(NEEDS_DECISION))]
    unanswered = outcomes["outcome"].str.startswith("pending: 0 listener") & ~outcomes["outcome"].str.contains("stale")
    differs = int(sum("differs from" in o for o in outcomes["outcome"]))
    lines += [f"{answers['reviewer_initials'].nunique()} listener(s); {len(takes)} takes of "
              f"{takes['participant_code'].nunique()} participants in the current recordings.", "",
              "## Listeners", "", md_table(listeners), "",
              "## Agreement between listeners", "",
              "Percent agreement and Cohen's kappa per pair of listeners, on the items both answered. Kappa corrects "
              "for agreement expected by chance (0 = chance, 1 = perfect); it is undefined (—) when both listeners "
              "gave the same single answer throughout. When nearly every answer is the same (for example almost "
              "all 'Yes'), kappa can be near 0 despite high percent agreement, so read the two together. Boundary "
              "judgements made on boundaries that have since changed are left out.", "", md_table(agree), "",
              "## Decisions used by the pipeline", "",
              f"- Take choices: **{len(takes_out)}** ({differs} differ from the automatic choice) → "
              f"`{rel(OUT_TAKES)}`",
              f"- Corrected word boundaries: **{len(segs_out)}** → `{rel(OUT_SEGMENTS)}`",
              f"- Missing spoken words filled: **{len(corr_out)}** → `{rel(OUT_CORRECTIONS)}`", "",
              f"A decision needs at least {MIN_LISTENERS} listeners giving the same answer (corrected boundaries within "
              f"{BOUNDARY_TOL_S * 1000:.0f} ms; their mean is used). The decisions apply from the next run of 02 and "
              "03. A row for the same participant (and take) in the group's own `data/processed_audio/take_overrides.csv`, "
              "`data/processed_audio/segment_overrides.csv` or `data/metadata_corrections.csv` wins.", "",
              "## Needs a joint decision", "",
              "Listen together, agree, and record the decision in the group's own override or correction file.", "",
              md_table(need), "",
              "## Every answered item", "",
              f"Items no listener has answered yet ({int(unanswered.sum())}) are left out here; "
              f"`{rel(DECISIONS_CSV)}` lists every item.", "",
              md_table(outcomes[~unanswered]), "", how, ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(ANSWERS_DIR.glob("*.csv"))
    takes = load_takes() if files else pd.DataFrame(columns=["participant_code", "drive_file_id", "review_take"])
    answers = load_answers(files, takes)
    raw = pd.read_csv(RAW_METADATA_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    sheet_words = {normalise_code(c) or c: w for c, w in zip(raw["Participant Code"], raw["Spoken Word"])}
    if answers.empty:
        agree = pd.DataFrame(columns=["judgement", "listeners", "n_items", "percent_agreement", "cohen_kappa"])
        takes_out, segs_out, corr_out = (pd.DataFrame(columns=c) for c in (TAKE_COLUMNS, SEGMENT_COLUMNS, CORRECTION_COLUMNS))
        outcomes = pd.DataFrame(columns=["participant_code", "take", "judgement", "outcome", "answers"])
    else:
        agree = agreement(answers, takes)
        takes_out, segs_out, corr_out, outcomes = decide(answers, takes, sheet_words)
    takes_out.to_csv(OUT_TAKES, index=False, encoding="utf-8")
    segs_out.to_csv(OUT_SEGMENTS, index=False, encoding="utf-8")
    corr_out.to_csv(OUT_CORRECTIONS, index=False, encoding="utf-8")
    agree.to_csv(AGREEMENT_CSV, index=False, encoding="utf-8-sig")
    outcomes.to_csv(DECISIONS_CSV, index=False, encoding="utf-8-sig")
    write_report(answers, takes, agree, takes_out, segs_out, corr_out, outcomes)
    log.info("%d answers file(s), %d listener(s); decisions: %d take choice(s), %d boundary correction(s), "
             "%d word fill(s); report %s", len(files), answers["reviewer_initials"].nunique() if len(answers) else 0,
             len(takes_out), len(segs_out), len(corr_out), rel(REPORT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
