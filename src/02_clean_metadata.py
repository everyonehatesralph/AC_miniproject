"""
02_clean_metadata.py -- PHASE 1: cleaned metadata (raw values always preserved).

Input
    data/metadata_raw/G3_metadata_extracted_raw.csv   verbatim metadata
    data/metadata_corrections.csv                     OPTIONAL, group-verified corrections:
        participant_code, field, new_value, source, verified_by, date_verified
        (template: data/metadata_corrections_TEMPLATE.csv)

Outputs
    data/metadata_cleaned.csv           one row per participant, raw + cleaned columns
    reports/phase1/cleaning_log.csv     every change: participant, field, raw, cleaned, rule
    reports/phase1/cleaning_summary.json

Cleaning rules (nothing else is changed)
    R1  participant code: remove whitespace          ("Y2- 041" -> "Y2-041")
    R2  strip leading/trailing spaces in text fields  ("Kapoy " -> "Kapoy")
    R3  year level "2nd year" -> 2
    R4  class activity mapped to the protocol category it is a part of
        ("Lecture" -> "Lecture/Discussion"); flagged as pending confirmation
    R5  session dash normalised to the protocol spelling ("Afternoon-Evening" -> "Afternoon–Evening")
    R6  ratings "4 — Pleasant" -> 4 after checking that the label matches the number
    R7  a struck-through rating is NOT used: value set to missing, raw text kept, status STRUCK_THROUGH
    R8  consent "Yes" -> "Y"
    R9  group-verified corrections (if supplied) applied with their source logged
Never done: inferring a spoken word from the feeling text, inferring ratings from
text, correcting spellings, or filling blanks.

Usage: .venv/Scripts/python.exe src/02_clean_metadata.py
"""

from __future__ import annotations

import json
import re
import sys

import pandas as pd

from common import (AROUSAL_LABELS, CLEAN_METADATA_CSV, CODE_STRICT_RE, DATA, PROTOCOL_ACTIVITIES, PROTOCOL_SESSIONS,
                    PROTOCOL_WORDS, RAW_METADATA_CSV, REPORTS, SESSION_TOKEN_ASSUMED, VALENCE_LABELS, rel,
                    session_key, setup_logging, sha256_of)

OUT_DIR = REPORTS / "phase1"
CORRECTIONS = DATA / "metadata_corrections.csv"
CORRECTION_TEMPLATE = DATA / "metadata_corrections_TEMPLATE.csv"
CORRECTABLE = {"spoken_word", "self_reported_feeling", "valence", "arousal", "subject",
               "class_activity_protocol", "date_collected", "collector_initials"}
RATING_RE = re.compile(r"^\s*(\d+)\s*(?:[‐-―−-]\s*(.*?))?\s*$")
YEAR_RE = re.compile(r"(?i)^\s*([1-4])\s*(?:st|nd|rd|th)\s*year\s*$")

log = setup_logging("clean", OUT_DIR / "cleaning.log")


def main() -> int:
    raw = pd.read_csv(RAW_METADATA_CSV, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig")
    changes: list[dict] = []
    out_rows: list[dict] = []

    def change(code, field, before, after, rule):
        if before != after:
            changes.append({"participant_code": code, "field": field, "raw_value": before,
                            "cleaned_value": after, "rule": rule})

    for _, r in raw.iterrows():
        struck = {f for f in r.get("_struck_fields", "").split(";") if f}
        code_raw = r["Participant Code"]
        code = re.sub(r"\s+", "", code_raw)
        if not CODE_STRICT_RE.match(code):
            raise ValueError(f"participant code {code_raw!r} cannot be normalised")
        change(code, "participant_code", code_raw, code, "R1 remove whitespace")
        flags, actions = [], []

        m_year = YEAR_RE.match(r["Year Level"])
        if not m_year:
            raise ValueError(f"{code}: unrecognised year level {r['Year Level']!r}")
        year = int(m_year.group(1))
        change(code, "year_level", r["Year Level"], str(year), "R3 year level to integer")

        act_raw = r["Class Activity"]
        act = act_raw.strip()
        exact = [c for c in PROTOCOL_ACTIVITIES if c.casefold() == act.casefold()]
        part = [c for c in PROTOCOL_ACTIVITIES if act.casefold() in {p.strip().casefold() for p in c.split("/")}]
        if exact:
            act_protocol, act_status = exact[0], "EXACT"
        elif len(part) == 1:
            act_protocol, act_status = part[0], "MAPPED_PENDING_CONFIRMATION"
            flags.append("ACTIVITY_MAPPING_PENDING_CONFIRMATION")
            change(code, "class_activity_protocol", act_raw, act_protocol, "R4 map to protocol category (confirm)")
        else:
            act_protocol, act_status = "", "UNRECOGNISED"
            flags.append("ACTIVITY_UNRECOGNISED")

        ses_raw = r["Session"]
        ses = next((s for s in PROTOCOL_SESSIONS if session_key(s) == session_key(ses_raw)), "")
        if not ses:
            raise ValueError(f"{code}: unrecognised session {ses_raw!r}")
        change(code, "session", ses_raw, ses, "R5 protocol spelling")

        def text(field_raw, field_name):
            value = r[field_raw]
            cleaned = value.strip()
            change(code, field_name, value, cleaned, "R2 strip spaces")
            return cleaned

        subject = text("Subject", "subject")
        word = text("Spoken Word", "spoken_word")
        feeling = text("Self-Reported Feeling", "self_reported_feeling")

        ratings = {}
        for key, col, labels in (("valence", "Valence (1-5)", VALENCE_LABELS), ("arousal", "Arousal (1-5)", AROUSAL_LABELS)):
            m = RATING_RE.match(r[col])
            if not m or int(m.group(1)) not in labels:
                raise ValueError(f"{code}: invalid {key} {r[col]!r}")
            value, label = int(m.group(1)), (m.group(2) or "").strip()
            if label and label.casefold() != labels[value].casefold():
                raise ValueError(f"{code}: {key} label {label!r} does not match {value}")
            if col in struck:
                ratings[key], ratings[f"{key}_status"] = None, "STRUCK_THROUGH"
                flags.append(f"{key.upper()}_STRUCK_THROUGH")
                change(code, key, r[col], "", "R7 struck-through rating not used")
            else:
                ratings[key], ratings[f"{key}_status"] = value, "VALID"
                change(code, key, r[col], str(value), "R6 rating to integer (label verified)")

        consent_raw = r["Consent Obtained (Y/N)"]
        consent = {"yes": "Y", "y": "Y", "no": "N", "n": "N"}.get(consent_raw.strip().casefold(), "")
        if not consent:
            raise ValueError(f"{code}: unrecognised consent value {consent_raw!r}")
        change(code, "consent_obtained", consent_raw, consent, "R8 consent to Y/N")

        if not word:
            flags.append("SPOKEN_WORD_MISSING")
        elif word.casefold() not in {w.casefold() for w in PROTOCOL_WORDS}:
            flags.append("WORD_NOT_IN_PROTOCOL_LIST")
        if not feeling:
            flags.append("SELF_REPORT_TEXT_MISSING")
        elif word and word.casefold() == feeling.casefold():
            flags.append("FEELING_IDENTICAL_TO_WORD")
        if not subject:
            flags.append("SUBJECT_MISSING")

        out_rows.append({
            "participant_code": code, "participant_code_raw": code_raw,
            "year_level": year, "year_level_raw": r["Year Level"],
            "class_activity_raw": act_raw, "class_activity_protocol": act_protocol,
            "class_activity_mapping": act_status,
            "subject_raw": r["Subject"], "subject": subject,
            "session_raw": ses_raw, "session": ses,
            "session_token_assumed": SESSION_TOKEN_ASSUMED[session_key(ses)],
            "spoken_word_raw": r["Spoken Word"], "spoken_word": word,
            "spoken_word_source": "metadata" if word else "missing",
            "spoken_word_in_protocol_list": (word.casefold() in {w.casefold() for w in PROTOCOL_WORDS}) if word else None,
            "self_reported_feeling_raw": r["Self-Reported Feeling"], "self_reported_feeling": feeling,
            "valence_raw": r["Valence (1-5)"], "valence": ratings["valence"], "valence_status": ratings["valence_status"],
            "arousal_raw": r["Arousal (1-5)"], "arousal": ratings["arousal"], "arousal_status": ratings["arousal_status"],
            "consent_raw": consent_raw, "consent_obtained": consent,
            "date_collected": r["Date Collected"], "collector_initials": r["Collector Initials"], "notes": r["Notes"],
            "audio_filename_raw": r["Audio Filename"],
            "qa_flags": ";".join(flags), "corrections_applied": "",
        })

    df = pd.DataFrame(out_rows)
    if df["participant_code"].duplicated().any():
        raise ValueError(f"duplicate participant codes: {df.loc[df['participant_code'].duplicated(), 'participant_code'].tolist()}")

    # R9: group-verified corrections, applied with provenance (none exist unless the group adds the file)
    n_corr = 0
    if CORRECTIONS.is_file():
        corr = pd.read_csv(CORRECTIONS, dtype=str, keep_default_na=False)
        need = {"participant_code", "field", "new_value", "source", "verified_by", "date_verified"}
        if not need <= set(corr.columns):
            raise ValueError(f"{rel(CORRECTIONS)} needs columns {sorted(need)}")
        for _, c in corr.iterrows():
            if c["field"] not in CORRECTABLE:
                raise ValueError(f"correction field {c['field']!r} is not correctable; allowed: {sorted(CORRECTABLE)}")
            if not (c["source"].strip() and c["verified_by"].strip()):
                raise ValueError(f"correction for {c['participant_code']} lacks source/verified_by")
            idx = df.index[df["participant_code"] == c["participant_code"]]
            if len(idx) != 1:
                raise ValueError(f"correction for unknown participant {c['participant_code']!r}")
            i = idx[0]
            new = c["new_value"].strip()
            if c["field"] in ("valence", "arousal"):
                if not new.isdigit() or int(new) not in range(1, 6):
                    raise ValueError(f"{c['participant_code']}: {c['field']} correction must be 1-5")
                new = int(new)
                df.at[i, f"{c['field']}_status"] = "CORRECTED_VERIFIED"
            if c["field"] == "spoken_word":
                if not new.isalpha():
                    raise ValueError(f"{c['participant_code']}: spoken word must be letters only")
                df.at[i, "spoken_word_source"] = "verified_correction"
                df.at[i, "spoken_word_in_protocol_list"] = new.casefold() in {w.casefold() for w in PROTOCOL_WORDS}
            before = df.at[i, c["field"]]
            df.at[i, c["field"]] = new
            df.at[i, "corrections_applied"] = ";".join(filter(None, [df.at[i, "corrections_applied"],
                                                                     f"{c['field']} ({c['source']}, {c['verified_by']})"]))
            change(c["participant_code"], c["field"], "" if before is None else str(before), str(new),
                   f"R9 verified correction: {c['source']} / {c['verified_by']} / {c['date_verified']}")
            n_corr += 1
    if not CORRECTION_TEMPLATE.is_file():
        pd.DataFrame(columns=["participant_code", "field", "new_value", "source", "verified_by", "date_verified"]) \
            .to_csv(CORRECTION_TEMPLATE, index=False)

    for col in ("valence", "arousal", "year_level"):
        df[col] = df[col].astype("Int64")
    df.to_csv(CLEAN_METADATA_CSV, index=False, encoding="utf-8-sig")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(changes).to_csv(OUT_DIR / "cleaning_log.csv", index=False, encoding="utf-8-sig")
    summary = {
        "input": rel(RAW_METADATA_CSV), "input_sha256": sha256_of(RAW_METADATA_CSV),
        "output": rel(CLEAN_METADATA_CSV), "rows": int(len(df)),
        "changes_by_rule": pd.Series([c["rule"].split(" ")[0] for c in changes]).value_counts().sort_index().to_dict(),
        "verified_corrections_applied": n_corr,
        "flags": pd.Series([f for fl in df["qa_flags"] for f in fl.split(";") if f]).value_counts().sort_index().to_dict(),
        "valence_counts": df["valence"].value_counts(dropna=False).sort_index().rename(str).to_dict(),
        "arousal_counts": df["arousal"].value_counts(dropna=False).sort_index().rename(str).to_dict(),
    }
    summary = json.loads(json.dumps(summary, default=lambda o: int(o) if hasattr(o, "__int__") else str(o)))
    (OUT_DIR / "cleaning_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("cleaned %d rows; %d logged changes; %d verified corrections", len(df), len(changes), n_corr)
    log.info("flags: %s", summary["flags"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
