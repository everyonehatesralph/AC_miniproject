"""
01_audit_data.py -- PHASE 0: initial data audit (inventory only; no cleaning, no modelling).

Inputs
    data/metadata_raw/G3_metadata_extracted_raw.csv  raw metadata, verbatim (00_extract_metadata_pdf.py)
    data/raw/ (+ any stray media in the project)      supplied recordings
    data/raw/recording_map.csv                       file -> participant code (00_fetch_recordings.py)
    consent/                                         consent documentation, if supplied

Outputs (reports/phase0/)
    participant_qa_table.csv   one row per metadata row
    issues_log.csv             one row per detected issue
    media_inventory.csv        one row per media file
    audit_summary.json         every count used in the Phase 0 report
    audit.log
    ../phase0_data_audit.md    generated Phase 0 report

Rules
    * Read-only: nothing under data/ or consent/ is modified.
    * Raw values are never repaired; normalised candidates are reported next to them.
    * A spoken word is never inferred from the feeling text; ratings are never inferred from text.
    * Phase 0 only flags duplicates (DUPLICATE_RECORDING_REQUIRES_REVIEW). Which take is
      used, and signal-level QA (clipping, silence, speech), are decided in
      03_prepare_audio.py and reported in reports/data_quality_report.md.

Usage: .venv/Scripts/python.exe src/01_audit_data.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import (AROUSAL_LABELS, CODE_IN_NAME_RE, CODE_STRICT_RE, CONSENT_DIR, DASHES, FILENAME_RE,
                    GROUP_YEAR_LEVEL, MEDIA_EXTS, MIN_VALID_OBSERVATIONS, PROJECT_ROOT, PROTOCOL_ACTIVITIES,
                    PROTOCOL_SESSIONS, PROTOCOL_WORDS, RAW_DIR, RAW_METADATA_CSV, RECORDING_MAP, REPORTS,
                    SESSION_TOKEN_ASSUMED, VALENCE_LABELS, probe_media, rel, session_key, setup_logging,
                    sha256_of, sniff_container)

OUT_DIR = REPORTS / "phase0"
COL = {
    "code": "Participant Code", "year": "Year Level", "activity": "Class Activity",
    "subject": "Subject", "session": "Session", "word": "Spoken Word",
    "feeling": "Self-Reported Feeling", "valence": "Valence (1-5)", "arousal": "Arousal (1-5)",
    "audio": "Audio Filename", "consent": "Consent Obtained (Y/N)", "date": "Date Collected",
    "collector": "Collector Initials", "notes": "Notes",
}
SECTION_XI_FIELDS = ("code", "year", "activity", "session", "word", "feeling", "valence", "arousal", "audio")
RATING_RE = re.compile(r"^\s*(\d+)\s*(?:[‐-―−-]\s*(.*?))?\s*$")
YEAR_RE = re.compile(r"(?i)^\s*([1-4])\s*(?:st|nd|rd|th)\s*year\s*$")
SKIP_DIRS = {".venv", ".git", "__pycache__", "node_modules", "processed_audio"}
SEVERITY = {
    "BLOCKER": "observation/deliverable cannot count as valid until resolved",
    "MAJOR": "blocks a required deliverable or analysis step (naming, lexical analysis, protocol compliance)",
    "MINOR": "formatting/coding inconsistency with an obvious normalisation (Phase 1, raw kept)",
    "REVIEW": "needs human confirmation; nothing is changed unless confirmed",
    "INFO": "context for later phases",
}

log = setup_logging("audit", OUT_DIR / "audit.log")


# ---- small parsers ---------------------------------------------------------------
def parse_rating(raw: str, labels: dict[int, str], struck: bool) -> dict:
    text = raw.strip()
    if not text:
        return {"code": None, "status": "MISSING"}
    m = RATING_RE.match(text)
    if not m:
        return {"code": None, "status": "UNPARSEABLE"}
    code, label = int(m.group(1)), (m.group(2) or "").strip()
    if code not in labels:
        return {"code": code, "status": "OUT_OF_RANGE"}
    if label and label.casefold() != labels[code].casefold():
        return {"code": code, "status": "LABEL_MISMATCH"}
    if struck:
        return {"code": code, "status": "STRUCK_THROUGH"}
    return {"code": code, "status": "VALID"}


def match_activity(raw: str) -> tuple[str, str]:
    v = raw.strip()
    if not v:
        return "MISSING", ""
    for cat in PROTOCOL_ACTIVITIES:
        if v.casefold() == cat.casefold():
            return "EXACT", cat
    partial = [c for c in PROTOCOL_ACTIVITIES if v.casefold() in {p.strip().casefold() for p in c.split("/")}]
    return ("PARTIAL", partial[0]) if len(partial) == 1 else ("UNRECOGNISED", "")


def match_session(raw: str) -> tuple[str, str]:
    v = raw.strip()
    if not v:
        return "MISSING", ""
    if v in PROTOCOL_SESSIONS:
        return "EXACT", v
    for s in PROTOCOL_SESSIONS:
        if session_key(v) == session_key(s):
            return "VARIANT", s
    return "UNRECOGNISED", ""


# ---- metadata audit -----------------------------------------------------------------
def load_metadata(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"raw metadata not found: {path}\nRun src/00_extract_metadata_pdf.py first.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig")
    missing = [name for name in COL.values() if name not in df.columns]
    if missing:
        raise ValueError(f"metadata is missing expected columns: {missing}")
    if df.empty:
        raise ValueError("metadata has no participant rows")
    return df


def audit_metadata(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    rows, issues = [], []

    def issue(row, field, code, severity, detail, raw=None):
        issues.append({"scope": "participant", "participant_code_raw": row["code_raw"],
                       "participant_code": row["participant_code"], "field": field, "issue_code": code,
                       "severity": severity, "raw_value": "" if raw is None else raw, "detail": detail})

    for i, rec in df.iterrows():
        struck = {f for f in rec.get("_struck_fields", "").split(";") if f}
        raw = {k: rec[v] for k, v in COL.items()}
        code_candidate = re.sub(r"\s+", "", raw["code"])
        m_code = CODE_STRICT_RE.match(code_candidate)
        row = {"metadata_row": i + 1, "code_raw": raw["code"],
               "participant_code": code_candidate if m_code else raw["code"]}

        if not raw["code"].strip():
            issue(row, COL["code"], "CODE_MISSING", "BLOCKER", "no participant code")
        elif not m_code:
            issue(row, COL["code"], "CODE_INVALID_FORMAT", "BLOCKER",
                  "code does not match Y<year>-<3 digits> even after removing whitespace", raw["code"])
        elif raw["code"] != code_candidate:
            issue(row, COL["code"], "CODE_WHITESPACE", "MINOR",
                  f"whitespace inside/around code; Phase 1 candidate '{code_candidate}'", raw["code"])

        m_year = YEAR_RE.match(raw["year"])
        year = int(m_year.group(1)) if m_year else None
        row["year_level"] = year
        if not raw["year"].strip():
            issue(row, COL["year"], "YEAR_LEVEL_MISSING", "MAJOR", "required by Section XI")
        elif year is None:
            issue(row, COL["year"], "YEAR_LEVEL_UNRECOGNISED", "MAJOR", "not 1st/2nd/3rd/4th year", raw["year"])
        elif year != GROUP_YEAR_LEVEL:
            issue(row, COL["year"], "NOT_ASSIGNED_YEAR_LEVEL", "BLOCKER",
                  f"Group 3 is assigned year {GROUP_YEAR_LEVEL} (Sections IV-V)", raw["year"])
        if m_code and year is not None and int(m_code.group(1)) != year:
            issue(row, COL["code"], "CODE_YEAR_MISMATCH", "MAJOR",
                  f"code prefix Y{m_code.group(1)} disagrees with year level {year}", raw["code"])

        status, cand = match_activity(raw["activity"])
        if status == "MISSING":
            issue(row, COL["activity"], "ACTIVITY_MISSING", "MAJOR", "required by Section XI")
        elif status == "PARTIAL":
            issue(row, COL["activity"], "ACTIVITY_NOT_PROTOCOL_LABEL", "MINOR",
                  f"not an exact Section VII category; Phase 1 candidate '{cand}' (confirm)", raw["activity"])
        elif status == "UNRECOGNISED":
            issue(row, COL["activity"], "ACTIVITY_UNRECOGNISED", "MAJOR", "not a Section VII category", raw["activity"])
        s_status, s_cand = match_session(raw["session"])
        row["session_canonical"] = s_cand
        if s_status == "MISSING":
            issue(row, COL["session"], "SESSION_MISSING", "MAJOR", "required by Section XI")
        elif s_status == "VARIANT":
            issue(row, COL["session"], "SESSION_SPELLING_VARIANT", "MINOR",
                  f"punctuation differs from protocol; Phase 1 candidate '{s_cand}'", raw["session"])
        elif s_status == "UNRECOGNISED":
            issue(row, COL["session"], "SESSION_UNRECOGNISED", "MAJOR",
                  "must be Morning or Afternoon-Evening (Section VII)", raw["session"])

        if not raw["subject"].strip():
            issue(row, COL["subject"], "SUBJECT_MISSING", "REVIEW",
                  "optional template field; confirm the class context for this row")

        word = raw["word"].strip()
        row["spoken_word"] = word
        row["word_in_protocol_list"] = word.casefold() in {w.casefold() for w in PROTOCOL_WORDS} if word else None
        if not word:
            issue(row, COL["word"], "SPOKEN_WORD_MISSING", "MAJOR",
                  "blocks Section XII filename and lexical-control analysis; recoverable only by a "
                  "Bisaya-speaking group member listening to the recording, never from the feeling column")
        elif not row["word_in_protocol_list"]:
            issue(row, COL["word"], "WORD_NOT_IN_PROTOCOL_LIST", "MAJOR",
                  "Section VIII lists 7 words; whether off-list words are acceptable needs an instructor ruling",
                  raw["word"])

        feeling = raw["feeling"].strip()
        row["feeling"] = feeling
        row["feeling_identical_to_word"] = bool(word and feeling and word.casefold() == feeling.casefold())
        if not feeling:
            issue(row, COL["feeling"], "SELF_REPORT_TEXT_MISSING", "MAJOR",
                  "Section IX records the participant's own description; whether a missing description "
                  "invalidates the observation needs an instructor ruling")
        elif row["feeling_identical_to_word"]:
            issue(row, COL["feeling"], "FEELING_IDENTICAL_TO_WORD", "REVIEW",
                  "may be genuine; confirm the description was given independently and not copied "
                  "from the Spoken Word column", raw["feeling"])

        for key, labels in (("valence", VALENCE_LABELS), ("arousal", AROUSAL_LABELS)):
            parsed = parse_rating(raw[key], labels, COL[key] in struck)
            row[key] = parsed["code"]
            row[f"{key}_status"] = parsed["status"]
            if parsed["status"] == "STRUCK_THROUGH":
                issue(row, COL[key], f"{key.upper()}_STRUCK_THROUGH", "BLOCKER",
                      "cell text is formatted with strikethrough and no replacement value is present; "
                      "unclear whether the rating was retracted. Collector must confirm from the original record",
                      raw[key])
            elif parsed["status"] != "VALID":
                issue(row, COL[key], f"{key.upper()}_{parsed['status']}", "BLOCKER",
                      "Section IX requires a 1-5 rating with matching label", raw[key])

        audio = raw["audio"].strip()
        row["audio_filename_raw"] = raw["audio"]
        if not audio:
            issue(row, COL["audio"], "AUDIO_FILENAME_MISSING", "MAJOR",
                  "Section XI requires the corresponding WAV filename (assigned in Phase 4)")
        else:
            m_fn = FILENAME_RE.match(audio)
            if not m_fn:
                issue(row, COL["audio"], "FILENAME_CONVENTION_VIOLATION", "MAJOR",
                      "expected ParticipantCode_YearLevel_Session_Word.wav (Section XII)", raw["audio"])
            elif m_fn.group(1) != row["participant_code"] or (word and m_fn.group(4).casefold() != word.casefold()):
                issue(row, COL["audio"], "FILENAME_METADATA_MISMATCH", "MAJOR",
                      "code or word in the filename disagrees with the metadata", raw["audio"])

        consent = raw["consent"].strip()
        row["consent_recorded"] = consent.casefold() in {"y", "yes"}
        if not consent:
            issue(row, COL["consent"], "CONSENT_NOT_RECORDED", "BLOCKER", "no consent value recorded")
        elif consent.casefold() in {"n", "no"}:
            issue(row, COL["consent"], "CONSENT_NO", "BLOCKER",
                  "Section V: responses without consent must not be collected or used", raw["consent"])
        elif consent not in {"Y", "N"}:
            if row["consent_recorded"]:
                issue(row, COL["consent"], "CONSENT_CODING_NOT_Y_N", "MINOR",
                      "header asks for Y/N; Phase 1 candidate 'Y'", raw["consent"])
            else:
                issue(row, COL["consent"], "CONSENT_VALUE_UNRECOGNISED", "BLOCKER",
                      "consent value is neither yes nor no", raw["consent"])

        if not raw["date"].strip():
            issue(row, COL["date"], "DATE_COLLECTED_MISSING", "REVIEW",
                  "needed to verify session classification and collection timing")
        if not raw["collector"].strip():
            issue(row, COL["collector"], "COLLECTOR_INITIALS_MISSING", "REVIEW",
                  "needed to route data queries to the person who collected this observation")

        for key in ("subject", "word", "feeling", "valence", "arousal", "consent"):
            if raw[key].strip() and raw[key] != raw[key].strip():
                issue(row, COL[key], "EDGE_WHITESPACE", "MINOR",
                      "leading/trailing space; Phase 1 strips it and keeps the raw value", raw[key])
        for f in struck - {COL["valence"], COL["arousal"]}:
            issue(row, f, "STRUCK_THROUGH_TEXT", "REVIEW", "strikethrough formatting in the sheet", rec[f])
        for f in filter(None, rec.get("_nonblack_text_fields", "").split(";")):
            issue(row, f, "COLOURED_TEXT", "REVIEW", "non-black text in the sheet", rec[f])
        for f in filter(None, rec.get("_filled_fields", "").split(";")):
            issue(row, f.split(":")[0], "CELL_FILL", "REVIEW", f"coloured cell fill {f}")
        rows.append(row)

    counts = Counter(r["participant_code"] for r in rows)
    for r in rows:
        if counts[r["participant_code"]] > 1:
            issue(r, COL["code"], "DUPLICATE_REQUIRES_REVIEW", "BLOCKER",
                  f"code appears {counts[r['participant_code']]} times in the metadata", r["code_raw"])
    return rows, issues


# ---- media and consent inventory ---------------------------------------------------
def find_media(root: Path) -> list[Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        found += [Path(dirpath) / n for n in filenames if Path(n).suffix.lower() in MEDIA_EXTS]
    return sorted(found)


def load_recording_map() -> dict[str, str]:
    if not RECORDING_MAP.is_file():
        return {}
    df = pd.read_csv(RECORDING_MAP, dtype=str, keep_default_na=False)
    if not {"original_file", "participant_code"} <= set(df.columns):
        raise ValueError(f"{rel(RECORDING_MAP)} must have columns original_file, participant_code")
    out = {}
    for _, r in df.iterrows():
        code = re.sub(r"\s+", "", r["participant_code"])
        if not CODE_STRICT_RE.match(code):
            raise ValueError(f"{rel(RECORDING_MAP)}: invalid participant_code {r['participant_code']!r}")
        out[Path(r["original_file"]).as_posix()] = code
    return out


def inventory_media(known_codes: set[str]) -> tuple[pd.DataFrame, list[dict]]:
    mapping = load_recording_map()
    drive_names = (pd.read_csv(RECORDING_MAP, dtype=str, keep_default_na=False)
                   .set_index("original_file")["drive_name"].to_dict()
                   if RECORDING_MAP.is_file() and "drive_name" in pd.read_csv(RECORDING_MAP, nrows=0).columns else {})
    records, issues = [], []
    for p in find_media(PROJECT_ROOT):
        relpath = rel(p)
        under_raw = RAW_DIR in p.resolve().parents
        map_key = p.resolve().relative_to(RAW_DIR).as_posix() if under_raw else None
        stem_codes = sorted({f"Y{a}-{b}" for a, b in CODE_IN_NAME_RE.findall(p.name.split("__")[0])})
        if map_key and map_key in mapping:
            code, source = mapping[map_key], "recording_map.csv"
        elif len(stem_codes) == 1:
            code, source = stem_codes[0], "filename"
        else:
            code, source = "", "ambiguous" if stem_codes else "none"
        container = sniff_container(p)
        try:
            probe, probe_error = probe_media(p), ""
        except RuntimeError as exc:  # recorded as a finding
            probe, probe_error = {}, str(exc)
        rec = {
            "file": relpath, "drive_name": drive_names.get(map_key, ""), "under_data_raw": under_raw,
            "extension": p.suffix.lower(), "size_bytes": p.stat().st_size, "sha256": sha256_of(p),
            "container": container, "participant_code": code, "match_source": source,
            "in_metadata": code in known_codes if code else False,
            "follows_naming_convention": bool(FILENAME_RE.match(p.name)),
            "readable": bool(probe) and probe.get("has_audio", False),
            "duration_s": round(probe["duration_s"], 3) if probe else None,
            "audio_codec": probe.get("audio_codec", ""), "sample_rate": probe.get("sample_rate"),
            "channels": probe.get("channel_layout", ""), "n_audio_streams": probe.get("n_audio_streams"),
            "has_video": probe.get("has_video"), "video_codec": probe.get("video_codec", ""),
            "capture_time": probe.get("com.apple.quicktime.creationdate", probe.get("creation_time", "")),
            "device_model": probe.get("com.apple.quicktime.model", ""), "probe_error": probe_error,
        }
        records.append(rec)

        def media_issue(name, severity, detail):
            issues.append({"scope": "media", "participant_code_raw": "", "participant_code": code,
                           "field": relpath, "issue_code": name, "severity": severity,
                           "raw_value": rec["drive_name"] or p.name, "detail": detail})

        if not under_raw:
            media_issue("MEDIA_OUTSIDE_DATA_RAW", "MINOR", "move a copy into data/raw/ (originals stay untouched)")
        if source == "none":
            media_issue("MEDIA_NO_PARTICIPANT_CODE", "MAJOR", "no code in filename and no recording_map entry")
        elif source == "ambiguous":
            media_issue("MEDIA_AMBIGUOUS_CODE", "MAJOR", f"several codes in filename: {stem_codes}")
        elif not rec["in_metadata"]:
            media_issue("MEDIA_WITHOUT_METADATA", "MAJOR", f"{code} has no metadata row")
        if not rec["readable"]:
            media_issue("MEDIA_UNREADABLE", "BLOCKER", probe_error or "no audio stream")
        if container != "WAV":
            media_issue("FORMAT_NOT_WAV", "MINOR", f"{container}; Phase 3 derives WAV, original kept")
        if rec["drive_name"] and not FILENAME_RE.match(rec["drive_name"] + ".wav"):
            media_issue("RAW_NAME_NOT_CONVENTION", "INFO",
                        f"raw name '{rec['drive_name']}' is not ParticipantCode_YearLevel_Session_Word; "
                        "Phase 4 assigns the convention name to the processed WAV")
    media = pd.DataFrame.from_records(records)
    if not media.empty:
        for _, grp in media.groupby("sha256"):
            if len(grp) > 1:
                for f in grp["file"]:
                    issues.append({"scope": "media", "participant_code_raw": "", "participant_code": "",
                                   "field": f, "issue_code": "IDENTICAL_FILE_CONTENT", "severity": "REVIEW",
                                   "raw_value": "", "detail": f"byte-identical to {len(grp) - 1} other file(s)"})
    return media, issues


def inventory_consent(known_codes: set[str]) -> dict:
    """Count consent documents without printing their names (they may contain names)."""
    files = [p for p in CONSENT_DIR.rglob("*") if p.is_file() and not p.name.startswith(".")
             and p.name.lower() not in {"readme.md", "readme.txt"}] if CONSENT_DIR.is_dir() else []
    codes = set()
    for p in files:
        codes |= {f"Y{a}-{b}" for a, b in CODE_IN_NAME_RE.findall(p.name)}
    return {"consent_dir": rel(CONSENT_DIR), "n_files": len(files),
            "extensions": dict(Counter(p.suffix.lower() for p in files)),
            "participant_codes_in_filenames": sorted(codes & known_codes)}


# ---- QA table ------------------------------------------------------------------------
def recordings_for(code: str, media: pd.DataFrame) -> pd.DataFrame:
    return media[media["participant_code"] == code] if not media.empty else media


def recording_issues(rows: list[dict], media: pd.DataFrame) -> list[dict]:
    out = []
    for r in rows:
        recs = recordings_for(r["participant_code"], media)
        n = len(recs)
        if n == 0:
            code, detail = "NO_RECORDING", "no media file matched to this participant code"
        elif n > 1:
            names = ", ".join(f"{d or f.rsplit('/', 1)[-1]} ({t[11:19] if t else 'no capture time'})"
                              for d, f, t in zip(recs["drive_name"], recs["file"], recs["capture_time"]))
            code, detail = "DUPLICATE_RECORDING_REQUIRES_REVIEW", (
                f"{n} files: {names}; resolved (or not) in Phase 2 by 03_prepare_audio.py")
        elif not recs.iloc[0]["readable"]:
            code, detail = "RECORDING_UNREADABLE", recs.iloc[0]["probe_error"] or "no audio stream"
        else:
            code, detail = "AUDIO_QA_PENDING", "signal-level QA runs in 03_prepare_audio.py"
        out.append({"scope": "participant", "participant_code_raw": r["code_raw"],
                    "participant_code": r["participant_code"], "field": "recording",
                    "issue_code": code, "severity": "BLOCKER", "raw_value": "", "detail": detail})
    return out


def build_qa_table(rows: list[dict], issues: list[dict], media: pd.DataFrame) -> pd.DataFrame:
    by_code: dict[str, list[dict]] = {}
    for it in issues:
        if it["scope"] == "participant":
            by_code.setdefault(it["participant_code"], []).append(it)
    qa = []
    for r in rows:
        code = r["participant_code"]
        recs = recordings_for(code, media)
        n = len(recs)
        mine = sorted(by_code.get(code, []), key=lambda it: it["field"] != "recording")
        blockers = [it["issue_code"] for it in mine if it["severity"] == "BLOCKER"]
        flags = [it["issue_code"] for it in mine if it["severity"] != "BLOCKER"]
        if n == 1 and recs.iloc[0]["readable"]:
            m = recs.iloc[0]
            audio = {"audio_readable": True, "audio_duration": m["duration_s"], "sample_rate": m["sample_rate"],
                     "channels": m["channels"]}
        else:
            state = "NA_NO_RECORDING" if n == 0 else ("DUPLICATE_REQUIRES_REVIEW" if n > 1 else "UNREADABLE")
            audio = {"audio_readable": state if n != 1 else False, "audio_duration": state,
                     "sample_rate": state, "channels": state}
        token = SESSION_TOKEN_ASSUMED.get(session_key(r["session_canonical"])) if r["session_canonical"] else None
        can_name = CODE_STRICT_RE.match(code) and r["year_level"] and token and r["spoken_word"]
        qa.append({
            "participant_code": code, "participant_code_raw": r["code_raw"], "metadata_row": r["metadata_row"],
            "metadata_present": True, "recording_present": n > 0, "n_recordings": n,
            "recording_files": ";".join(recs["drive_name"].where(recs["drive_name"] != "", recs["file"])) if n else "",
            "duplicate_recording": n > 1, "consent_recorded": r["consent_recorded"],
            "spoken_word_present": bool(r["spoken_word"]), "spoken_word": r["spoken_word"],
            "spoken_word_in_protocol_list": r["word_in_protocol_list"],
            "self_report_present": bool(r["feeling"]), "self_reported_feeling": r["feeling"],
            "feeling_identical_to_word": r["feeling_identical_to_word"],
            "valence": r["valence"], "valence_valid": r["valence_status"] == "VALID", "valence_status": r["valence_status"],
            "arousal": r["arousal"], "arousal_valid": r["arousal_status"] == "VALID", "arousal_status": r["arousal_status"],
            **audio, "clipping_detected": "PHASE_3", "silence_problem": "PHASE_3",
            "filename_valid": bool(r["audio_filename_raw"].strip()) and FILENAME_RE.match(r["audio_filename_raw"].strip()) is not None,
            "proposed_filename": f"{code}_Y{r['year_level']}_{token}_{r['spoken_word']}.wav" if can_name else "",
            "include_in_analysis": not blockers,
            "exclusion_reason": ";".join(dict.fromkeys(blockers)),
            "review_flags": ";".join(dict.fromkeys(flags)),
        })
    return pd.DataFrame(qa)


def counts(values) -> dict:
    return {str(k): int(v) for k, v in sorted(Counter(values).items(), key=lambda kv: str(kv[0]))}


def dataset_issues(df, media, consent, provenance_present) -> list[dict]:
    out = []

    def add(code, severity, detail):
        out.append({"scope": "dataset", "participant_code_raw": "", "participant_code": "", "field": "",
                    "issue_code": code, "severity": severity, "raw_value": "", "detail": detail})

    if media.empty:
        add("NO_MEDIA_FILES_SUPPLIED", "BLOCKER", "no audio/video file found; no observation can be validated")
    if consent["n_files"] == 0:
        add("CONSENT_DOCUMENTATION_NOT_SUPPLIED", "BLOCKER",
            "Section XVIII item 2 requires consent documentation; the sheet's 'Yes' is an attestation, not documentation")
    covered = media["participant_code"].nunique() if not media.empty else 0
    if covered < MIN_VALID_OBSERVATIONS:
        add("FEWER_RECORDED_PARTICIPANTS_THAN_MINIMUM", "BLOCKER",
            f"recordings exist for {covered} participant codes; the protocol minimum is {MIN_VALID_OBSERVATIONS}")
    for key in ("year", "activity", "session"):
        vals = {v.strip() for v in df[COL[key]] if v.strip()}
        if len(vals) == 1:
            add(f"CONSTANT_{key.upper()}", "INFO",
                f"every row has {COL[key]} = '{vals.pop()}'; {COL[key]} x Affect cannot be analysed within Group 3")
    add("SESSION_TOKEN_ASSUMED", "REVIEW",
        "Section XII shows 'PM' in its example but never defines session tokens; AM/PM mapping needs confirmation")
    if provenance_present:
        add("METADATA_SOURCE_IS_PDF_EXPORT", "REVIEW",
            "only a PDF export of the Google Sheet was supplied; cell comments and edit history are lost")
    return out


def summarize(df, rows, qa, issues, media, consent) -> dict:
    order = [r["participant_code"] for r in rows]
    missing_fields = {}
    for key, name in COL.items():
        blank = [r["participant_code"] for r, (_, rec) in zip(rows, df.iterrows()) if not rec[name].strip()]
        missing_fields[name] = {"n_missing": len(blank), "codes": blank, "required_by_section_XI": key in SECTION_XI_FIELDS}
    media_codes = set(media["participant_code"]) if not media.empty else set()
    dup = qa.loc[qa["duplicate_recording"], ["participant_code", "n_recordings"]]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {"metadata_csv": rel(RAW_METADATA_CSV), "metadata_sha256": sha256_of(RAW_METADATA_CSV)},
        "tool_availability": {"ffmpeg": True, "soundfile": importlib.util.find_spec("soundfile") is not None,
                              "librosa": importlib.util.find_spec("librosa") is not None,
                              "parselmouth": importlib.util.find_spec("parselmouth") is not None},
        "severity_scale": SEVERITY,
        "A_total_metadata_rows": int(len(df)),
        "B_total_media_files": int(len(media)),
        "C_unique_participant_codes": {"metadata": len(set(order)), "with_recordings": len(media_codes),
                                       "recordings_and_metadata": len(media_codes & set(order))},
        "D_metadata_codes_missing_recordings": [c for c in order if c not in media_codes],
        "E_recordings_without_matching_metadata": media.loc[~media["in_metadata"], "file"].tolist() if not media.empty else [],
        "F_duplicate_participant_recordings": {c: int(n) for c, n in zip(dup["participant_code"], dup["n_recordings"])},
        "G_missing_metadata_fields": missing_fields,
        "H_invalid_or_out_of_range_values": [
            {"participant_code": r["participant_code"], "field": COL[k], "status": r[f"{k}_status"],
             "raw": df.iloc[r["metadata_row"] - 1][COL[k]]}
            for r in rows for k in ("valence", "arousal") if r[f"{k}_status"] != "VALID"],
        "I_naming": {"metadata_rows_without_audio_filename": int((~qa["filename_valid"]).sum()),
                     "raw_files_not_following_convention": int((~media["follows_naming_convention"]).sum()) if not media.empty else 0,
                     "rows_with_proposed_filename": int((qa["proposed_filename"] != "").sum())},
        "J_formats": counts(media["container"]) if not media.empty else {},
        "J_audio_streams": counts(media["audio_codec"] + " " + media["sample_rate"].astype(str) + " Hz " + media["channels"])
        if not media.empty else {},
        "K_consent": {"metadata_values": counts(df[COL["consent"]]), "documentation": consent},
        "L_audio_quality": "assessed in 03_prepare_audio.py (reports/data_quality_report.md)",
        "M_eligible_after_phase0": qa.loc[qa["include_in_analysis"], "participant_code"].tolist(),
        "N_blocked_at_phase0": int((~qa["include_in_analysis"]).sum()),
        "O_phase0_blockers": dict(zip(qa["participant_code"], qa["exclusion_reason"])),
        "recording_status_counts": counts(qa["exclusion_reason"].str.split(";").str[0]),
        "issue_counts_by_code": counts(i["issue_code"] for i in issues),
        "issue_counts_by_severity": counts(i["severity"] for i in issues),
    }


# ---- markdown report ------------------------------------------------------------------
def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False))


def write_report(summary: dict, qa: pd.DataFrame, media: pd.DataFrame, issues: pd.DataFrame) -> None:
    s = summary
    dup = s["F_duplicate_participant_recordings"]
    lines = [
        "# Phase 0: Initial Data Audit (Group 3)",
        "",
        f"*Generated by `src/01_audit_data.py` on {s['generated_at_utc']} (UTC). Every number below is computed "
        "from the files; nothing is typed by hand. Inventory only: which duplicate take is used, signal-level "
        "audio QA and exclusions are decided in Phases 2–5 (`reports/data_quality_report.md`).*",
        "",
        "Earlier, metadata-only version: `reports/phase0_data_audit_v1_metadata_only.md` (written before the "
        "recordings were supplied).",
        "",
        "## A–O summary",
        "",
        "| Item | Result |", "|---|---|",
        f"| A. Total metadata rows | {s['A_total_metadata_rows']} |",
        f"| B. Total media files | {s['B_total_media_files']} |",
        f"| C. Unique participant codes | metadata {s['C_unique_participant_codes']['metadata']}; with recordings "
        f"{s['C_unique_participant_codes']['with_recordings']} |",
        f"| D. Metadata codes without any recording | {len(s['D_metadata_codes_missing_recordings'])}: "
        f"{', '.join(s['D_metadata_codes_missing_recordings']) or '—'} |",
        f"| E. Recordings without matching metadata | {len(s['E_recordings_without_matching_metadata'])} |",
        f"| F. Participants with more than one recording | {len(dup)}: "
        f"{', '.join(f'{c} ({n} files)' for c, n in dup.items()) or '—'} |",
        f"| G. Missing metadata fields | see table below |",
        f"| H. Invalid / out-of-range values | {len(s['H_invalid_or_out_of_range_values'])}: "
        f"{'; '.join(h['participant_code'] + ' ' + h['field'] + ' ' + h['status'] for h in s['H_invalid_or_out_of_range_values']) or '—'} |",
        f"| I. Naming | Audio Filename blank in {s['I_naming']['metadata_rows_without_audio_filename']} rows; "
        f"{s['I_naming']['raw_files_not_following_convention']} raw files not named by convention (expected for raw files) |",
        f"| J. Formats | {', '.join(f'{k}: {v}' for k, v in s['J_formats'].items())}; audio streams: "
        f"{', '.join(f'{k}: {v}' for k, v in s['J_audio_streams'].items())} |",
        f"| K. Consent | sheet values {s['K_consent']['metadata_values']}; documentation files supplied: "
        f"{s['K_consent']['documentation']['n_files']} |",
        f"| L. Audio quality | {s['L_audio_quality']} |",
        f"| M. Eligible after Phase 0 | {len(s['M_eligible_after_phase0'])} (no participant can be validated before audio QA) |",
        f"| N. Blocked at Phase 0 | {s['N_blocked_at_phase0']} |",
        f"| O. Reasons | per participant in the table below |",
        "",
        "## Missing metadata fields (G)",
        "",
        md_table(pd.DataFrame([{"Field": f, "Required by §XI": "yes" if v["required_by_section_XI"] else "no",
                                "Missing": v["n_missing"], "Codes": ", ".join(v["codes"]) if 0 < v["n_missing"] < 40 else
                                ("all" if v["n_missing"] == 40 else "—")}
                               for f, v in s["G_missing_metadata_fields"].items()])),
        "",
        "## Media inventory",
        "",
        md_table(media[["drive_name", "participant_code", "container", "duration_s", "audio_codec", "sample_rate",
                        "channels", "n_audio_streams", "capture_time"]].rename(columns={"drive_name": "Drive name"})),
        "",
        "## Participant QA table (Phase 0)",
        "",
        md_table(qa[["participant_code", "spoken_word", "self_reported_feeling", "valence", "arousal", "n_recordings",
                     "recording_files", "exclusion_reason"]].replace({"": "—"})),
        "",
        "## Issue counts",
        "",
        md_table(pd.DataFrame([{"Severity": k, "Count": v} for k, v in s["issue_counts_by_severity"].items()])),
        "",
        md_table(pd.DataFrame([{"Issue": k, "Count": v} for k, v in s["issue_counts_by_code"].items()])),
        "",
        "Full detail: `reports/phase0/issues_log.csv`, `reports/phase0/participant_qa_table.csv`, "
        "`reports/phase0/media_inventory.csv`, `reports/phase0/audit_summary.json`.",
        "",
    ]
    (REPORTS / "phase0_data_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    df = load_metadata(RAW_METADATA_CSV)
    log.info("metadata: %d rows from %s", len(df), rel(RAW_METADATA_CSV))
    rows, issues = audit_metadata(df)
    known = {r["participant_code"] for r in rows}
    media, media_issues = inventory_media(known)
    log.info("media files found: %d (%d participant codes)", len(media),
             media["participant_code"].nunique() if not media.empty else 0)
    consent = inventory_consent(known)
    log.info("consent documents found: %d", consent["n_files"])
    issues = recording_issues(rows, media) + issues
    qa = build_qa_table(rows, issues, media)
    provenance = RAW_METADATA_CSV.with_name(RAW_METADATA_CSV.stem + "_provenance.json").is_file()
    all_issues = dataset_issues(df, media, consent, provenance) + issues + media_issues
    summary = summarize(df, rows, qa, all_issues, media, consent)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qa.to_csv(OUT_DIR / "participant_qa_table.csv", index=False, encoding="utf-8-sig")
    issues_df = pd.DataFrame(all_issues)
    issues_df.to_csv(OUT_DIR / "issues_log.csv", index=False, encoding="utf-8-sig")
    media.to_csv(OUT_DIR / "media_inventory.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, qa, media, issues_df)
    log.info("recording status: %s", summary["recording_status_counts"])
    log.info("issues: %s", summary["issue_counts_by_severity"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
