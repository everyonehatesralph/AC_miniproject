"""
00b_compare_sheet_versions.py -- a newer sheet that is not analysed yet: every cell that differs.

The group may complete or correct its sheet after an analysis. A newer export is never put over the sheet in
use. It goes into data/metadata_raw/received/, is extracted with step 00,
    python src/00_extract_metadata_pdf.py --pdf "data/metadata_raw/received/<file>.pdf"
        --out data/metadata_raw/received/G3_metadata_extracted_raw_<date>.csv
and gets a row in data/metadata_raw/received/sheet_versions.csv saying when it arrived and, in the group's own
words, where its changes come from.

This script compares the newest received extraction with the sheet in use, cell by cell, and writes
reports/sheet_update_pending.md and .json. Nothing is applied: the analysis keeps using the sheet in use until
the received sheet replaces it and the pipeline is run again. When the two agree in every data cell, the pending
report is removed, so the report exists exactly as long as a newer sheet is waiting.

Only the standard library is used:
    python src/00b_compare_sheet_versions.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "metadata_raw"
IN_USE = RAW / "G3_metadata_extracted_raw.csv"
RECEIVED = RAW / "received"
VERSIONS = RECEIVED / "sheet_versions.csv"
OUT_MD = ROOT / "reports" / "sheet_update_pending.md"
OUT_JSON = OUT_MD.with_suffix(".json")
AUDIT = ROOT / "reports" / "supplied_audio_audit.csv"
WORD, FEELING, SUBJECT, STRUCK = "Spoken Word", "Self-Reported Feeling", "Subject", "_struck_fields"
RATINGS = ("Valence (1-5)", "Arousal (1-5)")


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def by_code(rows: list[dict]) -> dict[str, dict]:
    return {r["Participant Code"].replace(" ", ""): r for r in rows}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_pdf(extraction: Path) -> str:
    """The PDF an extraction was made from, as step 00 recorded it."""
    prov = extraction.with_name(extraction.stem + "_provenance.json")
    return json.loads(prov.read_text(encoding="utf-8")).get("source_pdf", "") if prov.is_file() else ""


def compare(old: dict[str, dict], new: dict[str, dict]) -> list[dict]:
    """Every differing data cell. Extraction bookkeeping (page, row, wrapping) is not data and is left out."""
    columns = [c for c in next(iter(new.values())) if not c.startswith("_")] + [STRUCK]
    changes = []
    for code in sorted(set(old) | set(new)):
        for col in columns:
            a = old.get(code, {}).get(col, "")
            b = new.get(code, {}).get(col, "")
            if a != b:
                changes.append({"participant_code": code, "column": col, "in_use": a, "received": b})
    return changes


def main() -> int:
    extractions = sorted(RECEIVED.glob("G3_metadata_extracted_raw_*.csv")) if RECEIVED.is_dir() else []
    if not extractions:
        for stale in (OUT_MD, OUT_JSON):
            stale.unlink(missing_ok=True)
        print("sheet versions: no received sheet")
        return 0
    newest = extractions[-1]
    old, new = by_code(read(IN_USE)), by_code(read(newest))
    changes = compare(old, new)
    if not changes:
        for stale in (OUT_MD, OUT_JSON):
            stale.unlink(missing_ok=True)
        print(f"sheet versions: {newest.name} is the sheet in use; nothing is pending")
        return 0

    said = next((r for r in (read(VERSIONS) if VERSIONS.is_file() else []) if r["extraction"] == newest.name), {})
    pick = lambda col, test: sorted(c["participant_code"] for c in changes if c["column"] == col and test(c))  # noqa: E731
    rating_codes = sorted({c["participant_code"] for c in changes if c["column"] in RATINGS})
    kapoy = lambda rows: sum(r[WORD].strip().lower() == "kapoy" for r in rows.values())  # noqa: E731
    audit = {r["participant_code"]: r.get("file_word", "") for r in read(AUDIT)} if AUDIT.is_file() else {}
    disagree = sorted(c for c in new if audit.get(c) and audit[c].strip().lower() != new[c][WORD].strip().lower())
    summary = {
        "received_extraction": newest.relative_to(ROOT).as_posix(), "received_pdf": source_pdf(newest),
        "received_pdf_sha256": sha256_of(ROOT / source_pdf(newest)) if source_pdf(newest) else "",
        "received_on": said.get("received_on", ""), "in_use_extraction": IN_USE.relative_to(ROOT).as_posix(),
        "in_use_pdf": source_pdf(IN_USE), "changed_cells": len(changes),
        "participants_changed": sorted({c["participant_code"] for c in changes}),
        "words_filled": pick(WORD, lambda c: not c["in_use"].strip() and c["received"].strip()),
        "words_changed": pick(WORD, lambda c: c["in_use"].strip() and c["received"].strip()),
        "ratings_changed": rating_codes,
        "feelings_filled": pick(FEELING, lambda c: not c["in_use"].strip()),
        "subjects_filled": pick(SUBJECT, lambda c: not c["in_use"].strip()),
        "strikethrough_removed": pick(STRUCK, lambda c: c["in_use"].strip() and not c["received"].strip()),
        "blank_words_in_use": sum(not r[WORD].strip() for r in old.values()),
        "blank_words_received": sum(not r[WORD].strip() for r in new.values()),
        "kapoy_in_use": kapoy(old), "kapoy_received": kapoy(new),
        "word_differs_from_audio_file_name": disagree,
        "ratings_source": said.get("ratings_source", ""), "words_source": said.get("words_source", ""),
        "reported_on": said.get("reported_on", ""),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cell = lambda v: (v if v.strip() else "(blank)").replace("|", "/")  # noqa: E731
    L = ["# A newer sheet is waiting: what differs", "",
         "*Generated by `src/00b_compare_sheet_versions.py`. Nothing in this report has been applied: every result in "
         "`reports/` still comes from the sheet in use.*", "",
         "## The two sheets", "",
         "| | File | Extracted to |", "|---|---|---|",
         f"| In use (analysed) | `{summary['in_use_pdf']}` | `{summary['in_use_extraction']}` |",
         f"| Received {summary['received_on']} (not analysed) | `{summary['received_pdf']}` | `{summary['received_extraction']}` |",
         "", f"SHA-256 of the received PDF: `{summary['received_pdf_sha256']}`.", "",
         "## Summary", "",
         f"- {len(changes)} cells differ, for {len(summary['participants_changed'])} participants.",
         f"- **Spoken words:** {len(summary['words_filled'])} blank words are filled in "
         f"({', '.join(summary['words_filled']) or 'none'}); {len(summary['words_changed'])} existing "
         f"word{' is' if len(summary['words_changed']) == 1 else 's are'} changed "
         f"({', '.join(summary['words_changed']) or 'none'}). Blank words: {summary['blank_words_in_use']} in the sheet in "
         f"use, {summary['blank_words_received']} in the received sheet.",
         f"- **Ratings:** changed for {len(rating_codes)} participants ({', '.join(rating_codes) or 'none'}). A rating is "
         "the label the models learn from, so these changes alter the analysis itself.",
         f"- **Strikethrough:** removed for {', '.join(summary['strikethrough_removed']) or 'no participant'}.",
         f"- **Feeling text:** filled in for {', '.join(summary['feelings_filled']) or 'no participant'}.",
         f"- **Subject:** filled in for {len(summary['subjects_filled'])} participants.",
         f"- **Kapoy speakers:** {summary['kapoy_in_use']} in the sheet in use, {summary['kapoy_received']} in the received "
         "sheet. The word analysis will change.", "",
         "## Where the changes come from, as the group reports it", ""]
    if said:
        L += [f"Reported by the group on {said.get('reported_on', '')}:", "",
              f"- **Ratings:** corrected {said.get('ratings_source', '')}.",
              f"- **Words:** filled in {said.get('words_source', '')}.", "",
              "The pipeline cannot check these statements. The answer forms of the participants whose ratings changed "
              f"({', '.join(rating_codes) or 'none'}) are the evidence for the corrected ratings and must be kept.", ""]
    else:
        L += ["**Not stated yet.** Add a row for this sheet to `data/metadata_raw/received/sheet_versions.csv` before it "
              "is used: a rating may be changed only to correct a copying mistake against the participant's own "
              "answer, and a word may not be taken from the feeling text.", ""]
    if disagree:
        L += ["## Still open", "",
              f"For {len(disagree)} participants the word in the received sheet is not the word in the name of the audio "
              f"file ({', '.join(disagree)}). The listening review should settle these.", ""]
    L += ["## Every changed cell", "", "| Participant | Column | Sheet in use | Received sheet |", "|---|---|---|---|"]
    L += [f"| {c['participant_code']} | {c['column'].replace(STRUCK, 'struck-through cells')} | {cell(c['in_use'])} | "
          f"{cell(c['received'])} |" for c in changes]
    L += ["", "## What it takes to use the received sheet", "",
          "- The originals stay: move the PDF in use to `data/metadata_raw/archive/` (do not delete it) and put the "
          "received PDF in its place, under the name step 00 reads.",
          "- Run the whole pipeline again from step 00 in the pinned Python 3.12 environment. Step 06 (training) must "
          "run, because the ratings changed. It takes more than an hour.",
          "- After that run this report disappears by itself, and every report, slide and document is rebuilt from the "
          "new results.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"sheet versions: {newest.name} differs from the sheet in use in {len(changes)} cells "
          f"({len(rating_codes)} participants with changed ratings) -> {OUT_MD.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
