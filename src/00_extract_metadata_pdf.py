"""
00_extract_metadata_pdf.py -- Phase 0 helper: PDF metadata export -> raw CSV.

Group 3 supplied its participant metadata only as a PDF export of a Google
Sheet ("G3 - mini project - TEmplate.pdf"). This script converts that PDF into
a machine-readable CSV WITHOUT altering any cell value, so the audit and later
cleaning steps work from a reproducible, verifiable copy of the raw metadata.

Research-relevant extraction rules
----------------------------------
* Cell text is rebuilt character by character from the PDF text layer, keeping
  internal spaces exactly (e.g. "Y2- 041" stays "Y2- 041"). Nothing is
  normalised here; normalisation belongs to Phase 1 and keeps the raw value.
* Cells are located with the table's own drawn grid lines, so an empty cell
  stays empty instead of a neighbouring value sliding into it.
* Text that Google Sheets wrapped onto several lines inside one cell is
  re-joined: a line ending in "-" is joined without a space (Sheets wraps after
  hyphens), otherwise with one space. Every wrapped cell is listed in
  `_wrapped_fields` so the rule can be checked.
* A horizontal line drawn through the middle of a cell's text (strikethrough
  formatting) is recorded in `_struck_fields` / `_struck_detail`. The struck
  text itself is kept; what it means is decided in the audit, not here.
* Non-black text and coloured cell fills are recorded too, because collectors
  sometimes use colour to mark problems.

If the group later supplies a CSV/XLSX export of the original sheet, that file
should replace this PDF-derived CSV as the raw metadata source.

Usage (from the project root):
    .venv/Scripts/python.exe src/00_extract_metadata_pdf.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = PROJECT_ROOT / "data" / "metadata_raw" / "G3 - mini project - TEmplate.pdf"
DEFAULT_OUT = PROJECT_ROOT / "data" / "metadata_raw" / "G3_metadata_extracted_raw.csv"

# Header exactly as it appears in the Group 3 sheet. Extraction stops with an
# error if the PDF header differs, rather than guessing a column mapping.
EXPECTED_HEADER = [
    "Participant Code", "Year Level", "Class Activity", "Subject", "Session",
    "Spoken Word", "Self-Reported Feeling", "Valence (1-5)", "Arousal (1-5)",
    "Audio Filename", "Consent Obtained (Y/N)", "Date Collected",
    "Collector Initials", "Notes",
]

GRID_TOL = 1.0        # pt: border lines drawn closer than this are one border
MIN_GRID_REPEATS = 3  # vertical borders are drawn once per row, so they repeat
ROW_LINE_SPAN = 0.9   # horizontal borders span the whole table; a strike line does not
LINE_TOL = 2.0        # pt: chars whose centres differ less than this share a line

log = logging.getLogger("extract_metadata")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cluster(values: list[float], tol: float) -> list[float]:
    """Merge sorted positions closer than `tol` into their mean."""
    groups: list[list[float]] = []
    for v in sorted(values):
        if groups and v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def page_grid(page: pymupdf.Page):
    """Return column edges, row edges and non-grid horizontal segments."""
    vx: Counter = Counter()
    hsegs = []  # (x0, x1, y)
    fills = []  # (rect, rgb)
    for d in page.get_drawings():
        fill = d.get("fill")
        for item in d["items"]:
            kind = item[0]
            if kind == "l":
                a, b = item[1], item[2]
                if abs(a.x - b.x) < 0.6 and abs(a.y - b.y) > 3:
                    vx[round((a.x + b.x) / 2, 1)] += 1
                elif abs(a.y - b.y) < 0.6 and abs(a.x - b.x) > 0.5:
                    hsegs.append((min(a.x, b.x), max(a.x, b.x), round((a.y + b.y) / 2, 1)))
            elif kind == "re":
                rect = item[1]
                if rect.height <= 1.5:
                    hsegs.append((rect.x0, rect.x1, round((rect.y0 + rect.y1) / 2, 1)))
                elif fill is not None:
                    fills.append((rect, tuple(round(c, 3) for c in fill)))
    col_edges = cluster([x for x, n in vx.items() if n >= MIN_GRID_REPEATS], GRID_TOL)
    if len(col_edges) < 2:
        raise ValueError(f"page {page.number + 1}: no table grid found")
    table_width = col_edges[-1] - col_edges[0]
    is_row_border = [(x1 - x0) >= ROW_LINE_SPAN * table_width for x0, x1, _ in hsegs]
    row_edges = cluster([s[2] for s, keep in zip(hsegs, is_row_border) if keep], GRID_TOL)
    other = [s for s, keep in zip(hsegs, is_row_border) if not keep]
    return col_edges, row_edges, other, fills


def locate(value: float, edges: list[float]) -> int | None:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return None


def extract_page_cells(page: pymupdf.Page):
    """Map every character to a (row, col) cell; collect styling evidence."""
    col_edges, row_edges, other_hsegs, fills = page_grid(page)
    n_cols = len(col_edges) - 1
    if n_cols != len(EXPECTED_HEADER):
        raise ValueError(
            f"page {page.number + 1}: found {n_cols} grid columns, expected "
            f"{len(EXPECTED_HEADER)}; table layout differs from the Group 3 template"
        )
    cells: dict[tuple[int, int], list[dict]] = {}
    unassigned: list[str] = []
    raw = page.get_text("rawdict")
    for block in raw["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                for ch in span["chars"]:
                    x0, y0, x1, y1 = ch["bbox"]
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    r, c = locate(cy, row_edges), locate(cx, col_edges)
                    if r is None or c is None:
                        if ch["c"].strip():
                            unassigned.append(ch["c"])
                        continue
                    cells.setdefault((r, c), []).append(
                        {"c": ch["c"], "bbox": (x0, y0, x1, y1), "color": span["color"]}
                    )
    return cells, row_edges, col_edges, other_hsegs, fills, "".join(unassigned)


def cell_text(chars: list[dict]) -> tuple[str, bool]:
    """Rebuild a cell's text; returns (text, was_wrapped_over_several_lines)."""
    lines: list[list[dict]] = []
    for ch in sorted(chars, key=lambda c: (c["bbox"][1] + c["bbox"][3]) / 2):
        cy = (ch["bbox"][1] + ch["bbox"][3]) / 2
        if lines and abs(cy - lines[-1][0]["_cy"]) <= LINE_TOL:
            lines[-1].append({**ch, "_cy": cy})
        else:
            lines.append([{**ch, "_cy": cy}])
    texts = ["".join(c["c"] for c in sorted(ln, key=lambda c: c["bbox"][0])) for ln in lines]
    joined = texts[0] if texts else ""
    for nxt in texts[1:]:
        # The wrap point is a rendering artefact: drop whitespace at the break,
        # keep a hyphen that ends the line, and restore one space otherwise.
        joined = joined.rstrip()
        joined = joined + nxt.lstrip() if joined.endswith("-") else joined + " " + nxt.lstrip()
    return joined, len(texts) > 1


def struck_portion(chars: list[dict], segs) -> str | None:
    """Return the characters crossed by a mid-height line (strikethrough)."""
    hit = []
    for ch in chars:
        x0, y0, x1, y1 = ch["bbox"]
        h = y1 - y0
        for sx0, sx1, sy in segs:
            if y0 + 0.25 * h <= sy <= y1 - 0.25 * h and sx0 < x1 - 0.5 and sx1 > x0 + 0.5:
                hit.append(ch)
                break
    if not hit:
        return None
    return "".join(c["c"] for c in sorted(hit, key=lambda c: c["bbox"][0])).strip()


def extract(pdf_path: Path) -> tuple[pd.DataFrame, dict]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"metadata PDF not found: {pdf_path}. It is private (not in the public "
                                "repository): copy it from the group's private share (docs/RUNNING.md, step 3).")
    doc = pymupdf.open(pdf_path)
    records: list[dict] = []
    header_seen = False
    first_col_edges = None
    unassigned_all = []
    for page in doc:
        cells, row_edges, col_edges, segs, fills, unassigned = extract_page_cells(page)
        if unassigned:
            unassigned_all.append({"page": page.number + 1, "text": unassigned})
            log.warning("page %d: text outside the table grid: %r", page.number + 1, unassigned)
        if first_col_edges is None:
            first_col_edges = col_edges
        elif any(abs(a - b) > GRID_TOL for a, b in zip(first_col_edges, col_edges)):
            raise ValueError(f"page {page.number + 1}: column grid differs from page 1")
        for r in range(len(row_edges) - 1):
            row_vals, wrapped, struck, nonblack, filled = {}, [], [], [], []
            for c, name in enumerate(EXPECTED_HEADER):
                chars = cells.get((r, c), [])
                text, was_wrapped = cell_text(chars) if chars else ("", False)
                row_vals[name] = text
                if was_wrapped:
                    wrapped.append(name)
                portion = struck_portion(chars, segs) if chars else None
                if portion:
                    struck.append((name, portion, text))
                if any(ch["color"] != 0 for ch in chars):
                    nonblack.append(name)
                cell_rect = pymupdf.Rect(col_edges[c], row_edges[r], col_edges[c + 1], row_edges[r + 1])
                for rect, rgb in fills:
                    if rgb != (1.0, 1.0, 1.0) and rect.intersects(cell_rect) and \
                            (rect & cell_rect).get_area() > 0.5 * cell_rect.get_area():
                        filled.append(f"{name}:{rgb}")
            if not header_seen:
                header = [row_vals[n] for n in EXPECTED_HEADER]
                if header != EXPECTED_HEADER:
                    raise ValueError(f"unexpected header row: {header}")
                header_seen = True
                continue
            if not any(v.strip() for v in row_vals.values()):
                log.info("page %d row %d: empty row skipped", page.number + 1, r)
                continue
            row_vals.update({
                "_source_page": page.number + 1,
                "_source_row_on_page": r,
                "_wrapped_fields": ";".join(wrapped),
                "_struck_fields": ";".join(n for n, _, _ in struck),
                "_struck_detail": " | ".join(
                    f"{n}: '{p}' struck ({'entire cell' if p == t.strip() else 'part of cell'})"
                    for n, p, t in struck
                ),
                "_nonblack_text_fields": ";".join(nonblack),
                "_filled_fields": ";".join(filled),
            })
            records.append(row_vals)
    if not header_seen or not records:
        raise ValueError(f"no participant rows extracted from {pdf_path.name}; check the table grid")
    df = pd.DataFrame.from_records(records)
    provenance = {
        "source_pdf": pdf_path.relative_to(PROJECT_ROOT).as_posix(),
        "source_sha256": sha256_of(pdf_path),
        "source_bytes": pdf_path.stat().st_size,
        "pdf_metadata": {k: v for k, v in doc.metadata.items() if v},
        "pages": doc.page_count,
        "rows_extracted": len(df),
        "text_outside_grid": unassigned_all,
        "pymupdf_version": pymupdf.VersionBind,
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rules": [
            "cell text rebuilt from PDF characters; internal spaces preserved; no normalisation",
            "cells located by drawn grid lines; empty cells stay empty",
            "wrapped lines joined without space after a trailing hyphen, else one space",
            "mid-height horizontal line through text recorded as strikethrough; text kept",
        ],
    }
    return df, provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    log_dir = PROJECT_ROOT / "reports" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(log_dir / "00_extract_metadata_pdf.log", mode="w", encoding="utf-8")])

    df, prov = extract(args.pdf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel shows the em dash in "3 — Neutral" correctly.
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    prov_path = args.out.with_name(args.out.stem + "_provenance.json")
    prov["output_csv"] = args.out.relative_to(PROJECT_ROOT).as_posix()
    prov["output_sha256"] = sha256_of(args.out)
    prov_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("extracted %d participant rows -> %s", len(df), args.out.relative_to(PROJECT_ROOT))
    struck = df[df["_struck_fields"] != ""]
    for _, row in struck.iterrows():
        log.warning("strikethrough in %s: %s", row["Participant Code"], row["_struck_detail"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
