"""
03c_audit_supplied_audio.py -- audit of audio files supplied after the original recordings (e.g. CS2B_v2).

Each folder under data/supplied_audio/ is one delivery, listed with its source and hash in
data/supplied_audio/deliveries.csv. Every WAV is checked against the original recordings and the sheet and
gets a verdict. **Nothing from a supplied delivery enters the analysis**: an observation needs an original,
unedited recording (protocol section X). A supplied file can only point to a recording the group still has
to provide, or reveal a problem.

Checks per file
  format      sample rate, channels, sample format, duration
  origin      normalised cross-correlation (NCC) with every original take (the 16 kHz decodes of step 03):
              which take it was cut from, where (start/end within the take), and the gain applied
  editing     clipping (> 0.1 % of samples at full scale); exact digital silence of >= 50 ms (absent from
              every original phone recording); a located word touching the file edge (cut off)
  word        the word in the file name vs the sheet's spoken word and the feeling text
  processing  deliveries with the same file names (e.g. plain and denoised) are compared sample by sample

Verdicts
  COPY_OF_ORIGINAL      cut from this participant's own original, which is already analysed: adds nothing
  DIFFERENT_RECORDING   the participant has originals, but the file matches none of them: unexplained
  OTHER_PARTICIPANT     matches another participant's original: misattributed
  UNCERTAIN_MATCH       partial similarity (NCC between 0.5 and 0.95): cannot be resolved automatically
  NEW_RECORDING         the participant has no original: can count only once the original file is supplied
  PROCESSED_DELIVERY    the whole delivery was processed (e.g. denoised): never used (protocol section X)

Outputs: reports/supplied_audio_audit.md, reports/supplied_audio_audit.csv. Without the supplied WAVs on this
computer (they are not in Git) the existing reports are kept unchanged.

Usage: .venv/Scripts/python.exe src/03c_audit_supplied_audio.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve

from common import (CLEAN_METADATA_CSV, CLIP_FRACTION_LIMIT, DATA, REPORTS, TARGET_SR, locate_utterance,
                    normalise_code, praat_pitch, rel, setup_logging, sha256_of)

SUPPLIED = DATA / "supplied_audio"
DELIVERIES = SUPPLIED / "deliveries.csv"
QA_TABLE = REPORTS / "audio_quality_table.csv"
REVIEW_AUDIO = DATA / "review" / "audio"
REPORT_MD = REPORTS / "supplied_audio_audit.md"
REPORT_CSV = REPORTS / "supplied_audio_audit.csv"
SAME_RECORDING_NCC = 0.95   # at or above: the same recording (resampling differences keep real copies >= 0.97)
UNRELATED_NCC = 0.50        # below: unrelated audio
DIGITAL_SILENCE_S = 0.05    # exact zeros for this long do not occur in a phone recording of a hallway
PROCESSED_RE = re.compile(r"(?i)denois|enhanc|filter|normali[sz]")
NAME_RE = re.compile(r"^(?P<code>Y\d\s*-\s*\d{3})(?:_Y\d_(?:AM|PM)_(?P<word>[A-Za-z]+))?", re.I)

log = setup_logging("supplied_audio", REPORTS / "logs" / "03c_audit_supplied_audio.log")


# ---- building blocks (unit-tested) ---------------------------------------------------------------
def parse_name(filename: str) -> tuple[str | None, str]:
    """'Y2-073_Y2_PM_Kapoy.wav' -> ('Y2-073', 'Kapoy'); 'Y2-050.wav' -> ('Y2-050', '')."""
    m = NAME_RE.match(Path(filename).stem)
    return (normalise_code(m.group("code")), m.group("word") or "") if m else (None, "")


def ncc_best(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Best normalised cross-correlation of the shorter signal inside the longer one.

    Returns (ncc, offset_s, gain): offset of the supplied file `a` within the original `b` (negative when `b`
    lies inside `a`), and gain = rms(a) / rms(the aligned part of b)."""
    swap = len(a) > len(b)
    short, long_ = (b, a) if swap else (a, b)
    n = len(short)
    num = fftconvolve(long_, short[::-1], mode="valid")
    energy = np.concatenate([[0.0], np.cumsum(long_ ** 2)])
    window = np.sqrt(np.maximum(energy[n:] - energy[:-n], 1e-20))
    r = num / (np.linalg.norm(short) * window + 1e-20)
    k = int(np.argmax(np.abs(r)))
    gain = np.linalg.norm(short) / max(float(window[k]), 1e-20)
    return float(r[k]), (-k if swap else k) / TARGET_SR, (1 / gain if swap else gain)


def longest_zero_run(y: np.ndarray) -> int:
    """Length (samples) of the longest run of exact zeros."""
    z = np.concatenate([[0], (y == 0).astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(z))
    return int((edges[1::2] - edges[::2]).max()) if len(edges) else 0


def word_relation(file_word: str, sheet_word: str, feeling: str) -> str:
    if not file_word:
        return "no word in the file name"
    if sheet_word:
        return "same as the sheet" if file_word.casefold() == sheet_word.casefold() else f"differs from the sheet ({sheet_word})"
    if feeling and file_word.casefold() == feeling.casefold():
        return "sheet word blank; equals the FEELING text"
    return "sheet word blank; source unknown"


def verdict(code: str, own_codes: set[str], best_code: str | None, ncc: float, processed: bool) -> str:
    if processed:
        return "PROCESSED_DELIVERY"
    strong = abs(ncc) >= SAME_RECORDING_NCC
    if strong and best_code == code:
        return "COPY_OF_ORIGINAL"
    if strong:
        return "OTHER_PARTICIPANT"
    if abs(ncc) >= UNRELATED_NCC:
        return "UNCERTAIN_MATCH"
    return "DIFFERENT_RECORDING" if code in own_codes else "NEW_RECORDING"


# ---- audit --------------------------------------------------------------------------------------------
def load_originals(qa: pd.DataFrame) -> dict:
    takes = {}
    for _, t in qa.iterrows():
        y, sr = sf.read(REVIEW_AUDIO / f"{t['participant_code']}_t{t['review_take']}.wav", dtype="float64")
        if sr != TARGET_SR:
            raise ValueError(f"original decode at {sr} Hz; expected {TARGET_SR}")
        takes[(t["participant_code"], int(t["review_take"]))] = (y, t)
    return takes


def audit_file(path: Path, delivery: str, processed: bool, takes: dict, meta: pd.DataFrame) -> dict:
    info = sf.info(path)
    y_int, sr = sf.read(path, dtype="int16", always_2d=True)
    y = y_int.mean(axis=1).astype(np.float64) / 32768.0
    code, file_word = parse_name(path.name)
    row = {"delivery": delivery, "file": path.name, "participant_code": code or "", "file_word": file_word,
           "sha256": sha256_of(path), "sample_rate": info.samplerate, "channels": info.channels,
           "subtype": info.subtype, "duration_s": round(info.duration, 3),
           "peak_dbfs": round(20 * np.log10(max(np.abs(y).max(), 1e-10)), 2),
           "clip_pct": round(100 * float(np.mean(np.abs(y_int) >= 32767)), 3),
           "longest_digital_silence_s": round(longest_zero_run(y_int[:, 0]) / sr, 3)}
    flags = []
    if row["clip_pct"] > 100 * CLIP_FRACTION_LIMIT:
        flags.append("CLIPPING")
    if row["longest_digital_silence_s"] >= DIGITAL_SILENCE_S:
        flags.append("DIGITAL_SILENCE_INSERTED")
    if sr != TARGET_SR or info.channels != 1:
        flags.append("FORMAT_DIFFERS")
    if sr == TARGET_SR:
        loc = locate_utterance(y.astype(np.float32), sr)
        t, f0 = praat_pitch(y, sr)
        inside = (t >= loc["seg_start_s"]) & (t <= loc["seg_end_s"])
        row.update({"word_start_s": loc["seg_start_s"], "word_end_s": loc["seg_end_s"], "sbr_db": loc["sbr_db"],
                    "voiced_frames_in_word": int(((f0 > 0) & inside).sum())})
        if loc["touches_start"]:
            flags.append("WORD_CUT_AT_START")
        if loc["touches_end"]:
            flags.append("WORD_CUT_AT_END")
        best = max(((ncc_best(y, yy), c, k) for (c, k), (yy, _) in takes.items()), key=lambda b: abs(b[0][0]))
        (ncc, offset, gain), best_code, best_k = best
        t_row = takes[(best_code, best_k)][1]
        row.update({"best_original": f"{best_code} take {best_k}", "best_ncc": round(ncc, 4),
                    "cut_start_s": round(offset, 3), "cut_end_s": round(offset + info.duration, 3),
                    "gain": round(gain, 3), "original_take_selected_by_pipeline": t_row["selected"] == "True"})
    else:
        best_code, ncc = None, 0.0
    own_codes = {c for c, _ in takes}
    row["verdict"] = verdict(code or "", own_codes, best_code, ncc, processed)
    if abs(ncc) < UNRELATED_NCC:  # no original is related: where it would sit in one means nothing
        for key in ("best_original", "cut_start_s", "cut_end_s", "gain", "original_take_selected_by_pipeline"):
            row[key] = ""
    m = meta.loc[code] if code in meta.index else None
    row["sheet_word"] = m["spoken_word"] if m is not None else ""
    row["feeling"] = m["self_reported_feeling"] if m is not None else ""
    row["word_relation"] = word_relation(file_word, row["sheet_word"], row["feeling"])
    row["flags"] = ";".join(flags)
    return row


def compare_deliveries(dirs: dict[str, Path]) -> list[dict]:
    """Deliveries sharing file names (e.g. a plain and a denoised version) compared sample by sample."""
    out = []
    names = {d: {p.name for p in path.glob("*.wav")} for d, path in dirs.items()}
    keys = sorted(dirs)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            common = sorted(names[a] & names[b])
            if not common:
                continue
            corr, removed = [], []
            for name in common:
                xa, _ = sf.read(dirs[a] / name, dtype="float64")
                xb, _ = sf.read(dirs[b] / name, dtype="float64")
                n = min(len(xa), len(xb))
                xa, xb = xa[:n], xb[:n]
                corr.append(float(np.corrcoef(xa, xb)[0, 1]))
                removed.append(10 * np.log10(max(np.sum((xa - xb) ** 2), 1e-20) / max(np.sum(xa ** 2), 1e-20)))
            out.append({"delivery_a": a, "delivery_b": b, "n_files_in_both": len(common),
                        "median_correlation": round(float(np.median(corr)), 3),
                        "median_difference_db": round(float(np.median(removed)), 1)})
    return out


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    cells = df.map(lambda v: "—" if (isinstance(v, float) and np.isnan(v)) or v == "" else str(v).replace("|", "\\|"))
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(r) + " |" for r in cells.itertuples(index=False)])


def write_report(audit: pd.DataFrame, deliveries: pd.DataFrame, comparisons: list[dict], qa: pd.DataFrame) -> None:
    lines = ["# Supplied audio audit", "",
             "_Generated by `src/03c_audit_supplied_audio.py`. Nothing from a supplied delivery enters the analysis: an "
             "observation needs an original, unedited recording (protocol §X)._", "",
             "## Deliveries", "", md_table(deliveries[["delivery", "source", "received", "n_wav", "note"]]), "",
             "## Verdicts", "", md_table(audit.groupby(["delivery", "verdict"]).size().rename("files").reset_index()), ""]
    plain = audit[audit["verdict"] != "PROCESSED_DELIVERY"]
    for d, a in plain.groupby("delivery"):
        copies = a[a["verdict"] == "COPY_OF_ORIGINAL"]
        unchanged = int((copies["gain"].astype(float).sub(1).abs() < 0.01).sum()) if len(copies) else 0
        lines += [f"## Findings: `{d}`", ""]
        lines.append(f"- **{len(copies)} files are cut from the participant's own original recording**, which the "
                     f"pipeline already analyses ({unchanged} with the volume unchanged); they add no data. Durations "
                     f"{copies['duration_s'].min():.2f}–{copies['duration_s'].max():.2f} s, so they are trimmed copies.")
        multi = set(qa.loc[qa["n_takes"].astype(int) > 1, "participant_code"])
        rt = copies[copies["participant_code"].isin(multi)]
        if len(rt):
            differs = rt[rt["original_take_selected_by_pipeline"] != True]  # noqa: E712
            lines.append(f"- For the {len(rt)} participants with retakes, the delivery uses "
                         + "; ".join(f"{r.participant_code}: {r.best_original}" for r in rt.itertuples())
                         + (f". This differs from the pipeline's label-blind choice for "
                            f"{', '.join(differs['participant_code'])}" if len(differs) else
                            ". These are the takes the pipeline also selected")
                         + ". Whether this was a deliberate choice is unknown: the group decides (`take_overrides.csv`).")
        new = a[a["verdict"] == "NEW_RECORDING"]
        if len(new):
            edited = new[new["flags"] != ""]
            lines.append(f"- **{len(new)} files are recordings of participants without an original** "
                         f"({', '.join(new['participant_code'])}). They are trimmed 16 kHz copies, so they cannot count "
                         "as observations until the original phone recordings are supplied. "
                         + (f"{len(edited)} also show signs of editing or damage: "
                            + "; ".join(f"{r.participant_code} ({r.flags.replace(';', ', ')})" for r in edited.itertuples())
                            + "." if len(edited) else ""))
        for label, v in (("do not match their own original", "DIFFERENT_RECORDING"),
                         ("match another participant's original", "OTHER_PARTICIPANT"),
                         ("only partly match an original", "UNCERTAIN_MATCH")):
            sel = a[a["verdict"] == v]
            if len(sel):
                lines.append(f"- **{len(sel)} file(s) {label}**: " + ", ".join(
                    f"{r.participant_code} (best NCC {r.best_ncc})" for r in sel.itertuples()) + ". The group must explain.")
        wr = a["word_relation"]
        lines.append(f"- **Words in the file names:** {int((wr == 'same as the sheet').sum())} of {len(a)} equal the "
                     f"sheet. {int(wr.str.startswith('differs').sum())} differ from it, "
                     f"{int(wr.str.contains('FEELING').sum())} fill a blank word with the participant's feeling text, and "
                     f"{int(wr.str.contains('source unknown').sum())} fill a blank word from an unknown source. The sheet "
                     "stays the record. A word is changed only through the group's verified corrections or a unanimous "
                     "blind listening review (rule R10); it is never inferred from the feeling text.")
        lines.append("")
    for c in comparisons:
        lines.append(f"- `{c['delivery_b']}` vs `{c['delivery_a']}` ({c['n_files_in_both']} files): median correlation "
                     f"{c['median_correlation']}, median difference {c['median_difference_db']} dB re signal. A processed "
                     "(e.g. denoised) delivery is never used: the protocol forbids denoising.")
    if comparisons:
        lines.append("")
    need_orig = plain.loc[plain["verdict"] == "NEW_RECORDING", "participant_code"].tolist()
    unexplained = plain.loc[plain["verdict"].isin(["DIFFERENT_RECORDING", "OTHER_PARTICIPANT", "UNCERTAIN_MATCH"]),
                            "participant_code"].tolist()
    word_issues = plain[plain["word_relation"].str.startswith(("differs", "sheet word blank"))]
    lines += ["## What the group needs to supply or decide", "",
              (f"1. The **original phone recordings** of {', '.join(need_orig)}, unedited, uploaded to the Drive folder and "
               "added to `data/raw/drive_manifest.csv` (`docs/RUNNING.md` §7). They then enter the analysis like every "
               "other recording." if need_orig else "1. No missing originals."),
              (f"2. An explanation for {', '.join(unexplained)}: which recording is the participant's observation?"
               if unexplained else "2. No unexplained files."),
              "3. The spoken words where the file names and the sheet disagree or the sheet is blank (below): confirm "
              "them from the collectors' notes (`data/metadata_corrections.csv`) or through the blind listening review.",
              "4. Whether the delivery's take choices for the retake participants were deliberate.", "",
              md_table(word_issues[["participant_code", "file_word", "sheet_word", "feeling", "word_relation"]]), "",
              "## Every file", "",
              "Processed deliveries are left out here; every row of every delivery is in "
              f"`{rel(REPORT_CSV)}`.", "",
              md_table(plain[["delivery", "file", "verdict", "duration_s", "best_original", "best_ncc", "cut_start_s",
                              "cut_end_s", "gain", "peak_dbfs", "clip_pct", "longest_digital_silence_s", "flags",
                              "word_relation"]]), ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not DELIVERIES.is_file():
        log.info("no %s: nothing to audit", rel(DELIVERIES))
        return 0
    deliveries = pd.read_csv(DELIVERIES, dtype=str, keep_default_na=False)
    dirs = {d: SUPPLIED / d for d in deliveries["delivery"] if any((SUPPLIED / d).glob("*.wav"))}
    if not dirs:
        log.info("the supplied WAVs are not on this computer (not in Git); keeping %s unchanged", rel(REPORT_MD))
        return 0
    missing = sorted(set(deliveries["delivery"]) - set(dirs))
    if missing:
        raise FileNotFoundError(f"deliveries listed but not found in {rel(SUPPLIED)}: {missing}")
    qa = pd.read_csv(QA_TABLE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    meta = pd.read_csv(CLEAN_METADATA_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig").set_index("participant_code")
    takes = load_originals(qa)
    rows = []
    for d, folder in dirs.items():
        processed = bool(PROCESSED_RE.search(d))
        for path in sorted(folder.glob("*.wav")):
            rows.append(audit_file(path, d, processed, takes, meta))
            log.info("%s/%s: %s %s", d, path.name, rows[-1]["verdict"], rows[-1]["flags"])
    audit = pd.DataFrame(rows)
    duplicates = audit[audit["verdict"] != "PROCESSED_DELIVERY"].groupby("delivery")["sha256"].apply(lambda s: s.duplicated().any())
    if duplicates.any():
        log.warning("byte-identical files inside a delivery: %s", duplicates[duplicates].index.tolist())
    comparisons = compare_deliveries(dirs)
    audit.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    write_report(audit, deliveries, comparisons, qa)
    log.info("audited %d files in %d deliveries: %s -> %s", len(audit), len(dirs),
             audit["verdict"].value_counts().to_dict(), rel(REPORT_MD))
    return 0


if __name__ == "__main__":
    sys.exit(main())
