"""
03_prepare_audio.py -- PHASES 2-5: recording matching, duplicate resolution, audio
standardisation, file naming and the data-quality report.

Inputs
    data/raw/download_log.csv                  raw file <-> Drive id <-> participant code
    data/metadata_cleaned.csv                  Phase 1 output
    data/processed_audio/take_overrides.csv    OPTIONAL group decision on which take to use:
                                               participant_code, drive_file_id, reason, decided_by
    data/processed_audio/segment_overrides.csv OPTIONAL human-verified word boundaries:
                                               participant_code, seg_start_s, seg_end_s, reviewer, note

Outputs
    data/processed_audio/recordings/<name>.wav  selected take, whole clip: WAV, PCM 16-bit, mono, 16 kHz
    data/processed_audio/segments/<name>.wav    the located utterance cut from that WAV (analysis unit)
    data/processed_audio/file_mapping.csv       original file -> participant -> processed file -> status
    data/processed_audio/segment_review_sheet.csv  for the group to verify words/boundaries by listening
    data/analysis_manifest.csv                  one row per participant: included or excluded, and why
    reports/audio_quality_table.csv             per-recording QA (all files, all takes)
    reports/data_quality_report.md              generated report
    figures/qa/*.png                            segmentation review pages, quality overview

Processing (identical for every file; nothing alters vocal characteristics)
    1. QA of the ORIGINAL audio at its native rate/channels: peak, clipping, DC offset, level.
    2. ffmpeg: first audio stream -> mono (channel average) -> 16 kHz (swresample) -> PCM 16-bit WAV,
       metadata stripped, bit-exact flags. No filtering, denoising, normalisation, pitch or speed change.
    3. Label-blind localisation of the dominant utterance (common.locate_utterance) and
       Praat voicing inside/outside it.
    4. One take per participant (rule in select_take); every other take is kept untouched in data/raw.
    5. Name = ParticipantCode_YearLevel_Session_Word.wav (Section XII); unverified word -> 'UNVERIFIED'.

Usage: .venv/Scripts/python.exe src/03_prepare_audio.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

import viz
from viz import plt

from common import (BACKGROUND_VOICED_S, CLEAN_METADATA_CSV, CLEARER_TAKE_DB, CLIP_FRACTION_LIMIT, DATA,  # noqa: E402
                    DOWNLOAD_LOG, FIGURES, FILENAME_RE, HOP_S, LOW_SBR_DB, MIN_SYLLABLE_S, MIN_VOICED_FRAMES,
                    PROCESSED_AUDIO_DIR, PROJECT_ROOT, REPORTS, SEG_COMPETING_DB, SEG_LONG_S, TARGET_SR,
                    UNVERIFIED_WORD_TOKEN, count_syllables, decode_audio, ffmpeg_exe, frame_energy_db,
                    locate_utterance, praat_pitch, probe_media, rel, setup_logging, sha256_of)

PROC_REC = PROCESSED_AUDIO_DIR / "recordings"
PROC_SEG = PROCESSED_AUDIO_DIR / "segments"
TAKE_OVERRIDES = PROCESSED_AUDIO_DIR / "take_overrides.csv"
SEGMENT_OVERRIDES = PROCESSED_AUDIO_DIR / "segment_overrides.csv"
MAPPING = PROCESSED_AUDIO_DIR / "file_mapping.csv"
REVIEW_SHEET = PROCESSED_AUDIO_DIR / "segment_review_sheet.csv"
MANIFEST = DATA / "analysis_manifest.csv"
QA_TABLE = REPORTS / "audio_quality_table.csv"
QA_FIG = FIGURES / "qa"
SEGMENTATION_CRITICAL = ("EDGE_START", "EDGE_END", "LONG_SEGMENT", "TOO_SHORT_FOR_WORD", "COMPETING_EVENT")
INCOMPLETE_WORD = ("EDGE_START", "EDGE_END", "LONG_SEGMENT", "TOO_SHORT_FOR_WORD")
RETAKE_DIR = PROCESSED_AUDIO_DIR / "retake_review"
CAPTURE_OUTLIER_MIN = 60   # capture > 60 min from the median capture time of all recordings
BACKGROUND_MARGIN_S = 0.10

FFMPEG_ARGS = ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(TARGET_SR), "-c:a", "pcm_s16le",
               "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact"]

RETAKE_README = """# Retake listening review

Each folder holds every take of one participant who was recorded more than once:
`<code>_take<k>_<hhmmss>_<id>.wav` is the whole clip (16 kHz mono), `..._word.wav` the automatically located word,
and `_SELECTED` marks the take chosen by the complete-word rule (src/03_prepare_audio.py, `select_take`).
Comparison figures: figures/qa/retakes/<code>_takes.png.

How to review (one Bisaya-speaking group member, ideally two independently):
1. Do NOT look at the participant's valence/arousal or feeling text while listening.
2. For each take, fill retake_review_sheet.csv: heard_word, whole_word_audible_Y_N, clarity_1_to_5,
   other_voice_over_word_Y_N.
3. Choose the take in which the WHOLE word is heard most clearly. Never choose the take that "sounds more
   emotional" - that would let the expected answer steer the data (protocol Section X).
4. If your choice differs from the automatic one, add a row to data/processed_audio/take_overrides.csv:
   participant_code,drive_file_id,reason,decided_by
   then re-run: .venv/Scripts/python.exe run_pipeline.py --skip-fetch --from 03
"""

log = setup_logging("prepare_audio", REPORTS / "logs" / "03_prepare_audio.log")


# ---- per-file analysis ---------------------------------------------------------------
def parse_capture(value: str) -> datetime | None:
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(value.replace("Z", "+0000"), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def decode_16k_int16(path: Path) -> np.ndarray:
    """Exactly the samples that the processed WAV will contain (same ffmpeg arguments)."""
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path), *FFMPEG_ARGS[:-6], "-f", "s16le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name}: {proc.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(proc.stdout, dtype="<i2").copy()


def write_processed_wav(src: Path, dst: Path, expected: np.ndarray) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-i", str(src), *FFMPEG_ARGS, str(dst)],
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed for {src.name}: {proc.stderr.decode(errors='replace')[-300:]}")
    data, sr = sf.read(dst, dtype="int16")
    info = sf.info(dst)
    if sr != TARGET_SR or info.channels != 1 or info.subtype != "PCM_16":
        raise RuntimeError(f"{dst.name}: unexpected format {sr} Hz, {info.channels} ch, {info.subtype}")
    if not np.array_equal(data, expected):
        raise RuntimeError(f"{dst.name}: WAV samples differ from the analysed samples")


def analyse_file(row: pd.Series) -> dict:
    path = PROJECT_ROOT / row["local_file"]
    probe = probe_media(path)
    rec = {
        "participant_code": row["participant_code"], "drive_name": row["drive_name"],
        "drive_file_id": row["drive_file_id"], "original_file": row["local_file"],
        "original_sha256": row["sha256"], "container": row["container"],
        "capture_time": probe.get("com.apple.quicktime.creationdate", probe.get("creation_time", "")),
        "device_model": probe.get("com.apple.quicktime.model", ""), "audio_codec": probe["audio_codec"],
        "orig_sample_rate": probe["sample_rate"], "orig_channels": probe["channel_layout"],
        "orig_audio_streams": probe["n_audio_streams"], "orig_kbps": probe["audio_kbps"],
        "orig_duration_s": round(probe["duration_s"], 3),
    }
    native, _ = decode_audio(path)                       # channels x samples, native rate, float32
    peak = float(np.abs(native).max())
    rec.update({
        "readable": True,
        "peak_dbfs": round(20 * np.log10(max(peak, 1e-10)), 2),
        "clip_fraction": float(np.mean(np.abs(native) >= 0.999)),
        "dc_offset": round(float(np.abs(native.mean(axis=1)).max()), 5),
        "rms_dbfs": round(20 * np.log10(max(float(np.sqrt(np.mean(native ** 2))), 1e-10)), 2),
        "interchannel_corr": round(float(np.corrcoef(native[0], native[1])[0, 1]), 4) if native.shape[0] == 2 else None,
    })
    y16 = decode_16k_int16(path)
    y = y16.astype(np.float32) / 32768.0
    rec["proc_duration_s"] = round(len(y) / TARGET_SR, 3)
    rec.update(locate_utterance(y, TARGET_SR))
    rec["segmentation_source"] = "automatic"
    times, f0 = praat_pitch(y, TARGET_SR)
    voiced = f0 > 0
    inside = (times >= rec["seg_start_s"]) & (times <= rec["seg_end_s"])
    far = (times < rec["seg_start_s"] - BACKGROUND_MARGIN_S) | (times > rec["seg_end_s"] + BACKGROUND_MARGIN_S)
    rec["voiced_frames_in_segment"] = int((voiced & inside).sum())
    rec["voiced_s_outside_segment"] = round(float((voiced & far).sum()) * HOP_S, 2)
    return rec, y16, (times, f0)


def file_flags(rec: dict, median_capture: datetime | None, majority_config: tuple) -> list[str]:
    flags = []
    if rec["touches_start"]:
        flags.append("EDGE_START")
    if rec["touches_end"]:
        flags.append("EDGE_END")
    if rec["seg_duration_s"] > SEG_LONG_S:
        flags.append("LONG_SEGMENT")
    if rec["word_syllables"] and rec["event_duration_s"] < rec["word_syllables"] * MIN_SYLLABLE_S:
        flags.append("TOO_SHORT_FOR_WORD")
    if rec["competing_margin_db"] < SEG_COMPETING_DB:
        flags.append("COMPETING_EVENT")
    if rec["sbr_db"] < LOW_SBR_DB:
        flags.append("LOW_SBR")
    if rec["clip_fraction"] > CLIP_FRACTION_LIMIT:
        flags.append("CLIPPING")
    if rec["voiced_s_outside_segment"] >= BACKGROUND_VOICED_S:
        flags.append("BACKGROUND_VOICES")
    if rec["voiced_frames_in_segment"] < MIN_VOICED_FRAMES:
        flags.append("NO_SPEECH")
    if rec["proc_duration_s"] > 4.0:
        flags.append("LONG_CLIP")
    cap = parse_capture(rec["capture_time"])
    if cap is None:
        flags.append("NO_CAPTURE_TIME")
    elif median_capture is not None and abs((cap - median_capture).total_seconds()) > CAPTURE_OUTLIER_MIN * 60:
        flags.append("CAPTURE_TIME_OUTLIER")
    if (rec["orig_channels"], rec["orig_audio_streams"]) != majority_config:
        flags.append("RECORDING_CONFIG_DIFFERS")
    return flags


def select_take(code: str, takes: list[dict], overrides: dict) -> tuple[dict | None, str]:
    """One observation per participant (protocol Section IV), chosen WITHOUT looking at affect labels.

    Complete-word rule:
      1. only usable takes (decodable, speech present) are candidates;
      2. a group decision in take_overrides.csv, if supplied, wins;
      3. completeness: drop takes whose located word is cut off by the clip edge (EDGE_START/EDGE_END),
         too long to be one word (LONG_SEGMENT) or too short to contain the recorded word
         (TOO_SHORT_FOR_WORD: < 0.10 s per syllable, i.e. faster than 10 syllables/s);
      4. unambiguous target: if some remaining take has no COMPETING_EVENT, drop those that do;
      5. the clearest remaining take (highest utterance-to-background ratio) replaces the earliest only
         if it is >= 3 dB clearer; otherwise the earliest is kept (Section X: natural first response).
    A take is never chosen for sounding more emotional: that would let the label steer the data.
    """
    usable = [t for t in takes if t["usable"]]
    if code in overrides:
        chosen = [t for t in takes if t["drive_file_id"] == overrides[code]["drive_file_id"]]
        if not chosen:
            raise ValueError(f"take override for {code} names an unknown file id")
        return chosen[0], f"group decision: {overrides[code]['reason']} ({overrides[code]['decided_by']})"
    if not usable:
        return None, "no usable take"
    if len(takes) == 1:
        return usable[0], "single recording"
    steps = []
    complete = [t for t in usable if not set(t["flags"]) & set(INCOMPLETE_WORD)]
    pool = complete or usable
    steps.append(f"{len(complete)} of {len(usable)} takes hold a complete word" if complete
                 else "no take holds a complete word, all usable takes kept")
    unambiguous = [t for t in pool if "COMPETING_EVENT" not in t["flags"]]
    if unambiguous and len(unambiguous) < len(pool):
        steps.append(f"{len(pool) - len(unambiguous)} with a competing event dropped")
        pool = unambiguous
    by_time = sorted(pool, key=lambda t: parse_capture(t["capture_time"]) or datetime.max.replace(tzinfo=None))
    earliest, clearest = by_time[0], max(pool, key=lambda t: t["sbr_db"])
    if clearest is not earliest and clearest["sbr_db"] - earliest["sbr_db"] >= CLEARER_TAKE_DB:
        chosen = clearest
        steps.append(f"clearest take is {clearest['sbr_db'] - earliest['sbr_db']:.1f} dB clearer than the earliest")
    else:
        chosen = earliest
        steps.append("only one take remains" if len(pool) == 1 else
                     f"earliest kept (no remaining take is >= {CLEARER_TAKE_DB:g} dB clearer)")
    others = "; ".join(f"{t['drive_name']} [{t['drive_file_id'][:6]}] {t['capture_time'][11:19]} "
                       f"({','.join(t['flags']) or 'no flags'})" for t in takes if t is not chosen)
    return chosen, f"retake ({len(takes)} takes), complete-word rule: " + "; ".join(steps) + f". Others: {others}"


def write_retake_packet(code: str, items: list, word: str) -> tuple[list[dict], str]:
    """Listening packet + comparison figure for one participant with several takes."""
    import librosa
    folder = RETAKE_DIR / code
    folder.mkdir(parents=True, exist_ok=True)
    items = sorted(items, key=lambda it: it[0]["capture_time"])
    rows = []
    fig, axes = plt.subplots(3 * len(items), 1, figsize=(11, 3.3 * len(items)),
                             gridspec_kw={"height_ratios": [1.3, 0.8, 0.7] * len(items), "hspace": 0.55})
    longest = max(it[0]["proc_duration_s"] for it in items)
    for k, (rec, y16, (times, f0)) in enumerate(items, start=1):
        tag = (f"{code}_take{k}_{rec['capture_time'][11:19].replace(':', '')}_{rec['drive_file_id'][:6]}"
               + ("_SELECTED" if rec["selected"] else ""))
        a, b = int(round(rec["seg_start_s"] * TARGET_SR)), int(round(rec["seg_end_s"] * TARGET_SR))
        sf.write(folder / f"{tag}.wav", y16, TARGET_SR, subtype="PCM_16")
        sf.write(folder / f"{tag}_word.wav", y16[a:b], TARGET_SR, subtype="PCM_16")
        y = y16.astype(np.float32) / 32768.0
        dur = len(y) / TARGET_SR
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=160, win_length=400)), ref=np.max)
        ax_s, ax_e, ax_f = axes[3 * (k - 1)], axes[3 * (k - 1) + 1], axes[3 * (k - 1) + 2]
        ax_s.imshow(S, origin="lower", aspect="auto", extent=[0, dur, 0, TARGET_SR / 2], cmap=viz.SEQUENTIAL,
                    vmin=-70, vmax=0)
        ax_s.set_ylim(0, 5000)
        ax_s.set_ylabel("Hz", fontsize=7)
        ax_s.grid(False)
        for x in (rec["seg_start_s"], rec["seg_end_s"]):
            ax_s.axvline(x, color=viz.SERIES[1], lw=1.4)
        db, t = frame_energy_db(y, TARGET_SR)
        ax_e.axvspan(rec["seg_start_s"], rec["seg_end_s"], color=viz.SERIES[2], alpha=0.15, lw=0)
        ax_e.plot(t, db, color=viz.SERIES[0], lw=1.2)
        ax_e.axhline(rec["threshold_db"], color=viz.MUTED, lw=0.8)
        ax_e.set_ylabel("dBFS", fontsize=7)
        ax_f.axvspan(rec["seg_start_s"], rec["seg_end_s"], color=viz.SERIES[2], alpha=0.15, lw=0)
        ax_f.plot(times, np.where(f0 > 0, f0, np.nan), color=viz.SERIES[1], lw=1.6)
        ax_f.set_ylim(60, 520)
        ax_f.set_ylabel("F0 Hz", fontsize=7)
        for ax in (ax_s, ax_e, ax_f):
            ax.set_xlim(0, longest)
            ax.tick_params(labelsize=6.5)
        sps = rec["event_s_per_syllable"]
        ax_s.set_title(f"take {k}  ·  {rec['drive_name']} [{rec['drive_file_id'][:6]}]  ·  captured "
                       f"{rec['capture_time'][11:19]}  ·  {'SELECTED' if rec['selected'] else 'not selected'}  ·  "
                       f"word event {rec['event_duration_s']:.2f} s = {sps:.3f} s/syllable  ·  SBR {rec['sbr_db']:.1f} dB"
                       f"  ·  {', '.join(rec['flags']) or 'no flags'}", fontsize=7.5)
        rows.append({"participant_code": code, "take": k, "drive_name": rec["drive_name"],
                     "drive_file_id": rec["drive_file_id"], "capture_time": rec["capture_time"], "metadata_word": word,
                     "word_syllables": rec["word_syllables"], "event_duration_s": rec["event_duration_s"],
                     "event_s_per_syllable": round(sps, 3), "sbr_db": rec["sbr_db"],
                     "competing_margin_db": rec["competing_margin_db"], "flags": ";".join(rec["flags"]),
                     "auto_selected": rec["selected"], "whole_clip_wav": rel(folder / f"{tag}.wav"),
                     "word_wav": rel(folder / f"{tag}_word.wav"), "heard_word": "", "whole_word_audible_Y_N": "",
                     "clarity_1_to_5": "", "other_voice_over_word_Y_N": "", "choose_this_take_Y_N": "",
                     "listener_initials": "", "notes": ""})
    axes[-1].set_xlabel("time (s).  Spectrogram (orange lines = located word); energy (grey line = threshold); "
                        "Praat F0.  Shaded = located word.", fontsize=7.5)
    fig.suptitle(f"{code} ('{word}', {items[0][0]['word_syllables']} syllables): all takes, earliest first", x=0.01,
                 ha="left", fontsize=10, fontweight="semibold")
    out = QA_FIG / "retakes" / f"{code}_takes.png"
    viz.save(fig, out, dpi=120)
    return rows, rel(out)


# ---- figures ----------------------------------------------------------------------
def plot_review_pages(entries: list[dict]) -> list[str]:
    """Two stacked panels per recording (shared time axis, one y-scale each):
    frame energy with the located utterance, and the Praat F0 contour."""
    paths = []
    per_page = 6
    for old in QA_FIG.glob("segmentation_review_page*.png"):
        old.unlink()
    for page, k in enumerate(range(0, len(entries), per_page), start=1):
        chunk = entries[k:k + per_page]
        fig, axes = plt.subplots(3 * len(chunk), 1, figsize=(11, 2.25 * len(chunk)), sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 0.85, 0.42] * len(chunk), "hspace": 0.12})
        for spacer in axes[2::3]:
            spacer.set_visible(False)
        axes = [a for j, a in enumerate(axes) if j % 3 != 2]
        for i, e in enumerate(chunk):
            ax_e, ax_f = axes[2 * i], axes[2 * i + 1]
            y = e["y16"].astype(np.float32) / 32768.0
            db, t = frame_energy_db(y, TARGET_SR)
            for ax in (ax_e, ax_f):
                ax.axvspan(e["seg_start_s"], e["seg_end_s"], color=viz.SERIES[2], alpha=0.12, lw=0)
            ax_e.plot(t, db, color=viz.SERIES[0], lw=1.2)
            ax_e.axhline(e["threshold_db"], color=viz.MUTED, lw=0.8)
            ax_e.set_ylabel("dBFS", fontsize=7)
            times, f0 = e["pitch"]
            ax_f.plot(times, np.where(f0 > 0, f0, np.nan), color=viz.SERIES[1], lw=1.6)
            ax_f.set_ylim(60, 520)
            ax_f.set_ylabel("F0 Hz", fontsize=7)
            for ax in (ax_e, ax_f):
                ax.tick_params(labelsize=6.5)
            sel = "SELECTED" if e["selected"] else "not selected"
            ax_e.set_title(f"{e['participant_code']}  ·  {e['drive_name']} [{e['drive_file_id'][:6]}]  ·  captured "
                           f"{e['capture_time'][11:19]}  ·  {sel}  ·  utterance {e['seg_start_s']:.2f}–"
                           f"{e['seg_end_s']:.2f} s  ·  {', '.join(e['flags']) or 'no flags'}", fontsize=7.5)
        axes[-1].set_xlim(0, 9.6)
        axes[-1].tick_params(labelbottom=True)
        axes[-1].set_xlabel("time (s).  Upper panel: frame energy (blue) and localisation threshold (grey line); "
                            "lower panel: Praat F0 (orange); shaded: located utterance", fontsize=7.5)
        out = QA_FIG / f"segmentation_review_page{page}.png"
        viz.save(fig, out, dpi=120)
        paths.append(rel(out))
    return paths


def plot_overview(qa: pd.DataFrame) -> str:
    sel = qa[qa["selected"]]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    ax[0].hist(sel["sbr_db"], bins=10, color=viz.SERIES[0], edgecolor=viz.SURFACE, linewidth=2)
    ax[0].axvline(LOW_SBR_DB, color=viz.INK_2, lw=1, label=f"LOW_SBR threshold ({LOW_SBR_DB:g} dB)")
    ax[0].legend(loc="upper right")
    ax[0].set(title="Utterance-to-background ratio", xlabel="dB", ylabel="selected recordings")
    ax[1].hist(sel["seg_duration_s"], bins=10, color=viz.SERIES[0], edgecolor=viz.SURFACE, linewidth=2)
    ax[1].axvline(SEG_LONG_S, color=viz.INK_2, lw=1, label=f"LONG_SEGMENT threshold ({SEG_LONG_S:g} s)")
    ax[1].legend(loc="upper right")
    ax[1].set(title="Located utterance duration", xlabel="s")
    flags = pd.Series([f for fl in sel["flags"] for f in fl.split(";") if f]).value_counts()
    bars = ax[2].barh(flags.index[::-1], flags.values[::-1], color=viz.SERIES[0], height=0.6)
    viz.bar_labels(ax[2], bars, horizontal=True)
    ax[2].set(title=f"QA flags (selected takes, n = {len(sel)})", xlabel="recordings")
    ax[2].tick_params(axis="y", labelsize=7)
    ax[2].grid(axis="y", visible=False)
    fig.tight_layout()
    out = QA_FIG / "audio_quality_overview.png"
    viz.save(fig, out)
    return rel(out)


# ---- report -------------------------------------------------------------------------
def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join("—" if (v is None or v == "" or (isinstance(v, float) and np.isnan(v)))
                                                else str(v) for v in row) + " |" for row in df.itertuples(index=False))


def write_report(qa: pd.DataFrame, manifest: pd.DataFrame, summary: dict, figs: dict) -> None:
    s = summary
    retake_rows = qa[qa["n_takes"] > 1].sort_values(["participant_code", "capture_time"])
    lines = [
        "# Data Quality Report (Phases 2–5)",
        "",
        f"*Generated by `src/03_prepare_audio.py` on {s['generated_at']}. Every number is computed from the files. "
        "Per-recording detail: `reports/audio_quality_table.csv`; mapping: `data/processed_audio/file_mapping.csv`; "
        "participant inclusion: `data/analysis_manifest.csv`.*",
        "",
        "## 1. Summary",
        "",
        "| Quantity | Value |", "|---|---|",
        f"| Total recordings (files) | {s['total_files']} |",
        f"| Unique participants with recordings | {s['participants_with_recordings']} |",
        f"| Metadata participants | {s['metadata_participants']} |",
        f"| Missing recordings (participants) | {len(s['participants_without_recording'])}: {', '.join(s['participants_without_recording'])} |",
        f"| Participants with duplicate recordings (retakes) | {len(s['retake_participants'])}: {', '.join(s['retake_participants'])} "
        f"({s['extra_takes']} extra files) |",
        f"| Unreadable recordings | {s['unreadable_files']} |",
        f"| Recordings without detectable speech | {s['no_speech_files']} |",
        f"| Valid recordings (one selected usable take per participant) | {s['valid_recordings']} |",
        f"| Excluded recordings | {s['excluded_files']} (non-selected retakes: {s['nonselected_retakes']}; unusable: {s['unusable_files']}) |",
        f"| Selected recordings with ≥1 quality flag | {s['selected_with_flags']} |",
        f"| Selected recordings with a segmentation-critical flag | {s['selected_segmentation_critical']} |",
        f"| **Participants included in the primary analysis** | **{s['included_primary']}** (protocol minimum: 40) |",
        "",
        "## 2. Preprocessing (identical for every recording)",
        "",
        "1. **Original audio QA** at native format (all files: AAC audio, 48 kHz, in MOV containers; see table). Peak, clipping "
        f"(> {CLIP_FRACTION_LIMIT:.1%} of samples at |x| ≥ 0.999), DC offset and level are measured before any processing.",
        f"2. **Standardisation:** `ffmpeg -i <original.mov> {' '.join(FFMPEG_ARGS)} <name>.wav` writes the first audio stream as WAV, PCM 16-bit, mono "
        "(channel average), 16 kHz, metadata stripped. Each WAV is read back and checked to be bit-identical to the "
        "analysed samples. There is no filtering, denoising, loudness normalisation, pitch or speed change.",
        "3. **Utterance localisation** (label-blind; `common.locate_utterance`). The clips contain continuous background "
        "babble, so the participant's word is located as the dominant short-time-energy event. Frame RMS is "
        "25 ms/10 ms in dB, smoothed over 50 ms. The floor is the 20th percentile and the threshold is the larger of "
        "floor + 6 dB and the dB mid-point between floor and peak. Dips < 0.25 s are bridged and 0.05 s of context is "
        "added on each side. The located segment is the analysis unit for acoustic features.",
        "4. **Speech check:** Praat autocorrelation pitch (75–500 Hz) must find ≥ 5 voiced frames inside the segment.",
        "5. **One take per participant:** see §5.",
        f"6. **Naming:** `ParticipantCode_YearLevel_Session_Word.wav`. Session token `PM` (ASSUMPTION from the §XII example). "
        f"A missing spoken word is written `{UNVERIFIED_WORD_TOKEN}` and is never inferred.",
        "",
        "**Flag definitions.**",
        "",
        "Segmentation-critical (excluded in the high-confidence sensitivity analysis):",
        "- `EDGE_START` / `EDGE_END`: the event touches the clip boundary, so the word may be cut off.",
        f"- `LONG_SEGMENT`: the event is longer than {SEG_LONG_S} s and may contain more than one word.",
        f"- `TOO_SHORT_FOR_WORD`: the energy event lasts less than {MIN_SYLLABLE_S:g} s per syllable of the recorded "
        "word (faster than 10 syllables/s), so it cannot hold the whole word. Syllables are counted as vowel "
        "letters, which suits Cebuano spelling (y and w are glides; maayo = ma-a-yo).",
        f"- `COMPETING_EVENT`: another event lies within {SEG_COMPETING_DB} dB of the peak, ≥ 0.3 s away, so the target is ambiguous.",
        "",
        "Quality notes:",
        f"- `LOW_SBR`: the event is less than {LOW_SBR_DB} dB above the background.",
        "- `CLIPPING`.",
        f"- `BACKGROUND_VOICES`: ≥ {BACKGROUND_VOICED_S} s of voicing outside the event (other speakers).",
        "- `LONG_CLIP`: the clip is longer than 4 s.",
        f"- `CAPTURE_TIME_OUTLIER`: captured more than {CAPTURE_OUTLIER_MIN} min from the median capture time.",
        "- `RECORDING_CONFIG_DIFFERS`: the channels or number of audio streams differ from the majority of recordings.",
        "",
        f"![overview](../{figs['overview']})",
        "",
        "## 3. Per-recording quality table (all files)",
        "",
        md_table(qa.sort_values(["participant_code", "capture_time"])[[
            "participant_code", "drive_name", "capture_time", "orig_duration_s", "orig_channels", "peak_dbfs",
            "clip_pct", "sbr_db", "seg_start_s", "seg_end_s", "voiced_frames_in_segment", "flags", "usable",
            "duplicate_status", "selected", "standardized_filename", "exclusion_reason"]].assign(
            capture_time=lambda d: d["capture_time"].str[:19].str.replace("T", " "),
            flags=lambda d: d["flags"].str.replace(";", ", "))),
        "",
        "## 4. Participant-level status (all metadata participants)",
        "",
        md_table(manifest[["participant_code", "n_takes", "include_primary", "exclusion_reason", "selected_file",
                           "selected_flags"]].assign(selected_flags=lambda d: d["selected_flags"].str.replace(";", ", "))),
        "",
        "## 5. Duplicate (retake) resolution",
        "",
        f"The {len(s['retake_participants'])} duplicate sets are separate recordings, not copies. They have different "
        "content and SHA-256, and embedded capture times seconds apart on the same device, so they are retakes.",
        "",
        "**Complete-word rule** (label-blind; the affect ratings are never consulted):",
        "1. Drop takes whose located word is cut off by the clip edge, too long to be one word, or too short to "
        f"contain the recorded word (< {MIN_SYLLABLE_S:g} s per syllable).",
        "2. Prefer a take without a competing event of similar loudness.",
        f"3. The clearest take replaces the earliest only if it is ≥ {CLEARER_TAKE_DB:g} dB clearer (signal-to-"
        "background); otherwise the earliest is kept, because §X puts the natural first response first.",
        "",
        "A take is **never** chosen for sounding more emotional: that would let the label steer the data. All takes "
        "stay untouched in `data/raw/`.",
        "",
        "**Listening confirmation:** every take is in `data/processed_audio/retake_review/` (whole clip + located "
        "word) with `retake_review_sheet.csv` and instructions. A choice made by listening goes into "
        "`take_overrides.csv` and replaces the rule. A sensitivity analysis excludes these participants entirely.",
        "",
        md_table(retake_rows[["participant_code", "drive_name", "drive_file_id", "capture_time", "word",
                              "word_syllables", "event_duration_s", "event_s_per_syllable", "sbr_db", "flags",
                              "selected", "selection_reason"]].assign(capture_time=lambda d: d["capture_time"].str[11:19],
                                                          event_s_per_syllable=lambda d: d["event_s_per_syllable"].round(3),
                                                          drive_file_id=lambda d: d["drive_file_id"].str[:8],
                                                          selection_reason=lambda d: np.where(d["selected"], d["selection_reason"].str.split(". Others").str[0], ""))),
        "",
        "",
        *[f"![{c}](../{f})\n" for c, f in sorted(figs.get("retakes", {}).items())],
        "## 6. Recording-configuration and timing anomalies",
        "",
        f"Majority configuration: {s['majority_config']}. Median capture time: {s['median_capture']}.",
        "",
        md_table(qa[qa["flags"].str.contains("CAPTURE_TIME_OUTLIER|RECORDING_CONFIG_DIFFERS")][[
            "participant_code", "drive_name", "capture_time", "orig_channels", "orig_audio_streams", "flags"]]),
        "",
        "Files captured about 9 hours after the session, in a different audio configuration (stereo plus a second "
        "audio track), may be later re-recordings or re-exports. The group must state how and when they were "
        "recorded. If these were not collected immediately after the class (§VII), they violate the protocol. The "
        "primary analysis keeps them, because the metadata records them as post-class observations; a sensitivity "
        "analysis excludes them.",
        "",
        "## 7. Segmentation review (human verification needed)",
        "",
        "The utterance location is automatic; nobody has listened to verify it. For each selected recording, a "
        "Bisaya-speaking member should listen to `data/processed_audio/segments/<name>.wav` and fill "
        "`data/processed_audio/segment_review_sheet.csv` (heard word, boundaries OK, corrected times). Corrected "
        "boundaries go into `segment_overrides.csv` and are used automatically on the next run.",
        "",
        *[f"![review page](../{p})" for p in figs["review_pages"]],
        "",
    ]
    (REPORTS / "data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---- main -----------------------------------------------------------------------------
def main() -> int:
    downloads = pd.read_csv(DOWNLOAD_LOG, dtype=str, keep_default_na=False)
    meta = pd.read_csv(CLEAN_METADATA_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    meta_by_code = meta.set_index("participant_code")
    for p in (PROC_REC, PROC_SEG, QA_FIG):
        p.mkdir(parents=True, exist_ok=True)
    # remove stale outputs so the folders always match the current selection
    for folder in (PROC_REC, PROC_SEG):
        for old in folder.glob("*.wav"):
            old.unlink()
    for old in list(RETAKE_DIR.glob("*/*.wav")) + list((QA_FIG / "retakes").glob("*.png")):
        old.unlink()

    take_over = ({r["participant_code"]: r for _, r in pd.read_csv(TAKE_OVERRIDES, dtype=str, keep_default_na=False).iterrows()}
                 if TAKE_OVERRIDES.is_file() else {})
    seg_over = ({r["participant_code"]: r for _, r in pd.read_csv(SEGMENT_OVERRIDES, dtype=str, keep_default_na=False).iterrows()}
                if SEGMENT_OVERRIDES.is_file() else {})

    # verify raw files are untouched before using them
    for _, d in downloads.iterrows():
        if sha256_of(PROJECT_ROOT / d["local_file"]) != d["sha256"]:
            raise RuntimeError(f"raw file changed since download: {d['local_file']}")

    analysed = []
    for _, d in downloads.iterrows():
        rec, y16, pitch = analyse_file(d)
        analysed.append((rec, y16, pitch))
        log.info("analysed %s [%s] segment %.2f-%.2f s, SBR %.1f dB", rec["drive_name"], rec["drive_file_id"][:6],
                 rec["seg_start_s"], rec["seg_end_s"], rec["sbr_db"])

    caps = [parse_capture(r["capture_time"]) for r, _, _ in analysed]
    caps_ok = sorted(c for c in caps if c is not None)
    median_capture = caps_ok[len(caps_ok) // 2] if caps_ok else None
    configs = pd.Series([(r["orig_channels"], r["orig_audio_streams"]) for r, _, _ in analysed])
    majority_config = configs.value_counts().index[0]

    for rec, y16, pitch in analysed:
        code = rec["participant_code"]
        m = meta_by_code.loc[code]
        word = m["spoken_word"] if m["spoken_word_source"] in ("metadata", "verified_correction") else ""
        rec["word"], rec["word_syllables"] = word, count_syllables(word)
        if code in seg_over:  # human-verified boundaries replace the automatic ones
            o = seg_over[code]
            rec["seg_start_s"], rec["seg_end_s"] = float(o["seg_start_s"]), float(o["seg_end_s"])
            rec["seg_duration_s"] = round(rec["seg_end_s"] - rec["seg_start_s"], 3)
            rec["event_duration_s"] = rec["seg_duration_s"]  # human boundaries have no added context
            rec["segmentation_source"] = f"human ({o['reviewer']})"
            rec["touches_start"] = rec["touches_end"] = False
            rec["competing_margin_db"] = float("inf")
            times, f0 = pitch
            inside = (times >= rec["seg_start_s"]) & (times <= rec["seg_end_s"])
            rec["voiced_frames_in_segment"] = int(((f0 > 0) & inside).sum())
        rec["event_s_per_syllable"] = (round(rec["event_duration_s"] / rec["word_syllables"], 4)
                                       if rec["word_syllables"] else float("nan"))
        rec["flags"] = file_flags(rec, median_capture, majority_config)
        rec["usable"] = rec["readable"] and "NO_SPEECH" not in rec["flags"]

    by_code: dict[str, list] = {}
    for item in analysed:
        by_code.setdefault(item[0]["participant_code"], []).append(item)

    mapping_rows, review_entries, manifest_rows = [], [], []
    for code in meta["participant_code"]:
        m = meta_by_code.loc[code]
        items = by_code.get(code, [])
        takes = [it[0] for it in items]
        n = len(takes)
        chosen, reason = select_take(code, takes, take_over) if takes else (None, "no recording supplied")
        word_ok = m["spoken_word"] != "" and m["spoken_word_source"] in ("metadata", "verified_correction")
        word_token = m["spoken_word"] if word_ok else UNVERIFIED_WORD_TOKEN
        if not word_token.isalpha():
            raise ValueError(f"{code}: spoken word '{word_token}' is not letters only; cannot build a filename")
        std_name = f"{code}_Y{m['year_level']}_{m['session_token_assumed']}_{word_token}.wav"
        if not FILENAME_RE.match(std_name):
            raise ValueError(f"{code}: generated name {std_name} violates the convention")
        for rec, y16, pitch in items:
            rec["n_takes"] = n
            rec["selected"] = rec is chosen
            rec["duplicate_status"] = ("single" if n == 1 else
                                       ("RETAKE_SELECTED" if rec is chosen else "RETAKE_NOT_SELECTED"))
            rec["selection_reason"] = reason if rec is chosen else (f"not selected ({reason.split(';')[0]})")
            rec["standardized_filename"] = std_name if rec is chosen else ""
            if rec is chosen:
                rec["exclusion_reason"] = ""
                write_processed_wav(PROJECT_ROOT / rec["original_file"], PROC_REC / std_name, y16)
                a, b = int(round(rec["seg_start_s"] * TARGET_SR)), int(round(rec["seg_end_s"] * TARGET_SR))
                sf.write(PROC_SEG / std_name, y16[a:b], TARGET_SR, subtype="PCM_16")
                rec["processed_file"] = rel(PROC_REC / std_name)
                rec["segment_file"] = rel(PROC_SEG / std_name)
                rec["processed_sha256"] = sha256_of(PROC_REC / std_name)
                rec["segment_sha256"] = sha256_of(PROC_SEG / std_name)
            else:
                rec["exclusion_reason"] = ("unusable: " + ",".join(f for f in rec["flags"] if f == "NO_SPEECH")
                                           if not rec["usable"] else "retake not selected")
                rec["processed_file"] = rec["segment_file"] = rec["processed_sha256"] = rec["segment_sha256"] = ""
            review_entries.append({**rec, "y16": y16, "pitch": pitch})
            mapping_rows.append({k: rec[k] for k in (
                "original_file", "drive_name", "drive_file_id", "participant_code", "capture_time", "selected",
                "duplicate_status", "selection_reason", "processed_file", "segment_file", "seg_start_s", "seg_end_s",
                "segmentation_source", "processed_sha256", "segment_sha256")} | {
                "status": "SELECTED" if rec["selected"] else rec["duplicate_status"] if rec["usable"] else "UNUSABLE"})

        include = chosen is not None and chosen["usable"] and m["consent_obtained"] == "Y"
        if chosen is None:
            excl = "NO_RECORDING" if n == 0 else "NO_USABLE_RECORDING"
        elif m["consent_obtained"] != "Y":
            excl = "NO_CONSENT"
        else:
            excl = ""
        flags = chosen["flags"] if chosen is not None else []
        manifest_rows.append({
            "participant_code": code, "n_takes": n, "include_primary": include, "exclusion_reason": excl,
            "selected_drive_file_id": chosen["drive_file_id"] if chosen else "",
            "selected_file": chosen["standardized_filename"] if chosen else "",
            "processed_file": chosen["processed_file"] if chosen else "",
            "segment_file": chosen["segment_file"] if chosen else "",
            "selected_flags": ";".join(flags),
            "subset_retake": n > 1,
            "subset_config_or_time_anomaly": bool({"CAPTURE_TIME_OUTLIER", "RECORDING_CONFIG_DIFFERS"} & set(flags)),
            "subset_segmentation_critical": bool(set(SEGMENTATION_CRITICAL) & set(flags)),
            "word_in_filename": word_token,
        })

    # listening packet for every participant with more than one take
    retake_rows, retake_figs = [], {}
    for code, items in by_code.items():
        if len(items) > 1:
            rows, fig_path = write_retake_packet(code, items, meta_by_code.loc[code, "spoken_word"])
            retake_rows += rows
            retake_figs[code] = fig_path
    if retake_rows:
        pd.DataFrame(retake_rows).to_csv(RETAKE_DIR / "retake_review_sheet.csv", index=False, encoding="utf-8-sig")
        (RETAKE_DIR / "README.md").write_text(RETAKE_README, encoding="utf-8")

    # outputs
    qa_rows = []
    for e in review_entries:
        r = {k: v for k, v in e.items() if k not in ("y16", "pitch")}
        r["flags"] = ";".join(r["flags"])
        r["clip_pct"] = round(100 * r["clip_fraction"], 3)
        qa_rows.append(r)
    qa = pd.DataFrame(qa_rows)
    qa.to_csv(QA_TABLE, index=False, encoding="utf-8-sig")
    pd.DataFrame(mapping_rows).to_csv(MAPPING, index=False, encoding="utf-8-sig")
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(MANIFEST, index=False, encoding="utf-8-sig")

    sel = qa[qa["selected"]]
    review = sel[["participant_code", "standardized_filename", "segment_file", "seg_start_s", "seg_end_s", "flags"]].copy()
    meta_words = meta.set_index("participant_code")["spoken_word"]
    review["metadata_word"] = review["participant_code"].map(meta_words)
    for col in ("heard_word", "boundaries_ok_Y_N", "corrected_start_s", "corrected_end_s", "reviewer_initials", "notes"):
        review[col] = ""
    review.to_csv(REVIEW_SHEET, index=False, encoding="utf-8-sig")

    figs = {"retakes": retake_figs,
            "review_pages": plot_review_pages(sorted(review_entries, key=lambda e: (e["participant_code"], e["capture_time"]))),
            "overview": plot_overview(qa)}
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_files": int(len(qa)),
        "participants_with_recordings": int(qa["participant_code"].nunique()),
        "metadata_participants": int(len(meta)),
        "participants_without_recording": manifest.loc[manifest["n_takes"] == 0, "participant_code"].tolist(),
        "retake_participants": sorted(qa.loc[qa["n_takes"] > 1, "participant_code"].unique().tolist()),
        "extra_takes": int((qa["duplicate_status"] == "RETAKE_NOT_SELECTED").sum()),
        "unreadable_files": int((~qa["readable"]).sum()),
        "no_speech_files": int(qa["flags"].str.contains("NO_SPEECH").sum()),
        "valid_recordings": int((qa["selected"] & qa["usable"]).sum()),
        "excluded_files": int((~qa["selected"]).sum()),
        "nonselected_retakes": int(((qa["duplicate_status"] == "RETAKE_NOT_SELECTED") & qa["usable"]).sum()),
        "unusable_files": int((~qa["usable"]).sum()),
        "selected_with_flags": int((sel["flags"] != "").sum()),
        "selected_segmentation_critical": int(sel["flags"].apply(lambda f: bool(set(f.split(";")) & set(SEGMENTATION_CRITICAL))).sum()),
        "included_primary": int(manifest["include_primary"].sum()),
        "majority_config": f"{majority_config[0]}, {majority_config[1]} audio stream(s)",
        "median_capture": median_capture.isoformat() if median_capture else "",
        "flag_counts_selected": pd.Series([f for fl in sel["flags"] for f in fl.split(";") if f]).value_counts().to_dict(),
        "subsets": {k: int(manifest[k].sum()) for k in ("subset_retake", "subset_config_or_time_anomaly",
                                                        "subset_segmentation_critical")},
    }
    (REPORTS / "data_quality_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_report(qa, manifest, summary, figs)
    log.info("valid recordings: %d; included in primary analysis: %d", summary["valid_recordings"], summary["included_primary"])
    log.info("selected-take flags: %s", summary["flag_counts_selected"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
