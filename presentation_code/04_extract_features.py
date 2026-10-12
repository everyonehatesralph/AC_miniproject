"""
04_extract_features.py -- PHASE 6: acoustic/prosodic features of each located utterance.

Input   data/analysis_manifest.csv (include_primary rows) and their segment WAVs
        (data/processed_audio/segments/*.wav: PCM 16-bit, mono, 16 kHz)
Output  data/acoustic_features.csv               participant_code + 108 numeric features (what the models read)
        data/acoustic_features_dictionary.csv    name, group, unit, definition
        data/acoustic_features_with_labels.csv   the same rows with the participant's labels in front (for
                                                 reading and submission; never a model input as a whole)
        reports/features_summary.json

Features (Protocol Section XIV; frames 25 ms / hop 10 ms, Hann window, uncentred so that
no zero padding enters the statistics; n_fft 512):
    duration (3)  utterance_duration_s, voiced_duration_s, voiced_ratio
    F0 (9)        Praat autocorrelation pitch on voiced frames: mean, std, median, min, max, IQR (Hz);
                  f0_std_st, f0_range_st (5th-95th percentile, semitones) and f0_slope_st_per_s.
                  Hz values mostly reflect voice type (no speaker baseline exists, one utterance per
                  participant); the semitone measures are scale-free.
    RMS (6)       frame RMS in dBFS: mean, std, median, min, max, IQR
    ZCR (6)       zero-crossing rate per frame: mean, std, median, min, max, IQR
    MFCC (26)     13 coefficients (c0..c12, 40 mel bands 0-8 kHz, DCT-II ortho): mean and std
    delta (26)    first derivative of the MFCCs: mean and std
    delta2 (26)   second derivative: mean and std
    spectral (6)  centroid, bandwidth, 85% roll-off: mean and std
Standard deviations use ddof=0. The participant code is an identifier only, never a feature.

Usage: .venv/Scripts/python.exe src/04_extract_features.py
"""

from __future__ import annotations

import json
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from common import (CLEAN_METADATA_CSV, DATA, FEATURES_CSV, FRAME_S, HOP_S, N_FFT, PITCH, PROJECT_ROOT, REPORTS,
                    TARGET_SR, TARGETS, TASK_ROLE, derive_target, praat_pitch, rel, setup_logging, sha256_of)

MANIFEST = DATA / "analysis_manifest.csv"
DICTIONARY = DATA / "acoustic_features_dictionary.csv"
LABELLED = DATA / "acoustic_features_with_labels.csv"
LABEL_COLS = ["year_level", "session", "class_activity_protocol", "spoken_word", "spoken_word_source",
              "self_reported_feeling", "valence", "valence_status", "arousal", "arousal_status"]
N_MFCC = 13
N_MELS = 40
MIN_VOICED_FOR_F0 = 3
STATS = ("mean", "std", "median", "min", "max", "iqr")

log = setup_logging("features", REPORTS / "logs" / "04_extract_features.log")


def six_stats(x: np.ndarray, prefix: str) -> dict:
    x = np.asarray(x, dtype=float)
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return {f"{prefix}_mean": float(x.mean()), f"{prefix}_std": float(x.std()), f"{prefix}_median": float(q50),
            f"{prefix}_min": float(x.min()), f"{prefix}_max": float(x.max()), f"{prefix}_iqr": float(q75 - q25)}


def extract(y: np.ndarray, sr: int) -> dict:
    win, hop = int(FRAME_S * sr), int(HOP_S * sr)
    feats: dict[str, float] = {}

    # --- F0 and voicing (Praat) ---
    times, f0 = praat_pitch(y, sr)
    voiced = f0 > 0
    n_voiced = int(voiced.sum())
    feats["utterance_duration_s"] = len(y) / sr
    feats["voiced_duration_s"] = n_voiced * HOP_S
    feats["voiced_ratio"] = n_voiced / len(f0) if len(f0) else 0.0
    if n_voiced >= MIN_VOICED_FOR_F0:
        fv, tv = f0[voiced], times[voiced]
        feats.update(six_stats(fv, "f0"))  # Hz
        st = 12 * np.log2(fv / np.median(fv))
        p5, p95 = np.percentile(fv, [5, 95])
        feats["f0_std_st"] = float(st.std())
        feats["f0_range_st"] = float(12 * np.log2(p95 / p5))
        feats["f0_slope_st_per_s"] = float(np.polyfit(tv, st, 1)[0])
    else:
        feats.update({f"f0_{s}": np.nan for s in STATS})
        feats.update({"f0_std_st": np.nan, "f0_range_st": np.nan, "f0_slope_st_per_s": np.nan})

    # --- energy and zero crossings ---
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=False)[0]
    feats.update(six_stats(20 * np.log10(np.maximum(rms, 1e-10)), "rms_db"))
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop, center=False)[0]
    feats.update(six_stats(zcr, "zcr"))

    # --- spectral representation ---
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=hop, win_length=win, window="hann", center=False))
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=N_MELS, fmin=0.0, fmax=sr / 2)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC, dct_type=2, norm="ortho")
    n_frames = mfcc.shape[1]
    width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
    if width < 3:
        raise ValueError(f"segment too short for delta features ({n_frames} frames)")
    d1 = librosa.feature.delta(mfcc, width=width, order=1)
    d2 = librosa.feature.delta(mfcc, width=width, order=2)
    for name, mat in (("mfcc", mfcc), ("delta_mfcc", d1), ("delta2_mfcc", d2)):
        for i in range(N_MFCC):
            feats[f"{name}_{i + 1}_mean"] = float(mat[i].mean())
            feats[f"{name}_{i + 1}_std"] = float(mat[i].std())
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    for name, v in (("spectral_centroid_hz", centroid), ("spectral_bandwidth_hz", bandwidth),
                    ("spectral_rolloff_hz", rolloff)):
        feats[f"{name}_mean"] = float(v.mean())
        feats[f"{name}_std"] = float(v.std())
    return feats


def dictionary(columns: list[str]) -> pd.DataFrame:
    rows = []
    for c in columns:
        if c in ("utterance_duration_s", "voiced_duration_s", "voiced_ratio"):
            group, unit = "duration/voicing", "s" if c.endswith("_s") else "ratio"
            text = {"utterance_duration_s": "duration of the located utterance segment (incl. 0.05 s context each side)",
                    "voiced_duration_s": "number of Praat-voiced 10 ms frames x 0.01 s",
                    "voiced_ratio": "voiced frames / all pitch frames in the segment"}[c]
        elif c.startswith("f0_"):
            group = "F0 (pitch)"
            unit = "st" if c.endswith("_st") else ("st/s" if c.endswith("per_s") else "Hz")
            text = {"f0_std_st": "std of F0 in semitones relative to the utterance median",
                    "f0_range_st": "F0 range between the 5th and 95th percentile, in semitones",
                    "f0_slope_st_per_s": "least-squares slope of the semitone F0 contour over time"}.get(
                c, f"{c.split('_')[1]} of F0 over voiced frames (Praat ac, 75-500 Hz)")
        elif c.startswith("rms_db"):
            group, unit, text = "energy", "dBFS", f"{c.split('_')[-1]} of frame RMS level (25 ms frames)"
        elif c.startswith("zcr"):
            group, unit, text = "zero-crossing rate", "crossings/sample", f"{c.split('_')[-1]} of frame ZCR"
        elif c.startswith("delta2_mfcc"):
            group, unit, text = "delta-delta MFCC", "a.u.", f"{c.split('_')[-1]} of 2nd derivative of MFCC {c.split('_')[2]}"
        elif c.startswith("delta_mfcc"):
            group, unit, text = "delta MFCC", "a.u.", f"{c.split('_')[-1]} of 1st derivative of MFCC {c.split('_')[2]}"
        elif c.startswith("mfcc"):
            group, unit, text = "MFCC", "a.u.", f"{c.split('_')[-1]} of MFCC {c.split('_')[1]} (c{int(c.split('_')[1]) - 1})"
        else:
            group, unit, text = "spectral", "Hz", f"{c.split('_')[-1]} of the frame spectral {c.split('_')[1]}"
        rows.append({"feature": c, "group": group, "unit": unit, "definition": text})
    return pd.DataFrame(rows)


def main() -> int:
    manifest = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    included = manifest[manifest["include_primary"] == "True"]
    rows = []
    for _, m in included.iterrows():
        path = PROJECT_ROOT / m["segment_file"]
        y, sr = sf.read(path, dtype="float32")
        if sr != TARGET_SR or y.ndim != 1:
            raise ValueError(f"{path.name}: expected mono {TARGET_SR} Hz, got {sr} Hz, ndim {y.ndim}")
        feats = extract(y, sr)
        rows.append({"participant_code": m["participant_code"], "segment_file": m["segment_file"],
                     "segment_sha256": sha256_of(path), **feats})
        log.info("%s: %d features, %d voiced frames", m["participant_code"], len(feats),
                 round(feats["voiced_duration_s"] / HOP_S))
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ("participant_code", "segment_file", "segment_sha256")]
    if df["participant_code"].duplicated().any():
        raise ValueError("more than one feature row for a participant")
    non_f0 = [c for c in feature_cols if not c.startswith("f0_")]
    if not np.isfinite(df[non_f0].to_numpy()).all():
        raise ValueError("non-finite value in a non-F0 feature")
    df.to_csv(FEATURES_CSV, index=False, float_format="%.6g")
    dictionary(feature_cols).to_csv(DICTIONARY, index=False)
    # the same rows with the labels in front, for people to read; 06 trains on FEATURES_CSV and refuses any
    # non-acoustic column, so this combined file can never leak a label into a model
    meta = pd.read_csv(CLEAN_METADATA_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    labels = meta[["participant_code", *LABEL_COLS]].copy()
    for task in (t for t, role in TASK_ROLE.items() if role in ("primary", "secondary")):
        labels[f"derived_{task}"] = derive_target(pd.to_numeric(labels[TARGETS[task]["source"]], errors="coerce"), task)
    flags = included.set_index("participant_code")["selected_flags"].rename("recording_flags")
    labelled = labels.merge(flags, left_on="participant_code", right_index=True, how="inner")
    labelled = labelled.merge(df, on="participant_code", how="right", validate="one_to_one")
    labelled.to_csv(LABELLED, index=False, float_format="%.6g")
    summary = {
        "rows": int(len(df)), "n_features": len(feature_cols),
        "features_by_group": dictionary(feature_cols)["group"].value_counts().to_dict(),
        "missing_values": {c: int(df[c].isna().sum()) for c in feature_cols if df[c].isna().any()},
        "pitch_settings": PITCH, "frame_s": FRAME_S, "hop_s": HOP_S, "n_fft": N_FFT, "n_mfcc": N_MFCC, "n_mels": N_MELS,
        "librosa_version": librosa.__version__, "output": rel(FEATURES_CSV), "output_sha256": sha256_of(FEATURES_CSV),
    }
    (REPORTS / "features_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("%d participants x %d features -> %s; missing: %s", len(df), len(feature_cols), rel(FEATURES_CSV),
             summary["missing_values"] or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
