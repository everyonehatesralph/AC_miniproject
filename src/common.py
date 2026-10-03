"""
common.py -- shared paths, protocol constants and helpers for the Group 3 pipeline.

Every script in src/ imports from here so that a constant (a protocol rule, a
path, the random seed) is defined exactly once.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---- paths (all relative to the project root) ---------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
RAW_DIR = DATA / "raw"
METADATA_RAW_DIR = DATA / "metadata_raw"
PROCESSED_AUDIO_DIR = DATA / "processed_audio"
CONSENT_DIR = PROJECT_ROOT / "consent"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = PROJECT_ROOT / "figures"
MODELS = PROJECT_ROOT / "models"

RAW_METADATA_CSV = METADATA_RAW_DIR / "G3_metadata_extracted_raw.csv"
CLEAN_METADATA_CSV = DATA / "metadata_cleaned.csv"
RECORDING_MAP = RAW_DIR / "recording_map.csv"
DRIVE_MANIFEST = RAW_DIR / "drive_manifest.csv"
DOWNLOAD_LOG = RAW_DIR / "download_log.csv"
FEATURES_CSV = DATA / "acoustic_features.csv"

RANDOM_STATE = 42

# ---- protocol constants: CS412 Research Mini-Project v2 -----------------------
GROUP_YEAR_LEVEL = 2          # Section IV: Group 3 = Second-Year CS students
MIN_VALID_OBSERVATIONS = 40   # Section IV / Section XVIII item 1
PROTOCOL_WORDS = ("Salamat", "Ambot", "Grabe", "Lisod", "Sayon", "Lingaw", "Kapoy")  # Section VIII
PROTOCOL_ACTIVITIES = (       # Section VII, Step 2
    "Lecture/Discussion", "Laboratory Activity", "Programming/Coding Activity", "Quiz",
    "Examination", "Presentation/Reporting", "Group Activity", "Recitation",
    "Project/Practical Activity", "Other",
)
PROTOCOL_SESSIONS = ("Morning", "Afternoon–Evening")  # Section VII (en dash)
VALENCE_LABELS = {1: "Very Unpleasant", 2: "Unpleasant", 3: "Neutral", 4: "Pleasant", 5: "Very Pleasant"}
AROUSAL_LABELS = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}  # Section IX

# Section XII example "Y2-017_Y2_PM_Grabe.wav" is the only definition of the
# session token: Morning -> AM, Afternoon-Evening -> PM is an ASSUMPTION.
SESSION_TOKEN_ASSUMED = {"morning": "AM", "afternoon-evening": "PM"}
UNVERIFIED_WORD_TOKEN = "UNVERIFIED"  # filename token when the spoken word is not verified

CODE_STRICT_RE = re.compile(r"^Y([1-4])-(\d{3})$")
CODE_IN_NAME_RE = re.compile(r"(?i)(?<![A-Za-z0-9])Y\s*([1-4])\s*[-_ ]?\s*(\d{3})(?!\d)")
FILENAME_RE = re.compile(r"^(Y[1-4]-\d{3})_(Y[1-4])_(AM|PM)_([A-Za-z]+)\.wav$")
DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−"})

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr"}
VIDEO_EXTS = {".mov", ".qt", ".mp4", ".m4v", ".3gp", ".avi", ".mkv", ".webm"}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS


# ---- generic helpers --------------------------------------------------------------
def setup_logging(name: str, logfile: Path | None = None) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, mode="w", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, force=True, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def normalise_code(raw: str) -> str | None:
    """'Y2- 041' / 'y2-041' -> 'Y2-041'; None if it is not a participant code."""
    m = CODE_IN_NAME_RE.search(raw)
    return f"Y{m.group(1)}-{m.group(2)}" if m else None


def session_key(value: str) -> str:
    return value.translate(DASHES).casefold().replace(" ", "")


# ---- media helpers -------------------------------------------------------------------
def sniff_container(path: Path) -> str:
    """Identify the container from the file signature, not the extension."""
    with path.open("rb") as fh:
        head = fh.read(16)
    if len(head) < 12:
        return "EMPTY_OR_TRUNCATED"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "WAV"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "AVI"
    if head[4:8] == b"ftyp":
        brand = head[8:12].decode("latin-1")
        if brand == "qt  ":
            return "QuickTime MOV"
        if brand.startswith("3g"):
            return "3GPP"
        return f"MPEG-4 ({brand.strip()})"
    if head[4:8] in (b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"):
        return "QuickTime MOV (legacy)"
    if head[:3] == b"ID3" or (head[0] == 0xFF and head[1] & 0xF6 in (0xF2, 0xF4) and head[1] & 0xE0 == 0xE0):
        return "MP3"
    if head[0] == 0xFF and head[1] & 0xF6 == 0xF0:
        return "ADTS-AAC"
    signatures = {b"fLaC": "FLAC", b"OggS": "Ogg", b"\x1aE\xdf\xa3": "Matroska/WebM", b"0&\xb2u": "ASF/WMA"}
    if head[:4] in signatures:
        return signatures[head[:4]]
    if head[:5] == b"#!AMR":
        return "AMR"
    if head[:6] == b"<!DOCT" or head[:5] == b"<html":
        return "HTML (download error page)"
    return "UNKNOWN"


CONTAINER_EXT = {"QuickTime MOV": ".mov", "QuickTime MOV (legacy)": ".mov", "WAV": ".wav", "3GPP": ".3gp",
                 "MP3": ".mp3", "FLAC": ".flac", "Ogg": ".ogg", "Matroska/WebM": ".mkv", "AVI": ".avi",
                 "AMR": ".amr", "ASF/WMA": ".wma", "ADTS-AAC": ".aac"}


def container_ext(container: str) -> str:
    if container.startswith("MPEG-4"):
        return ".m4a" if "M4A" in container else ".mp4"
    return CONTAINER_EXT.get(container, "")


def ffmpeg_exe() -> str:
    """ffmpeg binary: the one bundled by imageio-ffmpeg (pinned), else PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("ffmpeg not found: pip install imageio-ffmpeg (see requirements.txt)")
        return exe


# Metadata keys that may identify people or places are never copied out of the
# raw videos (iPhones store GPS coordinates in com.apple.quicktime.location.*).
_PRIVATE_TAG = re.compile(r"(?i)location|gps|iso6709|author|artist|comment|title|description")


def probe_media(path: Path) -> dict:
    """Parse `ffmpeg -i` stream information (imageio-ffmpeg ships no ffprobe)."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    info = proc.stderr  # ffmpeg prints stream info to stderr and exits 1 without an output
    dur = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", info)
    if dur is None:
        raise RuntimeError(f"ffmpeg could not read {path.name}: {info.strip().splitlines()[-1] if info.strip() else 'no output'}")
    h, m, s = dur.groups()
    audio = re.search(r"Stream #\d+:\d+[^:]*: Audio: (\w+)[^,]*, (\d+) Hz, ([^,]+), (\w+)(?:, (\d+) kb/s)?", info)
    video = re.search(r"Stream #\d+:\d+[^:]*: Video: (\w+)", info)
    tags = {}
    for key in ("creation_time", "com.apple.quicktime.creationdate", "com.apple.quicktime.make",
                "com.apple.quicktime.model", "com.apple.quicktime.software", "encoder", "major_brand"):
        mt = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+)$", info, flags=re.M)
        if mt and not _PRIVATE_TAG.search(key):
            tags[key] = mt.group(1).strip()
    return {
        "duration_s": int(h) * 3600 + int(m) * 60 + float(s),
        "has_audio": audio is not None,
        "audio_codec": audio.group(1) if audio else "",
        "sample_rate": int(audio.group(2)) if audio else None,
        "channel_layout": audio.group(3).strip() if audio else "",
        "sample_format": audio.group(4) if audio else "",
        "audio_kbps": int(audio.group(5)) if audio and audio.group(5) else None,
        "has_video": video is not None,
        "video_codec": video.group(1) if video else "",
        "n_audio_streams": len(re.findall(r"Stream #\d+:\d+[^:]*: Audio:", info)),
        **tags,
    }


def decode_audio(path: Path, sample_rate: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    """Decode the first audio stream to float32 in [-1, 1] (channels x samples).

    sample_rate=None keeps the native rate; mono=False keeps the native channels.
    No filtering, normalisation or dithering is applied.
    """
    probe = probe_media(path)
    if not probe["has_audio"]:
        raise RuntimeError(f"{path.name} has no audio stream")
    sr = sample_rate or probe["sample_rate"]
    n_ch = 1 if mono else _channel_count(probe["channel_layout"])
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn",
           "-ac", str(n_ch), "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path.name}: {proc.stderr.decode(errors='replace')[-400:]}")
    samples = np.frombuffer(proc.stdout, dtype="<f4")
    return samples.reshape(-1, n_ch).T.copy(), sr


def _channel_count(layout: str) -> int:
    named = {"mono": 1, "stereo": 2, "2.1": 3, "quad": 4, "5.0": 5, "5.1": 6, "7.1": 8}
    if layout in named:
        return named[layout]
    m = re.match(r"(\d+) channels", layout)
    if m:
        return int(m.group(1))
    raise RuntimeError(f"unrecognised channel layout '{layout}'")


# ---- audio analysis constants (fixed before any label was examined) ------------------
TARGET_SR = 16_000            # processed WAV: 16 kHz, mono, 16-bit PCM
FRAME_S = 0.025               # 25 ms analysis window
HOP_S = 0.010                 # 10 ms hop
N_FFT = 512                   # FFT size for spectral features / MFCC at 16 kHz

# Target-utterance localisation (see docs/DOCUMENTATION.md, "Segmentation").
# The recordings contain continuous background babble, so the participant's word
# is located as the dominant short-time-energy event of the clip.
SEG_SMOOTH_FRAMES = 5         # 50 ms moving average of frame energy (dB), edge-replicated
SEG_FLOOR_PERCENTILE = 20     # background floor = 20th percentile of smoothed energy
SEG_MIN_REL_DB = 6.0          # threshold at least 6 dB above the floor ...
SEG_REL_FRACTION = 0.5        # ... or half-way (in dB) between floor and peak, if higher
SEG_MAX_GAP_S = 0.25          # bridge dips shorter than this (syllable / stop closures)
SEG_PAD_S = 0.05              # context kept on both sides of the located event
SEG_LONG_S = 1.5              # longer events probably contain more than one word
SEG_COMPETING_DB = 3.0        # another event within 3 dB of the peak = ambiguous target
SEG_COMPETING_SEP_S = 0.30    # ... if it lies at least 0.3 s outside the event
LOW_SBR_DB = 10.0             # dominant-event-to-background ratio below this = low quality
MIN_SYLLABLE_S = 0.10         # an event shorter than 0.10 s per syllable (>10 syllables/s) cannot hold the whole word
CLEARER_TAKE_DB = 3.0         # a later retake replaces the earliest only if its SBR is >= 3 dB higher
CLIP_FRACTION_LIMIT = 0.001   # >0.1 % of samples at full scale = clipping
BACKGROUND_VOICED_S = 0.30    # >=0.3 s of voicing outside the event = other voices present
MIN_VOICED_FRAMES = 5         # fewer voiced frames in the event = no usable speech

# Praat autocorrelation pitch (Boersma 1993) via Parselmouth. Parameters are given
# explicitly so a change of library defaults cannot silently change the features.
PITCH = dict(time_step=HOP_S, pitch_floor=75.0, max_number_of_candidates=15, very_accurate=False,
             silence_threshold=0.03, voicing_threshold=0.45, octave_cost=0.01,
             octave_jump_cost=0.35, voiced_unvoiced_cost=0.14, pitch_ceiling=500.0)


def frame_energy_db(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame RMS in dBFS (uncentred frames) and the centre time of each frame."""
    import librosa
    win, hop = int(FRAME_S * sr), int(HOP_S * sr)
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=False)[0]
    times = (np.arange(len(rms)) * hop + win / 2) / sr
    return 20 * np.log10(np.maximum(rms, 1e-10)), times


def locate_utterance(y: np.ndarray, sr: int) -> dict:
    """Locate the dominant energy event (the participant's word) in a clip.

    Deterministic and label-blind. Returns start/end in seconds plus the
    evidence used to judge how trustworthy the location is.
    """
    from scipy.ndimage import uniform_filter1d
    db, _ = frame_energy_db(y, sr)
    if len(db) < SEG_SMOOTH_FRAMES:
        raise ValueError("clip too short to analyse")
    smooth = uniform_filter1d(db, size=SEG_SMOOTH_FRAMES, mode="nearest")
    floor = float(np.percentile(smooth, SEG_FLOOR_PERCENTILE))
    pk = int(np.argmax(smooth))
    peak = float(smooth[pk])
    thr = floor + max(SEG_MIN_REL_DB, SEG_REL_FRACTION * (peak - floor))
    hop = int(HOP_S * sr)
    win = int(FRAME_S * sr)
    max_gap = int(round(SEG_MAX_GAP_S / HOP_S))

    def expand(step: int) -> int:
        i, edge, gap = pk, pk, 0
        while 0 <= i + step < len(smooth):
            i += step
            if smooth[i] >= thr:
                edge, gap = i, 0
            else:
                gap += 1
                if gap > max_gap:
                    break
        return edge

    left, right = expand(-1), expand(+1)
    duration = len(y) / sr
    event_start, event_end = left * hop / sr, (right * hop + win) / sr
    start = max(0.0, event_start - SEG_PAD_S)
    end = min(duration, event_end + SEG_PAD_S)
    sep = int(round(SEG_COMPETING_SEP_S / HOP_S))
    outside = smooth.copy()
    outside[max(0, left - sep):min(len(smooth), right + sep + 1)] = -np.inf
    competing = float(peak - outside.max()) if np.isfinite(outside.max()) else float("inf")
    return {
        "seg_start_s": round(start, 3), "seg_end_s": round(end, 3), "seg_duration_s": round(end - start, 3),
        "event_duration_s": round(event_end - event_start, 3),
        "peak_time_s": round((pk * hop + win / 2) / sr, 3), "sbr_db": round(peak - floor, 2),
        "competing_margin_db": round(competing, 2), "threshold_db": round(thr, 2), "floor_db": round(floor, 2),
        "touches_start": left == 0, "touches_end": right == len(smooth) - 1,
    }


def count_syllables(word: str) -> int | None:
    """Syllable nuclei of a Bisaya (Cebuano) word: one per vowel letter.

    Cebuano spelling is close to phonemic: y and w are glides, and adjacent vowels are separate syllables
    split by a glottal stop (maayo = ma-a-yo). Used only to check that a located event is long enough
    to contain the recorded word; never as a model input.
    """
    n = len(re.findall(r"[aeiou]", word.casefold())) if word else 0
    return n or None


def praat_pitch(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Praat autocorrelation F0 track: (frame times, F0 in Hz; 0 = unvoiced)."""
    import parselmouth
    snd = parselmouth.Sound(np.asarray(y, dtype=np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch_ac(**PITCH)
    return pitch.xs(), pitch.selected_array["frequency"]


# ---- derived classification targets (Phase 8) --------------------------------------
# Groupings of the participants' OWN 1-5 ratings, fixed before modelling. They are
# derived, not collected; the original ratings always stay in the data.
TARGETS = {
    "valence_5": {"source": "valence", "map": {i: str(i) for i in range(1, 6)}, "order": ["1", "2", "3", "4", "5"]},
    "arousal_5": {"source": "arousal", "map": {i: str(i) for i in range(1, 6)}, "order": ["1", "2", "3", "4", "5"]},
    "valence_3": {"source": "valence", "map": {1: "Unpleasant", 2: "Unpleasant", 3: "Neutral", 4: "Pleasant", 5: "Pleasant"},
                  "order": ["Unpleasant", "Neutral", "Pleasant"]},
    "arousal_3": {"source": "arousal", "map": {1: "Low", 2: "Low", 3: "Moderate", 4: "High", 5: "High"},
                  "order": ["Low", "Moderate", "High"]},
    "valence_2": {"source": "valence", "map": {1: "NotPleasant", 2: "NotPleasant", 3: "NotPleasant", 4: "Pleasant", 5: "Pleasant"},
                  "order": ["NotPleasant", "Pleasant"]},
    "arousal_2": {"source": "arousal", "map": {1: "NotHigh", 2: "NotHigh", 3: "NotHigh", 4: "High", 5: "High"},
                  "order": ["NotHigh", "High"]},
}
TASK_ROLE = {"arousal_3": "primary", "valence_2": "primary", "arousal_2": "secondary",
             "valence_3": "attempted", "valence_5": "attempted", "arousal_5": "attempted"}
MIN_CLASS_SIZE = 5            # a task is modelled only if every class has >= 5 participants
N_OUTER_FOLDS = 5
N_REPEATS = 10
N_PERMUTATIONS = 500


def derive_target(ratings, task: str) -> list:
    spec = TARGETS[task]
    return [spec["map"].get(int(r)) if r is not None and not (isinstance(r, float) and np.isnan(r)) else None
            for r in ratings]
