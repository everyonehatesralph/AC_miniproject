"""Shared helpers in src/common.py: codes, sessions, syllables, derived targets, localisation, pitch, consent."""

import numpy as np
import pytest

import common
from common import (SEG_PAD_S, count_syllables, derive_target, locate_utterance, normalise_code, praat_pitch,
                    session_key)

SR = 16_000


@pytest.mark.parametrize("raw, code", [("Y2- 041", "Y2-041"), ("y2-041", "Y2-041"), ("Y2_041.MOV", "Y2-041"),
                                       ("Y2 041", "Y2-041"), ("Y2-0411", None), ("XY2-041", None),
                                       ("Y5-041", None), ("no code", None)])
def test_normalise_code(raw, code):
    assert normalise_code(raw) == code


def test_session_key_ignores_dash_type_case_and_spaces():
    assert session_key("Afternoon–Evening") == session_key("afternoon-evening") == session_key("Afternoon - Evening")
    assert session_key("Morning") != session_key("Afternoon-Evening")


@pytest.mark.parametrize("word, n", [("Kapoy", 2), ("Salamat", 3), ("maayo", 3), ("Lingaw", 2), ("", None), (None, None)])
def test_count_syllables_one_per_vowel_letter(word, n):
    assert count_syllables(word) == n


def test_derive_target_bins_and_missing():
    assert derive_target([1, 2, 3, 4, 5, None, float("nan")], "arousal_3") == \
        ["Low", "Low", "Moderate", "High", "High", None, None]
    assert derive_target([3, 4], "valence_2") == ["NotPleasant", "Pleasant"]
    assert derive_target([3, 4], "arousal_2") == ["NotHigh", "High"]


def tone(seconds: float, f0: float = 200.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return amp * (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(4 * np.pi * f0 * t) + 0.25 * np.sin(6 * np.pi * f0 * t)) / 1.75


def test_locate_utterance_finds_a_burst_in_quiet_noise():
    rng = np.random.default_rng(0)
    y = 0.001 * rng.standard_normal(3 * SR)
    y[SR:int(1.5 * SR)] += tone(0.5)
    loc = locate_utterance(y.astype(np.float32), SR)
    assert loc["seg_start_s"] == pytest.approx(1.0 - SEG_PAD_S, abs=0.04)
    assert loc["seg_end_s"] == pytest.approx(1.5 + SEG_PAD_S, abs=0.04)
    assert loc["event_duration_s"] == pytest.approx(0.5, abs=0.06)
    assert not loc["touches_start"] and not loc["touches_end"]
    assert loc["sbr_db"] > 30


def test_locate_utterance_flags_a_word_cut_off_at_the_start():
    rng = np.random.default_rng(1)
    y = 0.001 * rng.standard_normal(2 * SR)
    y[:int(0.4 * SR)] += tone(0.4)
    loc = locate_utterance(y.astype(np.float32), SR)
    assert loc["touches_start"] and loc["seg_start_s"] == 0.0


def test_praat_pitch_recovers_200_hz():
    times, f0 = praat_pitch(tone(1.0), SR)
    voiced = f0[f0 > 0]
    assert len(voiced) > 0.8 * len(f0)
    assert np.median(voiced) == pytest.approx(200.0, abs=1.0)


def test_consent_log_documented_only_with_date_and_both_yes(tmp_path, monkeypatch):
    log = tmp_path / "consent_log.csv"
    log.write_text(",".join(common.CONSENT_LOG_COLUMNS) + "\n"
                   "Y2-041,2026-09-30,AB,v1,Y,N,Y,\n"
                   "Y2- 042,2026-09-30,AB,v1,Y,Y,N,\n"
                   "Y2-043,,AB,v1,Y,Y,Y,\n"
                   "Y2-044,,,,,,,\n", encoding="utf-8")
    monkeypatch.setattr(common, "CONSENT_LOG", log)
    assert common.load_consent_log() == {"Y2-041": True, "Y2-042": False, "Y2-043": False, "Y2-044": False}


def test_consent_log_rejects_bad_values_and_duplicates(tmp_path, monkeypatch):
    log = tmp_path / "consent_log.csv"
    monkeypatch.setattr(common, "CONSENT_LOG", log)
    head = ",".join(common.CONSENT_LOG_COLUMNS) + "\n"
    log.write_text(head + "Y2-041,2026-09-30,AB,v1,yes,N,Y,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be Y, N or blank"):
        common.load_consent_log()
    log.write_text(head + "Y2-041,2026-09-30,AB,v1,Y,N,Y,\nY2-041,2026-09-30,AB,v1,Y,N,Y,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="listed twice"):
        common.load_consent_log()


def test_no_consent_log_means_not_yet_supplied(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "CONSENT_LOG", tmp_path / "absent.csv")
    assert common.load_consent_log() is None
