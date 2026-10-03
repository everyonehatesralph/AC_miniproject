"""Audit of audio supplied after the originals (src/03c_audit_supplied_audio.py)."""

import numpy as np
import pytest

SR = 16_000


@pytest.fixture(scope="module")
def m(script):
    return script("03c_audit_supplied_audio.py")


@pytest.mark.parametrize("name, parsed", [("Y2-073_Y2_PM_Kapoy.wav", ("Y2-073", "Kapoy")), ("Y2-050.wav", ("Y2-050", "")),
                                          ("y2- 041_Y2_AM_Salamat.wav", ("Y2-041", "Salamat")), ("notes.wav", (None, ""))])
def test_parse_name(m, name, parsed):
    assert m.parse_name(name) == parsed


def test_longest_zero_run(m):
    y = np.array([3, 0, 0, 5, 0, 0, 0, 0, 2, 0], dtype=np.int16)
    assert m.longest_zero_run(y) == 4
    assert m.longest_zero_run(np.array([1, 2, 3], dtype=np.int16)) == 0


@pytest.mark.parametrize("file_word, sheet, feeling, relation", [
    ("Kapoy", "Kapoy", "Tired", "same as the sheet"),
    ("kapoy", "Kapoy", "", "same as the sheet"),
    ("Salamat", "Nalipay", "Great", "differs from the sheet (Nalipay)"),
    ("Okay", "", "Okay", "sheet word blank; equals the FEELING text"),
    ("Grabe", "", "Amazed", "sheet word blank; source unknown"),
    ("", "Kapoy", "Pagod", "no word in the file name"),
])
def test_word_relation(m, file_word, sheet, feeling, relation):
    assert m.word_relation(file_word, sheet, feeling) == relation


@pytest.mark.parametrize("code, best, ncc, processed, expected", [
    ("Y2-041", "Y2-041", 0.99, False, "COPY_OF_ORIGINAL"),
    ("Y2-041", "Y2-042", 0.99, False, "OTHER_PARTICIPANT"),
    ("Y2-041", "Y2-041", 0.70, False, "UNCERTAIN_MATCH"),
    ("Y2-041", "Y2-041", 0.30, False, "DIFFERENT_RECORDING"),
    ("Y2-099", "Y2-041", 0.20, False, "NEW_RECORDING"),
    ("Y2-041", "Y2-041", 0.99, True, "PROCESSED_DELIVERY"),
])
def test_verdict(m, code, best, ncc, processed, expected):
    assert m.verdict(code, {"Y2-041", "Y2-042"}, best, ncc, processed) == expected


def signal(seconds, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    return 0.2 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.05 * rng.standard_normal(len(t))


def test_trimmed_copy_is_found_with_its_offset_and_gain(m):
    original = signal(4.0, 0)
    copy = 0.5 * original[int(1.25 * SR):int(2.75 * SR)]
    ncc, offset, gain = m.ncc_best(copy, original)
    assert ncc == pytest.approx(1.0, abs=1e-6) and offset == pytest.approx(1.25) and gain == pytest.approx(0.5, rel=1e-3)


def test_copy_longer_than_the_original(m):
    original = signal(1.0, 1)
    padded = np.concatenate([np.zeros(SR // 2), original, np.zeros(SR // 4)])
    ncc, offset, gain = m.ncc_best(padded, original)
    assert ncc == pytest.approx(1.0, abs=1e-6) and offset == pytest.approx(-0.5) and gain == pytest.approx(1.0, rel=1e-3)


def test_unrelated_recordings_do_not_match(m):
    rng = np.random.default_rng(7)
    a, b = rng.standard_normal(2 * SR), rng.standard_normal(3 * SR)
    assert abs(m.ncc_best(a, b)[0]) < m.UNRELATED_NCC
