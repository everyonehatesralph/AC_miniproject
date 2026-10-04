"""Take selection (complete-word rule) and quality flags in src/03_prepare_audio.py."""

from datetime import datetime

import pytest


def take(fid, hhmmss, sbr, flags=(), usable=True):
    return {"drive_file_id": fid, "drive_name": f"{fid}.MOV", "capture_time": f"2026-10-01T{hhmmss}+0800",
            "sbr_db": sbr, "flags": list(flags), "usable": usable}


@pytest.fixture(scope="module")
def m(script):
    return script("03_prepare_audio.py")


def test_single_recording(m):
    a = take("a", "18:10:00", 20)
    assert m.select_take("Y2-001", [a], {}) == (a, "single recording")


def test_complete_word_beats_an_earlier_cut_off_take(m):
    early_cut = take("a", "18:10:00", 30, ["EDGE_START"])
    later_ok = take("b", "18:10:20", 10)
    chosen, reason = m.select_take("Y2-001", [early_cut, later_ok], {})
    assert chosen is later_ok and "1 of 2 takes hold a complete word" in reason


def test_too_short_for_word_counts_as_incomplete(m):
    fragment = take("a", "18:10:00", 25, ["TOO_SHORT_FOR_WORD"])
    whole = take("b", "18:10:30", 12)
    assert m.select_take("Y2-001", [fragment, whole], {})[0] is whole


@pytest.mark.parametrize("later_sbr, expect_later", [(13.0, True), (12.9, False)])
def test_later_take_replaces_earliest_only_if_3_db_clearer(m, later_sbr, expect_later):
    early, later = take("a", "18:10:00", 10.0), take("b", "18:10:30", later_sbr)
    chosen, _ = m.select_take("Y2-001", [later, early], {})  # input order must not matter
    assert chosen is (later if expect_later else early)


def test_take_without_competing_event_preferred(m):
    early_competing = take("a", "18:10:00", 20, ["COMPETING_EVENT"])
    later_clean = take("b", "18:10:30", 15)
    chosen, reason = m.select_take("Y2-001", [early_competing, later_clean], {})
    assert chosen is later_clean and "competing event dropped" in reason


def test_no_complete_take_keeps_all_usable(m):
    a, b = take("a", "18:10:00", 10, ["EDGE_START"]), take("b", "18:10:30", 11, ["EDGE_END"])
    chosen, reason = m.select_take("Y2-001", [a, b], {})
    assert chosen is a and "no take holds a complete word" in reason


def test_group_override_wins_and_unknown_id_fails(m):
    a, b = take("a", "18:10:00", 30), take("b", "18:10:30", 5)
    chosen, reason = m.select_take("Y2-001", [a, b], {"Y2-001": {"drive_file_id": "b", "reason": "heard", "decided_by": "AB"}})
    assert chosen is b and reason.startswith("group decision")
    with pytest.raises(ValueError, match="unknown file id"):
        m.select_take("Y2-001", [a, b], {"Y2-001": {"drive_file_id": "zzz", "reason": "", "decided_by": ""}})


def test_no_usable_take(m):
    a = take("a", "18:10:00", 30, usable=False)
    assert m.select_take("Y2-001", [a, take("b", "18:10:30", 5, usable=False)], {}) == (None, "no usable take")


def rec(**over):
    base = {"touches_start": False, "touches_end": False, "seg_duration_s": 0.6, "word_syllables": 2,
            "event_duration_s": 0.5, "competing_margin_db": 10.0, "sbr_db": 20.0, "clip_fraction": 0.0,
            "voiced_s_outside_segment": 0.0, "voiced_frames_in_segment": 30, "proc_duration_s": 3.0,
            "capture_time": "2026-10-01T18:10:00+0800", "orig_channels": "mono", "orig_audio_streams": 1,
            "longest_digital_silence_s": 0.0}
    return base | over


MEDIAN = datetime.fromisoformat("2026-10-01T18:12:00+08:00")


def test_clean_recording_has_no_flags(m):
    assert m.file_flags(rec(), MEDIAN, ("mono", 1)) == []


@pytest.mark.parametrize("event_s, flagged", [(0.19, True), (0.20, False)])
def test_too_short_for_word_threshold_is_0_10_s_per_syllable(m, event_s, flagged):
    assert ("TOO_SHORT_FOR_WORD" in m.file_flags(rec(event_duration_s=event_s), MEDIAN, ("mono", 1))) is flagged


def test_flags_for_edges_config_and_capture_outlier(m):
    flags = m.file_flags(rec(touches_start=True, orig_channels="stereo", capture_time="2026-10-02T03:10:00+0800"),
                         MEDIAN, ("mono", 1))
    assert {"EDGE_START", "RECORDING_CONFIG_DIFFERS", "CAPTURE_TIME_OUTLIER"} <= set(flags)


def test_unknown_word_never_raises_too_short(m):
    assert "TOO_SHORT_FOR_WORD" not in m.file_flags(rec(word_syllables=None, event_duration_s=0.01), MEDIAN, ("mono", 1))


@pytest.mark.parametrize("zeros_s, flagged", [(0.06, True), (0.04, False)])
def test_inserted_digital_silence_is_flagged(m, zeros_s, flagged):
    assert ("DIGITAL_SILENCE" in m.file_flags(rec(longest_digital_silence_s=zeros_s), MEDIAN, ("mono", 1))) is flagged


def test_converted_copy_without_capture_time(m):
    assert "NO_CAPTURE_TIME" in m.file_flags(rec(capture_time=""), None, ("mono", 1))
