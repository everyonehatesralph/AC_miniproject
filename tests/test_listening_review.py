"""Blind listening review merge (src/01b_merge_listening_reviews.py): validation, agreement and decision rules."""

import math

import pandas as pd
import pytest

TAKES = pd.DataFrame([
    {"participant_code": "Y2-001", "drive_file_id": "a1", "review_take": 1, "selected": True,
     "auto_seg_start_s": 0.50, "auto_seg_end_s": 1.00},
    {"participant_code": "Y2-002", "drive_file_id": "b1", "review_take": 1, "selected": False,
     "auto_seg_start_s": 0.20, "auto_seg_end_s": 0.70},
    {"participant_code": "Y2-002", "drive_file_id": "b2", "review_take": 2, "selected": True,
     "auto_seg_start_s": 1.20, "auto_seg_end_s": 1.80},
    {"participant_code": "Y2-003", "drive_file_id": "c1", "review_take": 1, "selected": True,
     "auto_seg_start_s": 0.30, "auto_seg_end_s": 0.90},
])
SHEET = {"Y2-001": "Kapoy", "Y2-002": "Salamat", "Y2-003": ""}


@pytest.fixture(scope="module")
def m(script):
    return script("01b_merge_listening_reviews.py")


def answers_file(m, tmp_path, who, overrides=None, name=None):
    """One listener's file: every take answered 'boundaries OK', no word, unless overridden per take."""
    rows = []
    for t in TAKES.itertuples():
        row = {c: "" for c in m.ANSWER_COLUMNS}
        row.update({"reviewer_initials": who, "participant_code": t.participant_code, "take": t.review_take,
                    "drive_file_id": t.drive_file_id, "shown_start_s": f"{t.auto_seg_start_s:.3f}",
                    "shown_end_s": f"{t.auto_seg_end_s:.3f}", "whole_word_audible_Y_N": "Y", "boundaries_ok_Y_N": "Y",
                    "clarity_1_to_5": "4", "other_voice_over_word_Y_N": "N", "page_version": "test",
                    "saved_at": "2026-10-03T08:00:00.000Z"})
        row.update((overrides or {}).get(t.drive_file_id, {}))
        rows.append(row)
    path = tmp_path / (name or f"listening_review_{who}.csv")
    pd.DataFrame(rows, columns=m.ANSWER_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run(m, tmp_path, per_listener):
    files = [answers_file(m, tmp_path, who, o) for who, o in per_listener.items()]
    answers = m.load_answers(files, TAKES)
    return answers, m.decide(answers, TAKES, SHEET)


def outcome(outcomes, code, judgement, take=""):
    sel = outcomes[(outcomes["participant_code"] == code) & (outcomes["judgement"] == judgement)
                   & (outcomes["take"].astype(str) == str(take))]
    assert len(sel) == 1
    return sel["outcome"].iloc[0]


@pytest.mark.parametrize("typed, norm", [("Kapoy!", "kapoy"), (" ka-poy ", "kapoy"), ("'=Lisod", "lisod"), ("", "")])
def test_normalise_word(m, typed, norm):
    assert m.normalise_word(typed) == norm


def test_missing_word_filled_only_when_unanimous_and_a_protocol_word(m, tmp_path):
    both = {"c1": {"heard_word": "kapoy"}}
    _, (_, _, corr, outs) = run(m, tmp_path, {"AB": both, "CD": {"c1": {"heard_word": "Kapoy."}}})
    assert corr.to_dict("records") == [{"participant_code": "Y2-003", "field": "spoken_word", "new_value": "Kapoy",
                                        "source": "blind listening review, unanimous (2 listeners)",
                                        "verified_by": "AB;CD", "date_verified": "2026-10-03"}]
    _, (_, _, corr, outs) = run(m, tmp_path, {"AB": {"c1": {"heard_word": "hello"}}, "CD": {"c1": {"heard_word": "hello"}}})
    assert corr.empty and "not a protocol word" in outcome(outs, "Y2-003", "spoken word")
    _, (_, _, corr, outs) = run(m, tmp_path, {"AB": {"c1": {"heard_word": "kapoy"}}, "CD": {"c1": {"heard_word": "lisod"}}})
    assert corr.empty and "disagree" in outcome(outs, "Y2-003", "spoken word")


def test_one_listener_decides_nothing(m, tmp_path):
    _, (takes_out, segs, corr, outs) = run(m, tmp_path, {"AB": {"c1": {"heard_word": "kapoy"}}})
    assert takes_out.empty and segs.empty and corr.empty
    assert outcome(outs, "Y2-003", "spoken word").startswith("pending")


def test_word_different_from_the_sheet_is_reported_never_changed(m, tmp_path):
    o = {"a1": {"heard_word": "Lisod"}}
    _, (_, _, corr, outs) = run(m, tmp_path, {"AB": o, "CD": o})
    assert corr.empty and outcome(outs, "Y2-001", "spoken word").startswith("MISMATCH")
    o = {"a1": {"heard_word": "kapoy"}}
    _, (_, _, _, outs) = run(m, tmp_path, {"AB": o, "CD": o})
    assert outcome(outs, "Y2-001", "spoken word").startswith("confirmed")


def test_unanimous_take_choice_becomes_an_override(m, tmp_path):
    choose_1 = {"b1": {"chosen_take_Y_N": "Y"}, "b2": {"chosen_take_Y_N": "N"}}
    _, (takes_out, _, _, outs) = run(m, tmp_path, {"AB": choose_1, "CD": choose_1})
    assert takes_out[["participant_code", "drive_file_id"]].to_dict("records") == [
        {"participant_code": "Y2-002", "drive_file_id": "b1"}]
    assert "differs from the automatic choice" in outcome(outs, "Y2-002", "take choice")
    choose_2 = {"b1": {"chosen_take_Y_N": "N"}, "b2": {"chosen_take_Y_N": "Y"}}
    _, (takes_out, _, _, outs) = run(m, tmp_path, {"AB": choose_1, "CD": choose_2})
    assert takes_out.empty and "disagree" in outcome(outs, "Y2-002", "take choice")
    none = {"b1": {"chosen_take_Y_N": "N"}, "b2": {"chosen_take_Y_N": "N"}}
    _, (takes_out, _, _, outs) = run(m, tmp_path, {"AB": none, "CD": none})
    assert takes_out.empty and "no complete word" in outcome(outs, "Y2-002", "take choice")


def test_boundaries_corrected_only_within_50_ms(m, tmp_path):
    ab = {"a1": {"boundaries_ok_Y_N": "N", "corrected_start_s": "0.40", "corrected_end_s": "1.10"}}
    cd = {"a1": {"boundaries_ok_Y_N": "N", "corrected_start_s": "0.44", "corrected_end_s": "1.12"}}
    _, (_, segs, _, _) = run(m, tmp_path, {"AB": ab, "CD": cd})
    assert segs[["participant_code", "drive_file_id", "seg_start_s", "seg_end_s"]].to_dict("records") == [
        {"participant_code": "Y2-001", "drive_file_id": "a1", "seg_start_s": 0.42, "seg_end_s": 1.11}]
    cd = {"a1": {"boundaries_ok_Y_N": "N", "corrected_start_s": "0.50", "corrected_end_s": "1.12"}}
    _, (_, segs, _, outs) = run(m, tmp_path, {"AB": ab, "CD": cd})
    assert segs.empty and "disagree" in outcome(outs, "Y2-001", "boundaries", 1)


def test_answers_about_old_boundaries_are_ignored(m, tmp_path):
    old = {"a1": {"shown_start_s": "0.450", "boundaries_ok_Y_N": "N", "corrected_start_s": "0.40",
                  "corrected_end_s": "1.10"}}
    answers, (_, segs, _, outs) = run(m, tmp_path, {"AB": old, "CD": old})
    assert answers.loc[answers["drive_file_id"] == "a1", "boundary_stale"].all()
    assert segs.empty and "2 stale answer(s) ignored" in outcome(outs, "Y2-001", "boundaries", 1)


def test_confirmed_boundaries(m, tmp_path):
    _, (_, segs, _, outs) = run(m, tmp_path, {"AB": {}, "CD": {}})
    assert segs.empty and outcome(outs, "Y2-001", "boundaries", 1).startswith("confirmed")


@pytest.mark.parametrize("change, message", [
    ({"clarity_1_to_5": "7"}, "clarity_1_to_5 must be 1-5"),
    ({"boundaries_ok_Y_N": "yes"}, "must be Y, N or blank"),
    ({"corrected_start_s": "1.2", "corrected_end_s": "1.1"}, "not before its end"),
    ({"corrected_start_s": "abc"}, "must be a number"),
    ({"drive_file_id": "zz"}, "not in the current recordings"),
])
def test_invalid_answers_are_rejected(m, tmp_path, change, message):
    path = answers_file(m, tmp_path, "AB", {"a1": change})
    with pytest.raises(ValueError, match=message):
        m.load_answers([path], TAKES)


def test_one_file_per_listener(m, tmp_path):
    a = answers_file(m, tmp_path, "AB", name="first.csv")
    b = answers_file(m, tmp_path, "ab", name="second.csv")  # initials are case-insensitive
    with pytest.raises(ValueError, match="two answers files"):
        m.load_answers([a, b], TAKES)


def test_two_chosen_takes_rejected(m, tmp_path):
    path = answers_file(m, tmp_path, "AB", {"b1": {"chosen_take_Y_N": "Y"}, "b2": {"chosen_take_Y_N": "Y"}})
    with pytest.raises(ValueError, match="more than one chosen take"):
        m.load_answers([path], TAKES)


def test_agreement_statistics(m, tmp_path):
    ab = {"a1": {"heard_word": "kapoy"}, "c1": {"heard_word": "lisod"}}
    cd = {"a1": {"heard_word": "Kapoy"}, "c1": {"heard_word": "sayon"}}
    answers, _ = run(m, tmp_path, {"AB": ab, "CD": cd})
    agree = m.agreement(answers, TAKES).set_index("judgement")
    words = agree.loc["heard word (letters only, case ignored)"]
    assert words["n_items"] == 2 and words["percent_agreement"] == 50.0
    same = agree.loc["another voice over the word"]  # everybody answered N: kappa is undefined
    assert same["percent_agreement"] == 100.0 and math.isnan(same["cohen_kappa"])
