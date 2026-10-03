"""Metadata steps: rating parsing (01), the CSV-export path (00), and the leakage guard (06)."""

import pandas as pd
import pytest

from common import AROUSAL_LABELS, VALENCE_LABELS


@pytest.mark.parametrize("raw, struck, expected", [
    ("4 — Pleasant", False, {"code": 4, "status": "VALID"}),
    ("4", False, {"code": 4, "status": "VALID"}),
    ("4 - pleasant", False, {"code": 4, "status": "VALID"}),
    ("4 — Unpleasant", False, {"code": 4, "status": "LABEL_MISMATCH"}),
    ("7", False, {"code": 7, "status": "OUT_OF_RANGE"}),
    ("", False, {"code": None, "status": "MISSING"}),
    ("happy", False, {"code": None, "status": "UNPARSEABLE"}),
    ("5 — Very Pleasant", True, {"code": 5, "status": "STRUCK_THROUGH"}),
])
def test_parse_rating(script, raw, struck, expected):
    audit = script("01_audit_data.py")
    assert audit.parse_rating(raw, VALENCE_LABELS, struck) == expected


def test_arousal_labels_are_checked_too(script):
    audit = script("01_audit_data.py")
    assert audit.parse_rating("5 — Very High", AROUSAL_LABELS, False)["status"] == "VALID"
    assert audit.parse_rating("5 — Very Pleasant", AROUSAL_LABELS, False)["status"] == "LABEL_MISMATCH"


@pytest.fixture(scope="module")
def x00(script):
    module = script("00_extract_metadata_pdf.py")
    if not module.DEFAULT_PDF.is_file():
        pytest.skip("the metadata PDF is not in this checkout")
    return module


@pytest.fixture(scope="module")
def pdf_rows(x00):
    df, _ = x00.extract(x00.DEFAULT_PDF)
    return df


def export_from(rows: pd.DataFrame, header, path):
    rows[header].to_csv(path, index=False, encoding="utf-8")
    return path


def test_csv_export_keeps_values_and_carries_the_strike_forward(x00, pdf_rows, tmp_path):
    path = export_from(pdf_rows, x00.EXPECTED_HEADER, tmp_path / "export.csv")
    df, prov = x00.from_csv_export(path, x00.DEFAULT_PDF)
    assert (df[x00.EXPECTED_HEADER] == pdf_rows[x00.EXPECTED_HEADER]).all().all()
    struck_pdf = pdf_rows.loc[pdf_rows["_struck_fields"] != "", ["Participant Code", "_struck_fields"]]
    struck_csv = df.loc[df["_struck_fields"] != "", ["Participant Code", "_struck_fields"]]
    assert struck_csv.reset_index(drop=True).equals(struck_pdf.reset_index(drop=True)) and len(struck_csv) >= 1
    assert prov["struck_cells_changed_in_export"] == []


def test_csv_export_uses_a_changed_struck_cell_and_logs_it(x00, pdf_rows, tmp_path):
    rows = pdf_rows.copy()
    i = rows.index[rows["_struck_fields"] != ""][0]
    field = rows.at[i, "_struck_fields"].split(";")[0]
    rows.at[i, field] = "3 — Moderate"
    df, prov = x00.from_csv_export(export_from(rows, x00.EXPECTED_HEADER, tmp_path / "export.csv"), x00.DEFAULT_PDF)
    assert (df["_struck_fields"] != "").sum() == 0
    assert prov["struck_cells_changed_in_export"][0]["export_value"] == "3 — Moderate"


def test_csv_export_refuses_unknown_columns_with_values(x00, pdf_rows, tmp_path):
    rows = pdf_rows[x00.EXPECTED_HEADER].copy()
    rows["Extra"] = ""
    rows.loc[rows.index[0], "Extra"] = "x"
    with pytest.raises(ValueError, match="outside the template"):
        x00.from_csv_export(export_from(rows, list(rows.columns), tmp_path / "export.csv"), x00.DEFAULT_PDF)


def test_csv_export_refuses_a_changed_header(x00, pdf_rows, tmp_path):
    rows = pdf_rows[x00.EXPECTED_HEADER].rename(columns={"Spoken Word": "Word"})
    with pytest.raises(ValueError, match="must be exactly"):
        x00.from_csv_export(export_from(rows, list(rows.columns), tmp_path / "export.csv"), x00.DEFAULT_PDF)


def test_leakage_guard_rejects_non_acoustic_columns(script):
    train = script("06_train_models.py")
    acoustic = {"f0_median", "rms_db_mean"}
    train.check_no_leakage(["f0_median", "rms_db_mean"], acoustic)
    for bad in (["f0_median", "spoken_word"], ["valence"], ["participant_code"]):  # forbidden even if listed
        with pytest.raises(RuntimeError, match="leakage"):
            train.check_no_leakage(bad, acoustic | set(bad))
    with pytest.raises(RuntimeError, match="leakage"):
        train.check_no_leakage(["mystery_feature"], acoustic)


def test_audit_ignores_listening_review_copies(script, tmp_path, monkeypatch):
    """Pipeline-made review copies and later-delivered copies are not supplied recordings (they once doubled every take)."""
    audit = script("01_audit_data.py")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "review" / "audio").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "Y2-041.mov").write_bytes(b"")
    (tmp_path / "data" / "review" / "audio" / "Y2-041_t1.wav").write_bytes(b"")
    (tmp_path / "data" / "supplied_audio" / "CS2B_v2").mkdir(parents=True)
    (tmp_path / "data" / "supplied_audio" / "CS2B_v2" / "Y2-041_Y2_PM_Kapoy.wav").write_bytes(b"")
    monkeypatch.setattr(audit, "SKIP_PATHS", {tmp_path / "data" / "review", tmp_path / "data" / "supplied_audio"})
    assert [p.name for p in audit.find_media(tmp_path)] == ["Y2-041.mov"]


def test_validator_never_shadows_its_logger(script):
    """A local variable named `log` inside main() once hid the logger and crashed the last line of the run."""
    validate = script("10_validate_project.py")
    assert "log" not in validate.main.__code__.co_varnames
