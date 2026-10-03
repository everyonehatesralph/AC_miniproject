# Consent documentation (Protocol §VI, Expected Output 2)

**Status: no consent log yet.** The metadata records consent "Yes" for all 40 participants, but that is an
attestation, not documentation.

## What to do

1. Open `consent_log_TEMPLATE.csv`. It already lists every participant code (Y2-041 to Y2-080). Fill in, from
   the signed forms:

   | Column | What to write |
   |---|---|
   | `consent_date` | date of consent, YYYY-MM-DD |
   | `collector_initials` | initials of the group member who obtained consent |
   | `consent_form_version` | e.g. v1 |
   | `voice_recording_consented_Y_N` | Y or N |
   | `video_recording_consented_Y_N` | Y or N (the raw recordings are videos, so they show faces) |
   | `data_use_for_research_consented_Y_N` | Y or N |
   | `notes` | anything relevant, **never a name** |

2. Save it as **`consent/consent_log.csv`**: the same columns, codes only, **no names or student numbers**.
3. Re-run: `.venv\Scripts\python.exe run_pipeline.py --skip-fetch --from 03`.

## What the pipeline then does

- A participant counts as **documented** when the log has a consent date **and** `Y` for voice recording
  **and** `Y` for research data use.
- Undocumented participants are **excluded from the analysis automatically** (exclusion reason
  `CONSENT_NOT_DOCUMENTED`).
- The validator, the final report and its status matrix switch from BLOCKED to COMPLETE or INCOMPLETE,
  according to coverage.

## Privacy

Keep the **signed forms** (they contain names) outside this repository. Scans and photos placed in this folder
(`*.pdf`, `*.jpg`, `*.png`, `*.heic`, `*.docx`) are ignored by Git on purpose. An ethics/data-privacy clearance
letter without personal data can be shared with the instructor separately.
