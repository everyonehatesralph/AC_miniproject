# Consent documentation (Protocol §VI, Expected Output 2)

**Status: nothing supplied yet.** The metadata records consent "Yes" for all 40 participants, but that is an
attestation, not documentation.

Put here only material that can be shared with the instructor without identifying participants:

1. **De-identified consent log** (CSV), one row per participant: `participant_code`, `consent_date`,
   `collector_initials`, `consent_form_version`, `voice_recording_consented (Y/N)`,
   `video_recording_consented (Y/N)`. The raw recordings are videos and show faces.
2. **The blank consent form / script** that was read to participants.
3. **Ethics / data-privacy clearance** for the study, if the institution issued one.

Keep the **signed forms** (they contain names) outside this project folder, as the instructor directs. The
audit counts files in this folder and reads participant codes from their names, but never prints a file name.

After adding files, re-run: `.venv/Scripts/python.exe run_pipeline.py --skip-fetch --from 10`
