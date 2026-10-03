# Data collection checklist (one page; print it and bring it)

Use this for **every new participant** (codes **Y2-081 onward**; agree the range with Group 4 first so no
student is recorded twice). Each rule fixes a problem found in the first 40 recordings.

## Before the session
- [ ] **Instructor approval** to collect more participants (see `docs/next_steps/01_email_to_instructor.md`).
- [ ] **Consent forms** printed, plus the scale card (valence 1–5, arousal 1–5) to show participants.
- [ ] **Record audio only, not video**: Voice Memos on iPhone, the voice recorder on Android. *Why: the old
      videos show faces, which nobody consented to and which the analysis never needs.*
- [ ] Phone audio quality: **highest/lossless** (iPhone: Settings → Voice Memos → Audio Quality → Lossless).
      **One phone, one setting, for everyone.** *Why: 6 old files used a different setting and shifted 20
      acoustic features.*
- [ ] Field sheet ready (columns of the Google Sheet), with the **session time boundary** agreed in advance.

## For each participant (in this order)
1. [ ] Explain the study; get **written consent first**. No consent, no recording.
2. [ ] Assign the next code (Y2-0NN) and write it on the consent form and the sheet.
3. [ ] Ask the class activity and note the subject, date, time and **your initials**.
4. [ ] Move to a **quiet spot**, away from the hallway crowd. *Why: background voices were present in 27 of 32
       old clips.*
5. [ ] Ask the prompt: *"Human sa imong klase, pilia ang usa ka pulong nga pinakaduol sa imong reaksyon karon.
       Isulti lang kini sa natural nga paagi."*
6. [ ] Start recording, **wait about 1 second**, let them say the word, **wait about 1 second**, stop.
       Phone about 20–30 cm from the mouth. *Why: 3 old clips started mid-word.*
7. [ ] **Say nothing** while recording, and don't coach: never "say it happily/sadly".
8. [ ] **One take only.** Retake only after a technical failure (cut off, loud noise), never for expression.
       Write the reason in Notes. *Why: 5 old participants had unexplained retakes.*
9. [ ] Immediately ask: *"How do you feel right now after your class?"* Write their exact words.
10. [ ] Show the scale card: **valence 1–5** and **arousal 1–5**. Write the numbers. Never cross out: if they
        change an answer, write the new one and note it. *Why: Y2-076's crossed-out rating had to be dropped.*
11. [ ] Fill every column: code, year, activity, subject, session, **spoken word**, feeling, valence, arousal,
        consent Y, date, collector initials, notes. *Why: 9 words, 2 feelings and all dates were missing.*

## After the session (same day)
- [ ] Rename each recording to its code (for example `Y2-081.m4a`) and upload it **unedited** (no trimming, no
      filters) to the group's Drive folder.
- [ ] Add the participants to the Google Sheet. Export it: **File → Download → CSV**. Save it as
      `data/metadata_raw/G3_metadata_export.csv`.
- [ ] For each new file, add a row to `data/raw/drive_manifest.csv`: `drive_file_id,drive_name,,`. The id is
      the part of the file's share link between `/d/` and `/view`.
- [ ] Add each participant to `consent/consent_log.csv` (codes only, **no names**).
- [ ] Run the whole pipeline: `.venv\Scripts\python.exe run_pipeline.py` (see `docs/RUNNING.md`).
