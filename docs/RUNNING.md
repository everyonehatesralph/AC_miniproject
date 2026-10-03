# How to run the project (step by step, for every team member)

This guide takes you from nothing to the complete set of results on your own computer. It covers Windows,
macOS and Linux. Commands are shown first for **Windows (PowerShell)**, then for **macOS/Linux** where they
differ.

---

## 0. What you need

| Requirement | Details |
|---|---|
| Python **3.12** | <https://www.python.org/downloads/>. On Windows, tick **"Add python.exe to PATH"** in the installer. The project was built with 3.12.10. |
| Git | <https://git-scm.com/downloads> |
| Access to this repository | It is **private**: the owner must invite you (GitHub → Settings → Collaborators), and you must accept the invitation e-mail. |
| Internet | To install packages (about 750 MB) and to download the recordings (378 MB) from the group's Google Drive folder. |
| Disk space | About 1.5 GB in total. |
| Time | The full run takes **about 1 hour on a 16-core PC**, roughly 2 hours on 8 cores and 4 or more on 4 cores. Almost all of it is step 06 (permutation tests). The other steps take a few minutes. |

You do **not** need to install ffmpeg, Praat or anything else: everything comes from `requirements.txt`. That
includes a bundled ffmpeg (`imageio-ffmpeg`) and Praat (`praat-parselmouth`).

---

## 1. Get the code

```bash
git clone https://github.com/everyonehatesralph/AC_miniproject.git
cd AC_miniproject
```

If Git asks you to sign in, use your GitHub account (the one that was invited).

## 2. Create the Python environment (once)

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Check the installation (it should print `ok`):

```powershell
.venv\Scripts\python.exe -c "import librosa, parselmouth, sklearn, imageio_ffmpeg; print('ok')"
```

(macOS/Linux: replace `.venv\Scripts\python.exe` with `.venv/bin/python` here and everywhere below.)

## 3. The data

**Already in the repository:**
- the metadata (`G3 - mini project - TEmplate.pdf`, and its copy in `data/metadata_raw/`)
- the list of recordings (`data/raw/drive_manifest.csv`)
- all results produced so far

**Not in the repository, on purpose:** the **recordings** (videos show faces; voices are personal data).
Step 00 downloads them for you from the group's Google Drive folder, using the file list. The folder must stay
shared as "anyone with the link".

If the download fails, for example because Google limits downloads or sharing was changed:
1. Ask the person who already has the recordings to share their `data/raw` folder privately (a zip on the
   group Drive).
2. Unzip it so the `.mov` files are directly inside `data/raw/`.
3. Run the pipeline with `--skip-fetch` (step 4). The pipeline checks every file against its SHA-256 in
   `data/raw/download_log.csv`.

## 4. Run everything

```powershell
.venv\Scripts\python.exe run_pipeline.py
```

The run prints each step as it starts and finishes:

| Step | Script | What it does | Time (16 cores) |
|---|---|---|---|
| 00 | `00_fetch_recordings.py` | downloads the 38 recordings, checks their hashes | 2–3 min |
| 00 | `00_extract_metadata_pdf.py` | reads the metadata PDF (or a CSV export of the sheet) into a CSV; no value changed | seconds |
| 01 | `01_audit_data.py` | Phase 0 audit | seconds |
| 01b | `01b_merge_listening_reviews.py` | merges the listeners' answers (if any) into decisions | seconds |
| 02 | `02_clean_metadata.py` | Phase 1 cleaning (raw values kept) | seconds |
| 03 | `03_prepare_audio.py` | audio QA, retake choice, WAV conversion, naming, quality report | ~30 s |
| 03b | `03b_build_listening_review.py` | builds the listening-review page | seconds |
| 03c | `03c_audit_supplied_audio.py` | audits audio supplied later (e.g. the CS2B_v2 copies) against the originals | ~30 s |
| 04 | `04_extract_features.py` | 108 acoustic features per participant | seconds |
| 05 | `05_eda.py` | exploratory analysis and figures | ~1.5 min |
| 06 | `06_train_models.py` | speaker-independent training and permutation tests | **~55–60 min** |
| 07 | `07_evaluate_models.py` | metrics, intervals, comparisons | ~40 s |
| 08 | `08_lexical_control.py` | same-word analysis | ~3 min |
| 09, 09b, 09c | `09_build_final_report.py`, `09b_build_html_report.py`, `09c_build_presentation.py` | final report, browser version, slides | seconds |
| 10 | `10_validate_project.py` | automated checks | seconds |

Useful options:

| Command | Use |
|---|---|
| `run_pipeline.py --skip-fetch` | the recordings are already in `data/raw/` |
| `run_pipeline.py --skip-fetch --from 07` | re-run only the later steps (e.g. after editing a report script) |
| `run_pipeline.py --quick` | 5-minute installation test only (2 repeats, 20 permutations); see the warning below |

**Warning about `--quick`:** it overwrites `models/` and the reports with smoke-test numbers, and step 10 then
reports a FAIL on purpose. Never report those numbers. To get the real results back, run the full pipeline,
or restore the committed ones with `git checkout -- models reports figures data README.md`.

## 5. Look at the results

- **In a browser:** double-click `reports/html/index.html` (all reports, figures included, works offline).
  If you have not run the pipeline, build it first with `.venv\Scripts\python.exe src\09b_build_html_report.py`.
- **On GitHub:** open any `.md` file in `reports/`; GitHub shows the text and the figures.
- **Main documents:**
  - `reports/final_group_research_report.md` (the research report)
  - `reports/presentation/G3_results_presentation.pptx` (12 slides with speaker notes, built from the results)
  - `reports/validation_report.md`
  - `docs/DOCUMENTATION.md` (how everything works)
  - `docs/decision_log.md` (decisions still awaiting confirmation)

**Expected validation result:** `PASS 22, BLOCKED 2, WARNING 1, FAIL 0`.
- The two BLOCKED items are missing data that code cannot fix: fewer than 40 participants have recordings,
  and no consent documentation has been supplied.
- The WARNING is that the retake choices await a human listening decision (section 6). It turns into PASS
  once every retake participant has one.

**Reproducibility:** with the pinned versions, the same machine type gives byte-identical audio files and
identical results. On a different operating system or CPU, the last decimals of some numbers can differ
(floating-point libraries).

## 6. The group's review tasks (human in the loop)

Code cannot do these. Each has a file the pipeline reads automatically on the next run.

### 6.1 Blind listening review (at least two group members, independently)

1. Run the pipeline once, so the audio exists on your computer.
2. Double-click **`data/review/listening_review.html`**. It works offline in any browser.
   - It shows participant codes and take numbers only: no words, ratings or automatic choices, so you stay
     blind.
   - For each take you play the whole clip and the located word, then answer:
     - the word you hear;
     - whether the whole word is audible;
     - whether the whole word lies inside the blue band (if not, mark the real start and end);
     - whether another voice is over the word;
     - clarity, 1–5.
   - For the five participants with retakes, you also choose the take that holds the whole word most clearly.
3. Type your initials, answer everything, press **Save answers (CSV)**, and move the file from Downloads into
   **`data/review/answers/`**. Keep one file per person: replace your older file.
4. Commit and push your answers file, then re-run: `run_pipeline.py --skip-fetch --from 01b`.

Step 01b turns answers into decisions **only when every listener (at least two) gives the same answer**:
- the same take;
- corrected boundaries within 50 ms (their mean is used);
- the same *protocol* word for a word that is missing in the sheet (Y2-073).

A heard word that differs from the sheet is **reported, never changed**. Everything else is listed under "Needs a
joint decision" in `reports/listening_review_report.md`, together with the agreement between listeners
(percent agreement, Cohen's kappa).

### 6.2 The group's own decisions (these always win over the listening review)

| Task | File | Then re-run |
|---|---|---|
| A joint decision on a take | `data/processed_audio/take_overrides.csv`: `participant_code,drive_file_id,reason,decided_by` | `--skip-fetch --from 03` |
| Jointly agreed word boundaries | `data/processed_audio/segment_overrides.csv`: `participant_code,drive_file_id,seg_start_s,seg_end_s,reviewer,note` (times in seconds on the whole clip; `drive_file_id` may be blank only for a participant with one recording) | `--skip-fetch --from 03` |
| Verified corrections (Y2-076 arousal, a confirmed word, …) | `data/metadata_corrections.csv` (copy the TEMPLATE) | `--skip-fetch --from 02` |
| Consent documentation | `consent/consent_log.csv` (copy the TEMPLATE; see `consent/README.md`) | `--skip-fetch --from 03` (once the log exists, undocumented participants are excluded, which changes the analysis set) |

## 7. Adding new participants

Only with the instructor's approval, and with the checklist (`docs/data_collection_checklist.md`).

1. **Metadata.** Add the participants to the Google Sheet, then *File → Download → Comma-separated values*. Save
   the file as **`data/metadata_raw/G3_metadata_export.csv`**. From then on step 00 reads the cell values from
   this export instead of the PDF.
   - The header must stay exactly as in the template, and nothing may be written outside its columns (the run
     stops rather than drop anything).
   - Strikethrough cannot travel in a CSV, so the cells struck through in the PDF stay marked as long as they
     are unchanged.
2. **Recordings.** Upload each file unedited to the group's Drive folder. For each one, add a line to
   `data/raw/drive_manifest.csv`. The Drive file id is the part of the share link between `/d/` and `/view`.
3. **Consent.** Add each new code to `consent/consent_log.csv`.
4. **Run everything:** `run_pipeline.py` (without `--skip-fetch`, so the new recordings are downloaded). Then do
   the listening review for the new participants (6.1).

**Converted copies are not recordings.**
- Files that someone converted or trimmed (such as the `CS2B_v2` 16 kHz WAVs) are audited by step 03c
  (`reports/supplied_audio_audit.md`) but never analysed. Only the original phone files count.
- Denoised or otherwise cleaned versions are not kept at all; the protocol forbids denoising.
- To register a delivery, put its WAVs in `data/supplied_audio/<delivery name>/` (they stay out of Git) and add
  a row to `data/supplied_audio/deliveries.csv` (`kept = yes`).
- The audit shows, for each file:
  - which original take it was cut from;
  - whether it was edited (inserted silence, clipping, a cut-off word);
  - whether its file-name word agrees with the sheet.

## 8. Running the tests

The unit tests check the building blocks with synthetic signals and small tables. They need no recordings.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest
```

What they check:
- participant codes, sessions, syllables, derived targets;
- the word locator and the Praat pitch on synthetic signals;
- the consent rules, the take-selection rule and the quality flags;
- rating parsing, the CSV-export path and the leakage guard;
- every listening-review rule.

GitHub runs them on every push (`.github/workflows/tests.yml`; see the *Actions* tab). The tests never write to
`reports/logs/`.

## 9. Working as a team with Git

- Pull before you start: `git pull`.
- Commit your inputs (listening answers, override and correction files, the consent log) and push them.
- Let **one person** run the full pipeline afterwards and commit the regenerated outputs. Several people
  committing generated files at the same time causes conflicts.
- Recordings and WAV files are ignored by Git (`.gitignore`). Never force-add them: they are participants'
  personal data.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `py` is not recognised (Windows) | Install Python 3.12 from python.org with "Add to PATH", or use the full path to `python.exe`. |
| `pip install` fails building a package | Make sure the venv uses 64-bit Python 3.12, then upgrade pip (step 2). All pinned packages have ready-made wheels for Windows, macOS and Linux. |
| Step 00 fails: manifest not found | You are not on an up-to-date clone of the private repository; run `git pull`. |
| Step 00 fails to download | Google limits or sharing changed; use the fallback in step 3. |
| Step 03 says a raw file changed since download | A recording in `data/raw/` was modified or replaced; download it again (delete the file, re-run step 00). |
| Antivirus blocks ffmpeg | Allow `.venv/.../imageio_ffmpeg/binaries/ffmpeg-*` (it is the official static ffmpeg build). |
| Step 10 shows FAIL | Read `reports/validation_report.md`; each FAIL names the file and the reason (BLOCKED and WARNING do not stop the run). |
| It is slow | Step 06 needs about 1 hour on 16 cores. Use `--quick` only to test the installation. |
| The listening page plays nothing | The audio is created by a local run (step 03); the recordings are not in Git. Run the pipeline first, then open the page again. |
| Step 01b stops with an error about an answers file | The message names the file and the problem: two files from one listener, an unknown take, or an invalid value. Fix or replace that file. |
| Step 00 stops on `G3_metadata_export.csv` | The export's header differs from the template, or a value was typed outside its columns; the message says which. |
