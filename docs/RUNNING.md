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
| 00 | `00_extract_metadata_pdf.py` | reads the metadata PDF into a CSV (no value changed) | seconds |
| 01 | `01_audit_data.py` | Phase 0 audit | seconds |
| 02 | `02_clean_metadata.py` | Phase 1 cleaning (raw values kept) | seconds |
| 03 | `03_prepare_audio.py` | audio QA, retake choice, WAV conversion, naming, quality report | ~30 s |
| 04 | `04_extract_features.py` | 108 acoustic features per participant | seconds |
| 05 | `05_eda.py` | exploratory analysis and figures | ~1.5 min |
| 06 | `06_train_models.py` | speaker-independent training and permutation tests | **~55–60 min** |
| 07 | `07_evaluate_models.py` | metrics, intervals, comparisons | ~40 s |
| 08 | `08_lexical_control.py` | same-word analysis | ~3 min |
| 09 | `09_build_final_report.py`, `09b_build_html_report.py` | final report + browser version | seconds |
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
  - `reports/validation_report.md`
  - `docs/DOCUMENTATION.md` (how everything works)
  - `docs/decision_log.md` (decisions still awaiting confirmation)

**Expected validation result:** `PASS 22, BLOCKED 2, WARNING 1, FAIL 0`.
- The two BLOCKED items are missing data that code cannot fix: fewer than 40 participants have recordings,
  and no consent documentation has been supplied.
- The WARNING is that the retake choices await the group's listening confirmation.

**Reproducibility:** with the pinned versions, the same machine type gives byte-identical audio files and
identical results. On a different operating system or CPU, the last decimals of some numbers can differ
(floating-point libraries).

## 6. The group's review tasks (human in the loop)

Code cannot do these. Each has a file the pipeline reads automatically on the next run:

| Task | Where | Then re-run |
|---|---|---|
| Confirm which retake to use (Y2-044, 047, 056, 057, 064) by listening, without looking at the ratings | `data/processed_audio/retake_review/` (README inside) → `data/processed_audio/take_overrides.csv` | `--skip-fetch --from 03` |
| Check each located word by listening; give corrected boundaries if needed (especially Y2-045, Y2-055) | `data/processed_audio/segment_review_sheet.csv` → `data/processed_audio/segment_overrides.csv` | `--skip-fetch --from 03` |
| Verified corrections (Y2-076 arousal, a word heard on a recording, …) | `data/metadata_corrections.csv` (copy the TEMPLATE) | `--skip-fetch --from 02` |
| Consent documentation | `consent/` (see `consent/README.md`) | `--skip-fetch --from 10` |

## 7. Working as a team with Git

- Pull before you start: `git pull`.
- Commit your inputs (review sheets, override and correction files) and push them.
- Let **one person** run the full pipeline afterwards and commit the regenerated outputs. Several people
  committing generated files at the same time causes conflicts.
- Recordings and WAV files are ignored by Git (`.gitignore`). Never force-add them: they are participants'
  personal data.

## 8. Troubleshooting

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
