> **Superseded (kept for the audit trail).** This v1 audit was written on 2026-10-03 *before* the recordings were supplied. Its statements about recordings ("0 media files", ceiling of 39/40) are no longer current. See `reports/phase0_data_audit.md` (v2, generated from the data) and `reports/data_quality_report.md`.

# Phase 0: Initial Data Audit (Group 3)

**Study:** Bisaya Post-Class Speech and Affective State Recognition Among Computer Science Students
**Group / population:** Group 3, Second-Year CS students (codes `Y2-041` to `Y2-080`)
**Audit date:** 2026-10-03
**Scope:** Phase 0 only. Nothing was cleaned, converted, extracted or modelled. No supplied file was modified.

| Input | SHA-256 (first 12) |
|---|---|
| `RESEARCH MINI-PROJECT v2.docx.pdf` (official protocol, 12 pages) | `07ef780a2088` |
| `G3 - mini project - TEmplate.pdf` (Group 3 metadata, Google Sheets PDF export, 2 pages) | `e4b8db1559a3` |
| `expected_output.jpg` (screenshot of protocol Section XVIII) | `3aca78b9e518` |
| `data/metadata_raw/G3_metadata_extracted_raw.csv` (verbatim extraction, generated) | `9b756707dec7` |

Labels used: **FACT**: read directly from a supplied file. **DATA**: computed by `src/01_audit_data.py`. **ASSUMPTION**: not established by the supplied material. **RECOMMENDATION**: proposed action. **MISSING**: information that was not supplied.

---

## Bottom line

1. **Genuinely valid observations right now: 0 of the required 40.** No recording of any participant was supplied (DATA).
2. **Even if all 40 recordings arrive and pass audio QA, the metadata supports at most 39.** Y2-076's arousal rating is struck through in the sheet and has no replacement (FACT). Resolving it gives **exactly 40, with zero margin**.
3. **Consent:** the sheet says "Yes" for all 40, but **no consent documentation** was supplied (FACT).
4. **16 of the 31 recorded spoken words are not on the protocol's 7-word list** (DATA). The protocol wording is ambiguous on whether that is allowed. If the instructor rules them deviations, at most 24 observations remain.
5. **9 rows have no spoken word and 2 have no self-reported feeling** (DATA). Most of these are in the Y2-071 to Y2-080 block, which also has no Subject.

---

## 1. Understanding of the study

**FACT (Protocol §I, VIII, IX, XV, XVII, XX):**

- Students are **not** asked to act an emotion. Right after a class, the participant is asked in Bisaya to *choose* one word closest to their reaction and say it naturally. The researcher must give no instruction about emotional expression.
- Right after the recording, the participant independently reports how they feel. This is an open description ("How do you feel right now after your class?") plus **Valence (1–5)** and **Arousal (1–5)**.
- **The self-report is the reference label**, not the word's meaning and not the researchers' impression of the voice ("Kapoy" ≠ automatically sad).
- The **primary model** uses acoustic/prosodic features only. Spoken word, year level, class activity and session are contextual variables, not predictors.
- Evaluation must be **speaker-independent**: no participant's recordings may appear in both training and test data.
- A **lexically controlled analysis** compares recordings of the *same* word with different self-reports.
- Each group contributes ≥40 valid observations from its assigned year level; the class targets ≥280.

## 2. Research objective (Group 3 scope)

The objective is to deliver ≥40 valid, consented, protocol-compliant observations from second-year CS students. Within them, the study tests speaker-independently whether acoustic/prosodic features of the spoken word carry information about self-reported valence and arousal. Lexical-control and educational-context analysis are reported where the data allow. RQ2 (affect by year level, activity and session) cannot be answered inside Group 3 (see §4.3); it needs the consolidated class dataset.

## 3. Required outputs (Protocol §XVIII) and current status

| # | Deliverable | Status | What is missing |
|---|---|---|---|
| 1 | ≥40 valid participant observations | **BLOCKED** | 0 valid; no recordings; ceiling 39 until Y2-076 is resolved |
| 2 | Required consent documentation | **BLOCKED** | Nothing supplied; `consent/` is empty |
| 3 | Properly coded research metadata | INCOMPLETE | 40 rows audited; Phase 1 cleaning pending instructor rulings (§13) |
| 4 | Properly named audio recordings | **BLOCKED** | No recordings; 9 rows lack the spoken word needed for the filename |
| 5 | Data quality report | INCOMPLETE | Metadata QA done (this report); audio QA impossible without recordings |
| 6 | Extracted acoustic feature dataset | **BLOCKED** | No recordings; librosa/soundfile/ffmpeg not installed |
| 7 | Exploratory analysis | INCOMPLETE | Metadata-only description possible now; acoustic EDA blocked |
| 8 | Machine learning experiment | **BLOCKED** | Needs #1 and #6 |
| 9 | Model evaluation | **BLOCKED** | Needs #8 |
| 10 | Interpretation of findings | **BLOCKED** | Needs #9 |
| 11 | Research limitations | INCOMPLETE | Several already identified in this audit |
| 12 | Group research report | **BLOCKED** | Needs #1 to #11 |

---

## 4. Metadata audit

### 4.1 Source and extraction

- **FACT:** The metadata exists only as a PDF export of a Google Sheet (PDF creator "Google Sheets", title "G3 - mini project"): 14 columns, 40 data rows, codes Y2-041 to Y2-080 with no gaps. The file is named "Template" but is filled in. Its single strikethrough suggests it is a working copy, not a final version.
- **DATA:** `src/00_extract_metadata_pdf.py` rebuilt every cell from the PDF's characters, locating cells by the drawn table grid. Empty cells stay empty, and spaces and dashes are preserved exactly.
- **Verification:** The extraction was compared with an independent visual transcription made from the rendered pages beforehand: **240 cells, 0 mismatches.** No text sits outside the grid. There is no coloured text and no cell fill. Exactly **one** strikethrough exists in the document.
- **Limitation:** A PDF export cannot carry cell comments, edit history or hidden columns.

### 4.2 Field completeness

| Field | Required by §XI? | Missing | Rows |
|---|---|---|---|
| Participant Code | yes | 0 | — |
| Year Level | yes | 0 | — |
| Class Activity | yes | 0 | — |
| Session | yes | 0 | — |
| **Spoken Word** | yes | **9** | Y2-053, 071, 073, 074, 076, 077, 078, 079, 080 |
| **Self-Reported Feeling** | yes | **2** | Y2-071, Y2-080 |
| Valence (1-5) | yes | 0 | — |
| Arousal (1-5) | yes | 0 (1 struck through) | Y2-076 |
| **Audio Filename** | yes ("corresponding WAV recording") | **40** | all |
| Subject | no (template field) | 10 | Y2-071 to Y2-080, a contiguous block |
| Consent Obtained (Y/N) | template field | 0 | — |
| Date Collected | template field | 40 | all |
| Collector Initials | template field | 40 | all |
| Notes | template field | 40 | all (blank is acceptable) |

### 4.3 Values and coding (FACT/DATA)

| Field | Values found | Issue | Severity |
|---|---|---|---|
| Participant Code | 39 × `Y2-0NN`, 1 × `"Y2- 041"` | Space inside Y2-041; Phase 1 candidate `Y2-041` (raw kept) | MINOR |
| Year Level | `2nd year` × 40 | Matches Group 3's assignment; code prefix Y2 agrees | OK |
| Class Activity | `Lecture` × 40 | Not an exact §VII category; candidate `Lecture/Discussion` (**confirm**) | MINOR |
| Session | `Afternoon-Evening` × 40 | Hyphen vs the protocol's en dash; cosmetic | MINOR |
| Subject | `Environmental Science` × 30 (Y2-041 has a trailing space), blank × 10 | See §4.7 | REVIEW |
| Consent | `Yes` × 40 | Header asks for Y/N; candidate `Y` | MINOR |
| Valence / Arousal | `n — Label` | Every code is in 1–5 and every label matches its code | OK |
| Edge whitespace | Y2-041 Subject, Y2-049 word `"Okay "`, Y2-070 word `"Kapoy "`, Y2-067 valence | `"Kapoy "` would count as a different word from `"Kapoy"` if not stripped | MINOR |

**Constant context variables (DATA):** Year Level, Class Activity and Session each have **one value** across all 40 rows, and Subject has one value or is blank. **Year Level × Affect, Class Activity × Affect and Session × Affect therefore cannot be analysed with Group 3 data.** Those questions need the consolidated class dataset.

### 4.4 Self-reported affect (DATA, raw 1–5 values)

| Rating | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Valence (n=40) | 1 | 2 | 15 | 11 | 11 |
| Arousal, valid (n=39) | 3 | 6 | 13 | 12 | 5 (+1 struck: Y2-076) |

Valence × arousal (valid arousal only, n=39):

| V \ A | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1 | 1 | | | | |
| 2 | 2 | | | | |
| 3 | | 5 | 7 | 3 | |
| 4 | | 1 | 5 | 4 | 1 |
| 5 | | | 1 | 5 | 4 |

Preview for Phase 8 only; no target has been chosen yet:

- Only **3 rows** have valence ≤ 2. Five-class valence is not viable, and even a derived negative class (1–2) would hold 3 cases. Derived 3-level counts are **valence 3 / 15 / 22** and **arousal 9 / 13 / 17**. These are *derived* groupings and were not collected.
- Valence and arousal move together in this sample. All three rows with valence ≤ 2 have arousal 1, and no valence-5 row has arousal below 3.

### 4.5 Spoken words (DATA)

- **31 of 40** rows have a spoken word, in **14 distinct forms** (after stripping trailing spaces).
- **On the §VIII list (15 rows):** Kapoy 10, Salamat 3, Lingaw 2.
- **Off the list (16 rows):** Nalipay 3, Katulgon 2, Gutom 2, Nalingaw 2, Lipay 1, Malipay 1, Maayo 1, Milamdag 1, Liboga 1, Hangak 1, and **Okay 1 (English)**.
- **Never chosen:** Ambot, Grabe, Lisod, Sayon.
- **ASSUMPTION (confirm with a Bisaya-speaking member):** Nalingaw shares the root of the list word *Lingaw*, and Lipay/Malipay/Nalipay share a root. Whether to pool variants is a Phase 1/13 decision. Strict lexical control should use exact forms, because different forms differ acoustically.
- **Lexical-control feasibility:** only **Kapoy** recurs enough (n=10: valence 1×1, 3×7, 4×2; arousal 1×1, 2×2, 3×6, 4×1). Every other form has ≤3 cases, so the §XVII analysis will be descriptive, not inferential.

### 4.6 Self-reported feeling (DATA)

- **38 of 40** present; missing for Y2-071 and Y2-080.
- **16 rows** have feeling text identical to the spoken word: Kapoy/Kapoy × 7, Gutom/Gutom × 2, and Maayo, Milamdag, Okay, Nalingaw, Katulgon, Nalipay and Hangak once each.
- Five rows are identical across word, feeling, valence and arousal (Kapoy / Kapoy / 3 / 3): Y2-042, 048, 052, 063, 070.
- These may be genuine, but this is also the pattern a collector copying the word into the feeling column would produce. §IX requires an *independent* report. **REVIEW, no change.**
- Spellings are kept as written (e.g. "Knowledgable", Y2-054).

### 4.7 Row-level anomalies needing a human decision

| Row(s) | Finding | Why it matters | Safest handling |
|---|---|---|---|
| **Y2-076** | Arousal `5 — Very High` is **struck through** (a vector line in the PDF, confirmed at 8× zoom). No replacement value. Spoken word also blank. | Unclear whether the rating was retracted. | **BLOCKER for arousal.** The collector checks the original record. If confirmed, note it. If corrected, enter the value with a note. If unknown, exclude from arousal analyses. Never guess a value. |
| **Y2-071 to Y2-080** | Subject blank for exactly this contiguous block. 8 of the 9 missing words and both missing feelings are here. | Looks like a separate batch or collector; the class context is unverified. | Confirm these 10 were collected right after the same Environmental Science lecture (afternoon–evening). If not, record the real context. |
| Y2-053, 071, 073, 074, 076–080 | Spoken word blank. For 073/074 the feeling is "Kapoy"; for 076–079 the feeling is "Okay", "Enlightened", "Kaulion", "Amazed". | Blocks the §XII filename and lexical analysis. | **Do not copy the feeling into the word.** Verify only by listening to the recording (§13). |
| Y2-071, Y2-080 | Ratings present; no description and no word. | §V requires a completed self-report. | Instructor ruling (§13). Recover only from an original record, and note the source. |
| Y2-056 | Feeling "Confused" with valence 5 (Very Pleasant). | Logical-consistency check only; it is plausible. | Keep as recorded unless the original record shows a transcription error. |

---

## 5. Recording / file audit

- **FACT/DATA: 0 media files.** The whole project folder was searched recursively for 17 audio/video extensions, and `data/raw/` is empty.
- The Google Drive account connected to the analysis session was searched read-only by title, full text and media type. **No Group 3 sheet, folder or recording was found.** The group's files are probably in another account.
- Format, codec, duration, sample rate, channels, peak level, clipping, silence, speech presence and background speakers are **not assessable**.
- **Tooling (DATA):** `ffmpeg`/`ffprobe` are not on PATH, and `librosa`/`soundfile` are not installed. All are needed in Phase 3.
- The audit's media path was tested in a scratch copy of the project on 9 synthetic media files, plus a mapping CSV and a dummy consent file. The media files were a valid WAV, MOV and M4A containers, duplicate takes with identical content, a truncated WAV, an MP4 named `.wav`, a code not in the metadata, a file outside `data/raw/`, and a code-less file resolved through `recording_map.csv`. Every case was detected as intended, and the real `data/raw/` was not touched.

## 6. Metadata-to-recording matching table (participant QA table)

Full table: `reports/phase0/participant_qa_table.csv`.

These apply to **every** row and are not repeated below: no recording; Audio Filename blank; Class Activity "Lecture" not an exact category; Session dash variant; consent coded "Yes"; Date Collected and Collector Initials blank. The audio QA fields (readable, duration, sample rate, channels, clipping, silence) are `NA_NO_RECORDING` for all 40. Processed WAV and feature row: none (Phases 3 and 6).

| Code | Spoken word | Self-reported feeling | V | A | Recording | Excluded now because | Row-specific flags |
|---|---|---|---|---|---|---|---|
| Y2-041 | Katulgon | Kapoy | 3 | 3 | none | NO_RECORDING | code written "Y2- 041"; off-list word; trailing space in Subject |
| Y2-042 | Kapoy | Kapoy | 3 | 3 | none | NO_RECORDING | feeling = word |
| Y2-043 | Lipay | Chill | 4 | 4 | none | NO_RECORDING | off-list word |
| Y2-044 | Salamat | Relief | 4 | 4 | none | NO_RECORDING | — |
| Y2-045 | Maayo | Maayo | 4 | 3 | none | NO_RECORDING | off-list word; feeling = word |
| Y2-046 | Salamat | Charismatic | 5 | 4 | none | NO_RECORDING | — |
| Y2-047 | Milamdag | Milamdag | 5 | 4 | none | NO_RECORDING | off-list word; feeling = word |
| Y2-048 | Kapoy | Kapoy | 3 | 3 | none | NO_RECORDING | feeling = word |
| Y2-049 | Okay | Okay | 4 | 3 | none | NO_RECORDING | off-list word; feeling = word; trailing space in word |
| Y2-050 | Kapoy | Pagod | 3 | 2 | none | NO_RECORDING | — |
| Y2-051 | Gutom | Gutom | 4 | 3 | none | NO_RECORDING | off-list word; feeling = word |
| Y2-052 | Kapoy | Kapoy | 3 | 3 | none | NO_RECORDING | feeling = word |
| Y2-053 | — | Impatient | 3 | 2 | none | NO_RECORDING | **word missing** |
| Y2-054 | Nalipay | Knowledgable | 4 | 5 | none | NO_RECORDING | off-list word |
| Y2-055 | Nalingaw | Nalingaw | 4 | 3 | none | NO_RECORDING | off-list word; feeling = word |
| Y2-056 | Liboga | Confused | 5 | 3 | none | NO_RECORDING | off-list word |
| Y2-057 | Katulgon | Katulgon | 3 | 2 | none | NO_RECORDING | off-list word; feeling = word |
| Y2-058 | Kapoy | Exhausted | 3 | 2 | none | NO_RECORDING | — |
| Y2-059 | Kapoy | Good | 4 | 4 | none | NO_RECORDING | — |
| Y2-060 | Gutom | Gutom | 3 | 3 | none | NO_RECORDING | off-list word; feeling = word |
| Y2-061 | Malipay | Nalipay | 5 | 4 | none | NO_RECORDING | off-list word |
| Y2-062 | Nalingaw | Enjoy | 5 | 5 | none | NO_RECORDING | off-list word |
| Y2-063 | Kapoy | Kapoy | 3 | 3 | none | NO_RECORDING | feeling = word |
| Y2-064 | Nalipay | Nalipay | 5 | 5 | none | NO_RECORDING | off-list word; feeling = word |
| Y2-065 | Lingaw | Amazed | 3 | 4 | none | NO_RECORDING | — |
| Y2-066 | Salamat | Learned | 5 | 4 | none | NO_RECORDING | — |
| Y2-067 | Kapoy | Kapoy | 1 | 1 | none | NO_RECORDING | feeling = word; trailing space in valence |
| Y2-068 | Lingaw | Good | 5 | 4 | none | NO_RECORDING | — |
| Y2-069 | Kapoy | Kapoy | 4 | 3 | none | NO_RECORDING | feeling = word |
| Y2-070 | Kapoy | Kapoy | 3 | 3 | none | NO_RECORDING | feeling = word; trailing space in word |
| Y2-071 | — | — | 4 | 4 | none | NO_RECORDING | subject blank; **word missing**; **feeling missing** |
| Y2-072 | Nalipay | Great | 5 | 5 | none | NO_RECORDING | subject blank; off-list word |
| Y2-073 | — | Kapoy | 2 | 1 | none | NO_RECORDING | subject blank; **word missing** |
| Y2-074 | — | Kapoy | 2 | 1 | none | NO_RECORDING | subject blank; **word missing** |
| Y2-075 | Hangak | Hangak | 3 | 2 | none | NO_RECORDING | subject blank; off-list word; feeling = word |
| Y2-076 | — | Okay | 5 | 5 ~~struck~~ | none | NO_RECORDING, AROUSAL_STRUCK_THROUGH | subject blank; **word missing** |
| Y2-077 | — | Enlightened | 4 | 2 | none | NO_RECORDING | subject blank; **word missing** |
| Y2-078 | — | Kaulion | 3 | 4 | none | NO_RECORDING | subject blank; **word missing** |
| Y2-079 | — | Amazed | 3 | 4 | none | NO_RECORDING | subject blank; **word missing** |
| Y2-080 | — | — | 5 | 5 | none | NO_RECORDING | subject blank; **word missing**; **feeling missing** |

## 7. Missing participant codes

- **DATA:** None. The assigned range Y2-041 to Y2-080 is complete, with 40 consecutive codes.
- **Missing recordings: 40 of 40.**
- **MISSING:** How the range Y2-041 to Y2-080 was allocated relative to Group 4, the other Year-2 group. This is not documented.

## 8. Duplicate participant codes

- **DATA:** No duplicate codes in the metadata (40 unique, both raw and after normalising Y2-041).
- **Duplicate recordings:** not assessable (no files).
- **Cross-group duplicates:** Group 4 also samples second-years, and §IV requires the groups to coordinate. Whether any student was recorded by both groups **cannot be verified from pseudonymous codes**. Only the groups' private consent records can show it.

## 9. Missing / invalid fields: consolidated

| Severity | Count | Issue (rows) |
|---|---|---|
| BLOCKER | 40 | NO_RECORDING (all) |
| BLOCKER | 1 | AROUSAL_STRUCK_THROUGH (Y2-076) |
| BLOCKER | 2 | dataset: NO_MEDIA_FILES_SUPPLIED, CONSENT_DOCUMENTATION_NOT_SUPPLIED |
| MAJOR | 40 | AUDIO_FILENAME_MISSING (all) |
| MAJOR | 16 | WORD_NOT_IN_PROTOCOL_LIST |
| MAJOR | 9 | SPOKEN_WORD_MISSING |
| MAJOR | 2 | SELF_REPORT_TEXT_MISSING (Y2-071, Y2-080) |
| MAJOR | 1 | dataset: NO_MARGIN_ABOVE_MINIMUM |
| MINOR | 125 | activity label (40), session dash (40), consent "Yes" vs "Y" (40), code whitespace (1), edge whitespace (4) |
| REVIEW | 108 | date missing (40), collector missing (40), subject blank (10), feeling = word (16), session-token assumption (1), PDF-only source (1) |
| INFO | 3 | constant Year Level, Class Activity, Session |

**Out-of-range values: none.** No valence or arousal value lies outside 1–5, and no label contradicts its number.

## 10. Naming and format violations

- **Audio Filename is blank in all 40 rows** (§XI requires it). There are no files, so no file-level naming or format checks were possible.
- **ASSUMPTION:** The §XII example `Y2-017_Y2_PM_Grabe.wav` implies `PM` = Afternoon–Evening (and presumably `AM` = Morning), but the protocol never defines the tokens. On that assumption, **31 rows** could be named now, e.g. `Y2-042_Y2_PM_Kapoy.wav`. The **9 rows** without a verified word cannot be named.
- **Leakage note:** the convention embeds the spoken word in the filename, so filenames and file IDs must never reach the model.

## 11. Consent and privacy audit

- **FACT:** Consent Obtained = `Yes` for 40/40.
- **MISSING:** Consent documentation (forms or consent log). Also missing is evidence of the institutional ethics and data-privacy clearance that §VI requires *before* collection, because publication is intended.
- **DATA:** `consent/` is empty. **Deliverable 2: BLOCKED.**
- **FACT:** No names or student IDs appear in the metadata, and the codes are pseudonymous. ✓
- **RECOMMENDATION:** Keep signed forms *outside* the dataset folder, because they identify participants. In `consent/`, keep only shareable material: a de-identified consent log (code, consent date, collector, form version, voice-recording consent), the blank form and the clearance letter.
- **RECOMMENDATION:** If the raw recordings are videos, they show faces, while §VI's consent explanation mentions only voice recording. Confirm that consent covered video, and keep only the extracted audio in the analysis dataset.

## 12. Genuinely valid observations

A valid observation needs recorded consent, exactly one recording that passes audio QA, and valid valence *and* arousal.

| Scenario | Count | Excluded |
|---|---|---|
| **Now** | **0** | all 40: no recording |
| All 40 recordings arrive and pass QA | 39 | Y2-076 (arousal struck) |
| …and Y2-076's arousal confirmed | 40 | none: **zero margin** |
| Strict reading: description text required | 37 | Y2-071, Y2-076, Y2-080 |
| Off-list words ruled protocol deviations | ≤ 24 | 16 off-list rows, plus any blank-word row that proves off-list |
| Complete on every §XI field except filename | 31 | 9 rows without a word (recoverable by listening) |

**Group 3 does not meet the 40-observation minimum today. Under no scenario can the current metadata exceed 40.**

---

## 13. Issues to fix before feature extraction

**Hard blockers: feature extraction cannot start until these are resolved.**

1. **Recordings for all 40 participants**, unedited originals including any failed takes, placed in `data/raw/`. If filenames carry no participant code (e.g. `IMG_0123.MOV`), fill `data/raw/recording_map.csv` (`original_file,participant_code`) from the collection log. The mapping must not be guessed from file order.
2. **Tooling:** ffmpeg (for MOV/MP4), librosa and soundfile.
3. **Duplicates** resolved for any participant with more than one file (copy, retake, failed take). The reason for each kept file must be documented.
4. **Consent records confirmed to exist** for every participant whose recording is processed. This is an ethical precondition, not a technical one.

**Must be resolved before modelling (features can be extracted meanwhile).**

5. **Y2-076 arousal:** check against the original record.
6. **Instructor rulings:**
   - (a) Are off-list words acceptable?
   - (b) Is the open description required (Y2-071, Y2-080)?
   - (c) Does "Lecture" mean "Lecture/Discussion"?
   - (d) Is the AM/PM session token correct?
   - (e) What is the Morning / Afternoon–Evening time boundary? §VII requires it to be defined before collection; it is not in the supplied material.
   - (f) May a partially valid observation (e.g. valence only) be used?
7. **Spoken words for the 9 blank rows**, verified by a Bisaya-speaking member *listening to the recording*, ideally without seeing the affect columns. Log who verified each one.
8. **Context of the Y2-071 to Y2-080 block** confirmed: same class, subject, activity and session?

**On the off-list ruling (context for the instructor):**

- *Interpretation:* §VIII says "pilia" (*choose*) and lists 7 words, then adds "Any response is acceptable. There is no correct or preferred word."
- The closed-list reading fits the design better: the lexical-control analysis needs words to recur. Still, the text is genuinely ambiguous, and Group 3's collectors evidently read it as open.
- *Why it matters beyond compliance:* word identity is partly recoverable from MFCCs and duration, and participants pick words that match their affect. An "acoustic-only" model can therefore still exploit lexical content as a proxy for affect. More distinct words widen this confound, so the §XVII lexical-control analysis must be reported whatever the ruling.

## 14. Next actions in priority order

| # | Owner | Action |
|---|---|---|
| 1 | Group | Put **all original recordings** in `data/raw/` (+ `recording_map.csv` if needed). I then re-run the audit on them. |
| 2 | Y2-076's collector | Check Y2-076's arousal against the original record. |
| 3 | Group → instructor | Obtain written rulings 6(a) to 6(f). |
| 4 | Group | Confirm consent forms exist for all 40. Supply a de-identified consent log and the ethics/data-privacy clearance. |
| 5 | Bisaya-speaking member | After step 1, transcribe the spoken word for the 9 blank rows by listening, without seeing the affect columns. |
| 6 | Group | Supply the original Google Sheet as XLSX/CSV. Fill Date Collected and Collector Initials from records. Confirm the Y2-071 to Y2-080 context. |
| 7 | Collectors | Confirm the 16 "feeling = word" rows were elicited separately after the recording. |
| 8 | Group + Group 4 | Confirm no student was recorded by both groups and that code ranges do not overlap. |
| 9 | Me, after step 1 | Install ffmpeg/librosa/soundfile in `.venv`. Re-run the audit with signal-level checks, then start Phase 1 cleaning. |
| 10 | Group (contingent) | The margin is zero, so consider collecting **additional consented participants** under the identical protocol (codes Y2-081+, coordinated with Group 4), if the instructor allows. This is the only legitimate way to build a buffer. |

---

## Appendix A: Observations on the protocol document

- Section numbering jumps from **XVIII to XX**; there is no Section XIX. Ask the instructor whether a section was omitted.
- **§VII:** the session time boundary "must be defined before data collection"; it is not in the supplied material.
- **§XII:** session tokens for filenames are shown only by example (`PM`).
- **§VIII:** "Any response is acceptable" is ambiguous (see §13).

## Appendix B: What was produced, and how to re-run

- `data/metadata_raw/`: a byte-identical copy of the metadata PDF (SHA-256 verified), the verbatim CSV, and the extraction provenance JSON.
- `src/00_extract_metadata_pdf.py`: PDF → verbatim CSV.
- `src/01_audit_data.py`: read-only audit.
- `reports/phase0/`:
  - `participant_qa_table.csv` (40 rows, all requested QA columns)
  - `issues_log.csv` (347 issues)
  - `media_inventory.csv` (empty)
  - `audit_summary.json` (every count in this report)
  - `audit.log`
- Re-run with `.venv/Scripts/python.exe src/01_audit_data.py` whenever files are added.
- **Not done (by design):** cleaning, audio conversion, feature extraction, EDA plots and modelling.
