# Data Dictionary

All data files in the project, column by column. Paths are relative to the project root.
"Raw" columns keep the original value exactly; cleaned columns are derived by documented rules (see
[DOCUMENTATION.md](DOCUMENTATION.md) §5.1).

## 1. `data/metadata_raw/G3_metadata_extracted_raw.csv`: verbatim metadata (40 rows)

The 14 columns of the Group 3 sheet, exactly as written: `Participant Code`, `Year Level`, `Class Activity`,
`Subject`, `Session`, `Spoken Word`, `Self-Reported Feeling`, `Valence (1-5)`, `Arousal (1-5)`, `Audio Filename`,
`Consent Obtained (Y/N)`, `Date Collected`, `Collector Initials`, `Notes`.

Extraction evidence:

| Column | Meaning |
|---|---|
| `_source_page`, `_source_row_on_page` | location of the row in the PDF |
| `_wrapped_fields` | cells whose text wrapped onto several lines and was re-joined |
| `_struck_fields`, `_struck_detail` | cells with strikethrough formatting (only Y2-076 `Arousal (1-5)`) |
| `_nonblack_text_fields`, `_filled_fields` | coloured text / cell fills (none) |

## 2. `data/metadata_cleaned.csv`: Phase 1 output (40 rows)

| Column | Type | Meaning |
|---|---|---|
| `participant_code` | str | normalised code `Y2-NNN` (identifier only, never a feature) |
| `participant_code_raw` | str | as written (`Y2- 041`) |
| `year_level`, `year_level_raw` | int, str | 2 / `2nd year` |
| `class_activity_raw` | str | `Lecture` |
| `class_activity_protocol` | str | mapped protocol category `Lecture/Discussion` |
| `class_activity_mapping` | str | `EXACT` / `MAPPED_PENDING_CONFIRMATION` / `UNRECOGNISED` |
| `subject_raw`, `subject` | str | raw / whitespace-stripped (blank for Y2-071–080) |
| `session_raw`, `session` | str | `Afternoon-Evening` / protocol spelling `Afternoon–Evening` |
| `session_token_assumed` | str | `PM` (assumption from the §XII example) |
| `spoken_word_raw`, `spoken_word` | str | raw / stripped; blank = not recorded (never inferred) |
| `spoken_word_source` | str | `metadata`, `missing`, or `verified_correction` |
| `spoken_word_in_protocol_list` | bool | word is one of Salamat, Ambot, Grabe, Lisod, Sayon, Lingaw, Kapoy |
| `self_reported_feeling_raw`, `self_reported_feeling` | str | participant's own description, raw / stripped |
| `valence_raw`, `valence` | str, Int | `4 — Pleasant` / 4 (1 Very Unpleasant … 5 Very Pleasant) |
| `valence_status` | str | `VALID` / `CORRECTED_VERIFIED` |
| `arousal_raw`, `arousal` | str, Int | `3 — Moderate` / 3 (1 Very Low … 5 Very High); missing when struck through |
| `arousal_status` | str | `VALID` / `STRUCK_THROUGH` / `CORRECTED_VERIFIED` |
| `consent_raw`, `consent_obtained` | str | `Yes` / `Y` |
| `date_collected`, `collector_initials`, `notes` | str | as recorded (all blank) |
| `audio_filename_raw` | str | as recorded (all blank; the convention name is assigned in Phase 4) |
| `qa_flags` | str | `;`-separated flags (e.g. `SPOKEN_WORD_MISSING`, `WORD_NOT_IN_PROTOCOL_LIST`, `FEELING_IDENTICAL_TO_WORD`) |
| `corrections_applied` | str | verified corrections with source/verifier (none yet) |

Change log: `reports/phase1/cleaning_log.csv` (`participant_code`, `field`, `raw_value`, `cleaned_value`, `rule`).

## 3. Raw recordings: `data/raw/`

| File | Columns |
|---|---|
| `drive_manifest.csv` | `drive_file_id`, `drive_name` (name in the Drive folder "CS2B"), `listed_size`, `listed_modified_label` (as displayed in the web listing on 2026-10-03) |
| `download_log.csv` | `drive_file_id`, `drive_name`, `participant_code` (parsed from the Drive name), `local_file`, `bytes`, `sha256`, `container`, `listed_size`, `logged_at_utc` |
| `recording_map.csv` | `original_file` (relative to `data/raw/`), `participant_code`, `drive_name` |

Local file name: `<drive name>__<drive file id>.mov`; the original bytes are never modified.

## 4. `reports/audio_quality_table.csv`: per-recording QA (38 rows: every file, every take)

| Column(s) | Meaning |
|---|---|
| `participant_code`, `drive_name`, `drive_file_id`, `original_file`, `original_sha256`, `container` | identity and provenance |
| `capture_time` | embedded QuickTime creation date (local time with offset) |
| `device_model` | e.g. `iPhone 17` |
| `audio_codec`, `orig_sample_rate`, `orig_channels`, `orig_audio_streams`, `orig_kbps`, `orig_duration_s` | original audio stream |
| `readable` | decoded successfully |
| `peak_dbfs`, `clip_fraction`, `clip_pct`, `dc_offset`, `rms_dbfs`, `interchannel_corr` | original-signal QA (native rate/channels) |
| `proc_duration_s` | duration of the 16 kHz mono signal |
| `seg_start_s`, `seg_end_s`, `seg_duration_s`, `peak_time_s` | located utterance (with 0.05 s context each side) |
| `event_duration_s` | duration of the energy event itself (without the added context) |
| `word`, `word_syllables`, `event_s_per_syllable` | recorded word, its syllable count (vowel letters), event duration per syllable (< 0.10 → `TOO_SHORT_FOR_WORD`) |
| `sbr_db` | utterance peak minus background floor (smoothed frame energy) |
| `competing_margin_db` | peak minus the strongest other event ≥ 0.3 s away |
| `threshold_db`, `floor_db` | localisation threshold and floor (dBFS) |
| `touches_start`, `touches_end` | event touches the clip edge |
| `segmentation_source` | `automatic` or `human (<reviewer>)` |
| `voiced_frames_in_segment`, `voiced_s_outside_segment` | Praat voicing inside the event / outside it (± 0.1 s) |
| `flags` | `;`-separated QA flags (definitions: DOCUMENTATION.md §5.3) |
| `usable` | readable and speech present |
| `n_takes`, `selected`, `duplicate_status` | `single` / `RETAKE_SELECTED` / `RETAKE_NOT_SELECTED` |
| `selection_reason` | rule applied |
| `standardized_filename`, `exclusion_reason` | naming and exclusion |
| `processed_file`, `segment_file`, `processed_sha256`, `segment_sha256` | processed outputs (selected takes only) |

## 5. `data/processed_audio/file_mapping.csv`: original → processed (38 rows)

`original_file`, `drive_name`, `drive_file_id`, `participant_code`, `capture_time`, `selected`, `duplicate_status`,
`selection_reason`, `processed_file`, `segment_file`, `seg_start_s`, `seg_end_s`, `segmentation_source`,
`processed_sha256`, `segment_sha256`, `status` (`SELECTED` / `RETAKE_NOT_SELECTED` / `UNUSABLE`).

## 6. `data/analysis_manifest.csv`: participant inclusion (40 rows)

| Column | Meaning |
|---|---|
| `participant_code` | code |
| `n_takes` | number of recordings supplied |
| `include_primary` | in the primary analysis set (consent recorded + one usable selected take) |
| `exclusion_reason` | `NO_RECORDING`, `NO_USABLE_RECORDING`, `NO_CONSENT`, or blank |
| `selected_drive_file_id`, `selected_file`, `processed_file`, `segment_file` | the take used |
| `selected_flags` | QA flags of the selected take |
| `subset_retake` | participant had more than one take (sensitivity analysis 1) |
| `subset_config_or_time_anomaly` | `RECORDING_CONFIG_DIFFERS` or `CAPTURE_TIME_OUTLIER` (sensitivity analysis 2) |
| `subset_segmentation_critical` | `EDGE_START`/`EDGE_END`/`LONG_SEGMENT`/`COMPETING_EVENT` (sensitivity analysis 3) |
| `word_in_filename` | word token used in the file name (`UNVERIFIED` if not recorded) |

## 7. `data/processed_audio/segment_review_sheet.csv`: listening worksheet (32 rows)

Pre-filled: `participant_code`, `standardized_filename`, `segment_file`, `seg_start_s`, `seg_end_s`, `flags`,
`metadata_word`.

To be filled by a Bisaya-speaking reviewer: `heard_word`, `boundaries_ok_Y_N`, `corrected_start_s`,
`corrected_end_s`, `reviewer_initials`, `notes`. Verified boundaries go into `segment_overrides.csv`
(`participant_code`, `seg_start_s`, `seg_end_s`, `reviewer`, `note`).

## 7b. `data/processed_audio/retake_review/`: listening packet for participants with several takes

One folder per participant with every take: `<code>_take<k>_<hhmmss>_<id>[_SELECTED].wav` (whole clip, 16 kHz mono) and `..._word.wav` (located word). `retake_review_sheet.csv` has one row per take with the objective evidence (`event_duration_s`, `event_s_per_syllable`, `sbr_db`, `competing_margin_db`, `flags`, `auto_selected`) and blank listener columns (`heard_word`, `whole_word_audible_Y_N`, `clarity_1_to_5`, `other_voice_over_word_Y_N`, `choose_this_take_Y_N`, `listener_initials`, `notes`). A listening decision goes into `take_overrides.csv` (`participant_code`, `drive_file_id`, `reason`, `decided_by`).

## 8. `data/acoustic_features.csv`: feature table (32 rows × 111 columns)

| Column | Meaning |
|---|---|
| `participant_code` | identifier only |
| `segment_file`, `segment_sha256` | provenance |
| 108 numeric features | one row per included participant |

Feature definitions: `data/acoustic_features_dictionary.csv` (`feature`, `group`, `unit`, `definition`) and
DOCUMENTATION.md §5.8. Naming pattern:
- `utterance_duration_s`, `voiced_duration_s`, `voiced_ratio`
- `f0_{mean,std,median,min,max,iqr}` (Hz), `f0_std_st`, `f0_range_st`, `f0_slope_st_per_s`
- `rms_db_{stat}`, `zcr_{stat}`
- `mfcc_{1..13}_{mean,std}`, `delta_mfcc_{1..13}_{mean,std}`, `delta2_mfcc_{1..13}_{mean,std}`
- `spectral_{centroid,bandwidth,rolloff}_hz_{mean,std}`

## 9. Model outputs: `models/`

| File | Columns |
|---|---|
| `oof_predictions.csv` | `task`, `model`, `repeat`, `fold`, `participant_code`, `y_true`, `y_pred` |
| `cv_fold_scores.csv` | `task`, `model`, `repeat`, `fold`, `n_train`, `n_test`, `accuracy`, `balanced_accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `best_params` (JSON) |
| `cv_splits.csv` | `task`, `repeat`, `fold`, `participant_code` (outer test-fold membership) |
| `permutation_tests.csv` | `task`, `model`, `observed_pooled_ba_repeat0`, `null_mean`, `null_sd`, `null_p95`, `n_permutations`, `p_value` |
| `sensitivity_oof_predictions.csv`, `sensitivity_fold_scores.csv` | as above plus `scenario` |
| `final/<task>__<model>.joblib` | dict with `model` (fitted Pipeline), `features` (column order), `classes`, `task`, `task_definition` |
| `training_summary.json` | versions, seeds, grids, target viability, final-model cards |

## 10. Report tables

| File | Contents |
|---|---|
| `reports/phase0/participant_qa_table.csv` | Phase 0 QA columns requested in the brief (`metadata_present`, `recording_present`, `duplicate_recording`, `consent_recorded`, `spoken_word_present`, `self_report_present`, `valence_valid`, `arousal_valid`, `audio_readable`, `audio_duration`, `sample_rate`, `channels`, `clipping_detected`, `silence_problem`, `filename_valid`, `include_in_analysis`, `exclusion_reason`, …) |
| `reports/phase0/issues_log.csv` | `scope`, `participant_code_raw`, `participant_code`, `field`, `issue_code`, `severity` (BLOCKER/MAJOR/MINOR/REVIEW/INFO), `raw_value`, `detail` |
| `reports/eda/*.csv` | affect distributions, target viability, Spearman feature-vs-rating tests (with BH q-values), recording-configuration effect, word × affect |
| `reports/evaluation/*.csv` | summary metrics, per-class metrics, confusion matrices, model comparison, hyperparameter choices, sensitivity summary |
| `reports/lexical/*.csv` | word feasibility, within-word correlations, accuracy by word group |
