# Retake listening review

Each folder holds every take of one participant who was recorded more than once:
`<code>_take<k>_<hhmmss>_<id>.wav` is the whole clip (16 kHz mono), `..._word.wav` the automatically located word,
and `_SELECTED` marks the take chosen by the complete-word rule (src/03_prepare_audio.py, `select_take`).
Comparison figures: figures/qa/retakes/<code>_takes.png.

Preferred: the blind listening review, data/review/listening_review.html (docs/RUNNING.md section 6.1). It covers
these takes without showing which one was selected, and unanimous choices of two or more listeners are applied
automatically. This folder is for a joint session afterwards.

How to review here (Bisaya-speaking group members, independently first):
1. Do NOT look at the participant's valence/arousal or feeling text while listening.
2. For each take, fill retake_review_sheet.csv: heard_word, whole_word_audible_Y_N, clarity_1_to_5,
   other_voice_over_word_Y_N.
3. Choose the take in which the WHOLE word is heard most clearly. Never choose the take that "sounds more
   emotional" - that would let the expected answer steer the data (protocol Section X).
4. Record a joint decision in data/processed_audio/take_overrides.csv (it wins over the listening review):
   participant_code,drive_file_id,reason,decided_by
   then re-run: .venv/Scripts/python.exe run_pipeline.py --skip-fetch --from 03
