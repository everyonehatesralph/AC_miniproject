# Defense documentation: the project explained

*Group 3 · Bisaya post-class speech and self-reported affect · 40 participants. Built by `src/10c_build_defense_documentation.py` from the result files and the settings in the code, so every number matches the slides and the report.*

## How to use this

- This document explains **how the project works and why**. The presentation script (`presentation_script.html`) has **what to say** and the likely questions with answers.
- Read sections 1 to 3 once, from start to end. Use sections 4 to 6 to look things up.
- Each step in section 3 has three parts: what happens, why we did it this way, and what you may be asked.

## 1. The project on one page

- **The question.** Can the sound of one spoken Bisaya word, said right after class, tell us how a student feels?
- **The participants.** 40 second-year Computer Science students. Each one has a code such as Y2-041. No names are stored.
- **What each participant gave.** One spoken word, and two ratings from 1 to 5: valence (how pleasant the feeling is) and arousal (how much energy it has).
- **What the computer measured.** 108 numbers that describe the sound of the word.
- **The test.** Three kinds of models tried to guess the rating group from those numbers. Each model was always tested on participants it had not learned from.
- **The result.** Our data do not show that post-class feelings can be recognised from the voice. The best result was 0.70 for valence (pleasant vs not pleasant), where pure guessing gives 0.50. After the correction for 9 tests its p-value is 0.054, so it is not significant. Without the 9 files we could not check, that score is 0.63.
- **The trap we found.** The word is linked to the feeling: 13 of 15 Kapoy speakers felt not pleasant, against 7 of 25 speakers of other words. By sound alone the computer tells Kapoy from other words only weakly (balanced accuracy 0.69, not significant). Still, a model that picks up the word even partly could score above chance without hearing the feeling, and this data cannot separate the two.
- **Why the project is still useful.** The whole analysis is complete, tested and repeatable. We know the word must be controlled in a future study. And we know how many participants such a study needs: about 90.

## 2. From recording to result

| Step | What goes in | What happens | What comes out |
|---|---|---|---|
| Collect (before the pipeline) | participants | each one says a word on a phone video and gives two ratings | the videos and the sheet |
| Convert (before the pipeline) | the videos | a group member converted them to short 16 kHz WAV files using ChatGPT | 40 WAV files (`CS2B_v2_wav_16k`) |
| 1. Check | the sheet and the audio | every cell and every file is checked; nothing is changed | `reports/phase0_data_audit.md` |
| 2. Clean | the sheet | fixes by fixed rules only, each one logged | `data/metadata_cleaned.csv`, `reports/phase1/cleaning_log.csv` |
| 3. Find the word | the audio | each file is checked, the word is located, doubtful files are marked | `data/analysis_manifest.csv`, `reports/data_quality_report.md` |
| 4. Measure | the located words | 108 numbers per recording | `data/acoustic_features.csv` |
| 5. Explore | the numbers and the ratings | charts and simple links | `reports/exploratory_analysis.md`, `figures/eda/` |
| 6. Train | the numbers and the rating groups | 3 models and 2 baselines, tested on unseen participants; 500 shuffles per test | the `models/` folder |
| 7. Evaluate | the models' answers | scores, intervals, p-values, corrections | `reports/model_evaluation.md`, `reports/evaluation/` |
| 8. Word check | the numbers, the words, the answers | is it the voice or the word? | `reports/lexical_control_analysis.md` |
| 9. Report | all results | the report, the web pages and the slides are written | `reports/final_group_research_report.md`, `reports/presentation/` |
| 10. Validate | the whole project | 25 automatic checks | `reports/validation_report.md` |

One command runs steps 1 to 10: `python run_pipeline.py`. Step 6 is the slow one: about 432 minutes on our PC.

### The tools

| Tool | What we use it for |
|---|---|
| Python | the language of every step |
| pandas, numpy | tables and numbers |
| Praat (through parselmouth) | measuring pitch |
| librosa | measuring loudness, the spectrum and the MFCCs |
| scikit-learn | the models, the cross-validation and the baselines |
| scipy | the statistical tests |
| matplotlib | the figures |
| python-pptx | writing the slides |
| pytest | automatic tests of the code |
| Git and GitHub | keeping every version of the code and the reports |

## 3. Each step: what, why, and what you may be asked

### 3.1 The correct answers (labels)

- **What happens:** the correct answer for each recording is the participant's own rating. Because some scores were chosen by very few people, we group the scores: arousal in three groups: low (1–2), moderate (3), high (4–5); valence in two groups: not pleasant (1–3), pleasant (4–5); arousal in two groups: not high (1–3), high (4–5).
- **The rule:** a grouping is used only if every group has at least 5 people. Not used: the five valence scores one by one; the five arousal scores one by one; valence in three groups: unpleasant (1–2), neutral (3), pleasant (4–5).
- **Why this way:** the protocol defines the feeling as what the participant reports. A listener's guess, or the meaning of the word, would be our opinion and not data.
- **Details to know:** 0 spoken words are blank in the sheet and stay blank (label UNVERIFIED). 19 words are not on the protocol's suggested list; we kept them as the participants said them.
- **If asked “Did you change any answer?”** No. Ratings are only written as whole numbers. Nothing was changed because it looked unusual, and no word was guessed from the feeling text.

### 3.2 Checking and cleaning the sheet

- **What happens:** step 1 reads every cell and every file and lists the problems. It changes nothing. Step 2 then applies these fixed rules:
  - R1: remove stray spaces in the participant code (1 change)
  - R2: remove stray spaces in text cells (3 changes)
  - R3: write the year level as a number (40 changes)
  - R4: map the class activity to the protocol's category (the instructor still has to confirm this mapping) (40 changes)
  - R5: use the protocol's spelling for the session (40 changes)
  - R6: write each rating as a whole number (80 changes)
  - R7: do not use a rating that is crossed out in the sheet (0 changes)
  - R8: write consent as Y or N (40 changes)
- **Two more rules for later:** the group can supply corrections in a file, and a blank word can be filled from the listening review, but only when two listeners agree and only with a protocol word. Neither has been used yet.
- **The sheet was completed on 12 October:** 9 blank words were filled in by a group member who listened to each recording, and the ratings of Y2-076, Y2-077, Y2-080 were corrected from the participants' answer forms, because the first sheet had typing mistakes. The results come from the completed sheet. The sheet before it is kept in `data/metadata_raw/archive/`, and every changed cell, with the best results the earlier sheet gave, is in `reports/sheet_update_applied.md`. Those participants are all among the files we could not check, so the scores without those files are the same for both sheets.
- **Why this way:** no cell is edited by hand, so every change can be traced. The original value stays in the data beside the cleaned one, and each change is a line in `reports/phase1/cleaning_log.csv`.
- **If asked “How do we know you did not edit the data?”** Every change is one of the rules above and is logged with the old and the new value.

### 3.3 The audio and finding the word

- **The format:** WAV, 16 kHz, one channel, 16 bits. The pipeline does not remove noise, change the loudness or filter the sound. The protocol forbids that, and it would change the things we measure.
- **How the word is found:** the computer measures the loudness every 10 ms. The background level is the loudness that 20% of the clip stays below. The word is the stretch around the loudest point that stays clearly above the background (at least 6 dB above it, or half-way to the peak if that is higher). Short dips of less than 0.25 s inside a word are bridged, and 0.05 s is kept on each side.
- **Why this way:** the rule is automatic, the same for every file, and it never sees the ratings. So it cannot favour any result.
- **Doubtful files are marked, not removed.** The marks and how many of the 40 recordings carry each:
  - `NO_CAPTURE_TIME` (40): the file does not carry the date and time of the recording
  - `BACKGROUND_VOICES` (16): other voices are heard outside the word (at least 0.3 s of voiced sound)
  - `LOW_SBR` (10): the word is less than 10 dB louder than the background
  - `TOO_SHORT_FOR_WORD` (6): the located sound is shorter than the word should be
  - `COMPETING_EVENT` (6): another sound almost as loud as the word (within 3 dB) is elsewhere in the clip
  - `EDGE_END` (3): the word touches the end of the file, so its end may be cut
  - `DIGITAL_SILENCE` (3): at least 0.05 s of exact zeros, which a phone recording never contains
  - `EDGE_START` (1): the word touches the start of the file, so its beginning may be cut
  - `LONG_SEGMENT` (1): the located sound is longer than 1.5 s, so it may hold more than one word
  - `CLIPPING` (1): the sound was too loud for the recorder, so its peaks are cut off
- **Uncertain word location:** 14 of 40 recordings have a mark that puts the location of the word in doubt. Nobody has checked them by listening yet; that is the listening review in our next steps.
- **Where the files came from:** 31 match an original recording. 9 could not be checked (Y2-053, Y2-071, Y2-074, Y2-075, Y2-076, Y2-077, Y2-078, Y2-079, Y2-080), and 3 of the files are edited or clipped (Y2-076, Y2-077, Y2-078).
- **If asked “Is the automatic word location correct?”** We do not claim it is perfect. We mark the doubtful files and run the models again without them (section 3.9).

### 3.4 The 108 features

- **What happens:** the located word is cut into windows of 25 ms, one every 10 ms. The computer measures each window and then summarises the whole word.

| Group | How many | What it describes |
|---|---|---|
| MFCC | 26 | 13 numbers for the shape of the sound spectrum; we keep the average and the spread of each over the word |
| delta MFCC | 26 | how fast those 13 numbers change during the word |
| delta-delta MFCC | 26 | how fast that change itself changes |
| F0 (pitch) | 9 | how high the voice is: average, middle value, lowest, highest, spread, range and slope |
| energy | 6 | loudness: average, middle value, lowest, highest and spread |
| zero-crossing rate | 6 | how noisy or hissing the sound is |
| spectral | 6 | where the energy sits in the spectrum: brightness (centroid), width and roll-off |
| duration/voicing | 3 | how long the word is, and how much of it is voiced |

- **Total:** 26 + 26 + 26 + 9 + 6 + 6 + 6 + 3 = 108.
- **Why these features:** the protocol lists them (Section XIV), and they are the usual features in speech research. The list was fixed before we saw any result.
- **A caution about pitch:** pitch in Hz mostly shows whether a person's voice is naturally high or low. With one recording per person we cannot subtract the person's normal voice.
- **If asked “Why not let a neural network learn the features?”** That needs thousands of recordings. We have 40.

### 3.5 Exploring the data

- **What happens:** we compute the Spearman correlation between every feature and each rating, then correct for testing 108 features (the FDR correction).
- **The result:** 0 features are significant for arousal and 1 for valence. Of the six features chosen in advance, none is significant.
- **Why Spearman:** the ratings are ordered scores from 1 to 5, not measurements. Spearman uses ranks, so it fits.
- **Why choose six features in advance:** if you test many features and report only the best, you always find something by luck. Choosing first keeps us honest.
- **What the sample can show:** with 40 participants, only links of about 0.44 or stronger can be found reliably (80% of the time). A moderate link of 0.3 needs about 90 participants.

### 3.6 Training and testing the models

- **The three models:** logistic regression (straight boundaries between groups), SVM (curved boundaries) and random forest (300 decision trees that vote). They make different assumptions, so together they are a fair test.
- **The two baselines:** one always answers the most common group; one guesses at random in the same proportions as the groups. Neither looks at the audio.
- **The test:** the participants are split into 5 parts. The model learns from 4 parts and is tested on the last one, and each part is the test once. We repeat this 10 times with different splits, so one lucky split cannot decide the result.
- **Choosing the settings:** inside each training set there is a second, smaller split. It is used only to choose the model's settings. The test part is never used for that. This is called nested cross-validation.
- **Preparing the numbers:** a missing value is replaced by the middle value of the training data, and the features are put on the same scale using the training data only.
- **Unequal groups:** each group counts equally (class weights). We did not create artificial samples.
- **Why this way:** a real system would meet new students, so the model must be tested on people it has never heard. And the model must not see anything about the test participants while it learns.
- **The code stops by itself** if a column that is not an acoustic feature enters the model, if a participant is in both the training and the test part, or if a test part lacks a group.
- **If asked “Why not SMOTE?”** With so few people, artificial samples would be copies of the same few voices, and they leak information if they are made before the split.

### 3.7 Scoring and statistics

- **The score:** balanced accuracy, the average of the scores for each group. Pure guessing gives 1 divided by the number of groups.
- **The interval:** we draw the participants again at random, with replacement, 2000 times and compute the score each time. The middle 95% of those scores is the 95% interval.
- **The p-value:** we shuffle the ratings so they no longer belong to the right recordings and run the whole procedure again, 500 times. The p-value is (number of shuffles that score at least as high + 1) divided by (500 + 1). So the smallest possible p-value is 0.002.
- **Our smallest p-value:** 0.006, for the random forest on valence (pleasant vs not pleasant). It means 2 of the 500 shuffles scored at least as high as the real data.
- **The correction:** we ran 9 tests (3 models for each of 3 targets). With 9 tests, about 0.45 lucky results are expected. The Holm correction multiplies the smallest p-value by 9, the next by 8, and so on. 0.0060 × 9 = 0.054.
- **Comparing the models:** we use a t-test that is corrected for repeated cross-validation, because the same participants are tested again in every repeat. 0 comparisons have p below 0.05 before any correction.
- **If asked “Why is p below 0.05 not enough?”** Because it was one of 9 tests. One lucky result among 9 is not evidence.

### 3.8 The word check

- **Why we did it:** each participant chose their own word. If the word is linked to the feeling, a model can use the word as a shortcut.
- **Check 1, which words can be analysed:** only Kapoy was said by enough people (15).
- **Check 2, can the computer hear the word:** the same models tell Kapoy from other words with balanced accuracy 0.69 (p = 0.072). That is not significant: in this data the sound alone does not reliably give the word away.
- **Check 3, is the word linked to the feeling:** not pleasant for 13 of 15 Kapoy speakers and for 7 of 25 speakers of other words (Fisher test, p = 0.001).
- **Check 4, the same word with different feelings:** among Kapoy speakers alone, 0 of 12 links between voice and rating are significant.
- **What it means:** a score above chance could come from the word and not from the feeling. A future study must use one fixed word, or enough recordings of each word.

### 3.9 Running the models again without doubtful files

Scores of the random forest when a doubtful part of the data is left out. With all data: 0.41 for three-level arousal (low, moderate, high) and 0.70 for valence.

| What is left out | Participants left | Arousal score | Valence score |
|---|---|---|---|
| the recordings whose word location is uncertain | 26 | 0.38 | 0.63 |
| the copies of recordings made with a different setting | 35 | 0.47 | 0.72 |
| the edited or clipped files | 37 | 0.47 | 0.74 |
| the files that have no verified original | 31 | 0.37 | 0.63 |

- **How to read it:** the scores move up and down depending on which files are in. That shows the sample is too small for a stable answer.
- **Why it matters for the defense:** our conclusion does not depend on the doubtful files. With or without them, no claim of recognition is supported.

### 3.10 Checks, tests and repeatability

- **Validation:** step 10 checks the whole project: 24 checks pass and 1 is blocked (the consent log).
- **What is checked:** the number of participants, consent, no names in the data, one recording and one row of features per participant, the WAV format, no leakage, no participant in both training and test data, and that the report states the same numbers as the results.
- **Code tests:** the rules that decide the data (which flags are set, how a word is found, how consent is read) have automatic tests that run on GitHub for every change.
- **Same result every time:** the random choices use a fixed seed (42), and the versions of all tools are written down in `requirements.txt`.
- **What is not complete:** Required consent documentation.

## 4. How to read each table and figure

### The tables on the slides

- **Slide 6, “Voice measures and the ratings”:** one row per voice feature. ρ is the Spearman correlation with the rating; p is its p-value before any correction. Look for ρ far from 0 and a small p. None of the six is significant after the correction.
- **Slide 8, “No model beats chance after correction”:** one row per target, showing the best of the three models. Compare *Balanced accuracy* with *Chance*. The interval in brackets shows how uncertain the score is. *p-value* is before the correction, *p after correction* is the one that counts.
- **Slide 10, “Is it the voice, or just the word?”:** each cell is a number of participants. Read along a row: most Kapoy speakers are in *Not pleasant*, most speakers of other words are in *Pleasant*.

### The figures

| Figure | What it shows | How to read it |
|---|---|---|
| `eda/fig01` | how many participants chose each score | 37 of 40 valence ratings and 31 of 40 arousal ratings are 3 or higher |
| `eda/fig02` | each pair of valence and arousal scores, with the number of participants | numbers along the diagonal mean the two ratings rise together (ρ = 0.72) |
| `eda/fig03` | the size of each group in every grouping | the line marks 5 people; a grouping with a bar below the line is not modelled |
| `eda/fig04` | how many participants said each word | Kapoy was said by 15 participants; 0 words are blank |
| `eda/fig05` | the feeling words that participants wrote | descriptive only; it is not used by the models |
| `eda/fig06` | how selected features are spread over the recordings | wide spreads mean the voices differ a lot from person to person |
| `eda/fig07` | how strongly features are related to each other | strongly related features carry the same information |
| `eda/fig08` | the correlation of every feature with each rating | the panel titles say how many pass the correction: 0 for arousal, 1 for valence |
| `eda/fig09`, `eda/fig10` | the six features chosen in advance, by arousal group and by valence group | each dot is one participant; where the groups overlap, the feature does not separate them |
| `eda/fig11` | four features, for the two recording settings | the 5 files of setting B look different, but 0 of 108 features differ after the correction (closest p = 0.053; 18 before it). If the title printed inside the figure says something else, the figure is older than this document: use these numbers |
| `eda/fig12` | the ratings of the participants, by the word they said | descriptive only; the test is in the word check (section 3.8) |
| `eda/fig13` | how noisy the recordings are, by word | the black tick is the middle value; it shows whether some words were recorded in noisier conditions |
| `eda/fig14`, `eda/fig15` | the average pitch and the average loudness of each participant, from lowest to highest | the colours are the rating groups; where they are mixed along the line, the feature does not separate the groups |
| `evaluation/fig_eval_balanced_accuracy` | the score of every model, with its 95% interval (slide 9) | 6 of the 9 blue bars cross the chance line |
| `evaluation/fig_eval_confusion_…` | for each model, which group it answered for each true group | rows are the true group, columns the answer; a good model has high numbers on the diagonal |
| `evaluation/fig_eval_sensitivity` | the scores when doubtful files are left out (section 3.9) | the horizontal line is chance; compare each score with the line and with the score for all data |
| `lexical/fig_lexical_kapoy_features_vs_affect` | the six features against the ratings, for Kapoy speakers only | 0 of 12 links are significant |

## 5. Where each number on the slides comes from

| Slide | Number | What it is | File |
|---|---|---|---|
| 3 | 40 and 40 | rows in the sheet; participants in the analysis | `data/metadata_cleaned.csv`, `data/analysis_manifest.csv` |
| 3 | 31, 9, 3 | files that match an original; files that could not be checked; files edited or clipped | `reports/supplied_audio_audit.csv` |
| 3 | 0 | blank spoken words | `reports/phase1/cleaning_summary.json` |
| 3 | 16 of 40 | recordings with background voices | `reports/data_quality_summary.json` |
| 5 | 0.72 | Spearman correlation of valence with arousal (40 participants) | `reports/eda/eda_summary.json` |
| 6 | ρ and p in the table | correlation of each feature with each rating | `reports/eda/spearman_feature_vs_rating.csv` |
| 6, 11 | 0.44 and 90 | the weakest link that 40 participants can show, and the participants needed for a link of 0.3 (5% significance, 80% power) | computed in `src/09c_build_presentation.py` |
| 8 | 0.41 [0.29, 0.54], p 0.134 → 0.802 | three-level arousal (low, moderate, high): score, 95% interval, p-value before and after correction | `reports/evaluation/summary_metrics.csv` |
| 8 | 0.70 [0.58, 0.82], p 0.006 → 0.054 | valence (pleasant vs not pleasant): score, 95% interval, p-value before and after correction | `reports/evaluation/summary_metrics.csv` |
| 8 | 0.55 [0.43, 0.68], p 0.271 → 1.000 | two-level arousal (high vs not high): score, 95% interval, p-value before and after correction | `reports/evaluation/summary_metrics.csv` |
| 8, 9 | 6 of 9 | model intervals that include the chance level | `reports/evaluation/summary_metrics.csv` |
| 10 | 0.69, p = 0.072 | how well the audio tells Kapoy from other words | `reports/lexical/lexical_summary.json` |
| 10 | 13 of 15; 7 of 25; p = 0.001 | not-pleasant feelings by word, Fisher test | `reports/lexical/lexical_summary.json` |
| 10 | 0 of 12 | significant links among Kapoy speakers alone | `reports/lexical/lexical_summary.json` |
| 11 | 5 files; 18 of 108; 0.053 | files from the other recording setting; features that differ before the correction; closest corrected p-value | `data/analysis_manifest.csv`, `reports/eda/recording_configuration_effect.csv` |

## 6. Glossary

| Word | Meaning in plain words |
|---|---|
| Affect | A feeling. Here it is described by two ratings: valence and arousal. |
| Valence | How pleasant the feeling is, from 1 (very unpleasant) to 5 (very pleasant). |
| Arousal | How much energy the feeling has, from 1 (very low) to 5 (very high). |
| Self-report | What the participant says about their own feeling. It is our correct answer. |
| Acoustic feature | A number that describes a sound, measured by the computer. |
| Pitch (F0) | How high or low the voice is, in Hz. |
| Energy (RMS) | How loud the sound is. |
| MFCC | 13 numbers that describe the shape of the sound spectrum. They are what makes voices and vowels sound different. |
| Zero-crossing rate | How often the sound wave crosses zero. Noisy and hissing sounds have a high rate. |
| Spectral centroid | The brightness of the sound: high when the sound has much high-frequency content. |
| Sampling rate | How many sound values are stored per second. Ours is 16 000 (16 kHz), enough for speech. |
| SBR | Speech-to-background ratio: how much louder the word is than the background, in dB. |
| Group (class) | One of the answers a model can give, for example low, moderate or high arousal. |
| Model | A program that learns a rule from examples and then gives an answer for a new example. |
| Training and test data | The model learns from the training data. It is scored on the test data, which it has never seen. |
| Cross-validation | Splitting the participants into parts, and letting each part be the test data once. |
| Nested cross-validation | A second split inside the training data, used only to choose the model's settings. |
| Speaker-independent | No person is in both the training and the test data. |
| Data leakage | Information about the test data that reaches the model during training. It makes scores look too good. |
| Setting (hyperparameter) | A choice that controls how a model learns, for example how many trees a forest has. |
| Baseline (dummy model) | A model that never looks at the audio. A real model must beat it. |
| Class weight | Making each group count equally, even when it has fewer people. |
| Balanced accuracy | The average of the scores for each group. Always giving the most common answer scores only the chance level. |
| Chance level | The balanced accuracy of pure guessing: 1 divided by the number of groups. |
| 95% interval | The range where the true value probably lies. A wide interval means an uncertain result. |
| Bootstrap | Drawing the participants again at random, many times, to see how much a score changes. |
| p-value | How often results as good as ours appear when there is really nothing to find. Small means unlikely to be luck. |
| Permutation test | Shuffling the ratings so they mean nothing, running everything again, and counting how often the score is as high. |
| Multiple testing | Running many tests. Some will look good by luck alone. |
| Holm correction | A correction for multiple testing. It raises the p-values of the model tests. |
| FDR (Benjamini–Hochberg) | A correction for multiple testing, used for the many feature correlations. |
| Spearman correlation (ρ) | How strongly two things rise together, from −1 to +1. It uses ranks, so it suits 1-to-5 ratings. |
| Statistical power | The chance of finding an effect that really exists. Small samples have low power. |
| Confound | A hidden third thing that explains a link between two others. |
| Lexical confound | The spoken word is linked to the feeling, so a model can use the word instead of the feeling. |
| Sensitivity analysis | Repeating the analysis without a doubtful part of the data, to see whether the result holds. |
| Reproducible | Anyone can run the same code on the same data and get the same result. |
| Pseudonymous code | A code such as Y2-041 that stands for a participant, so no name is stored. |
