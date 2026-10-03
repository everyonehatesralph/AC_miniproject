# Pre-registered analysis plan: expanded or pooled study

**Version 1.0, 2026-10-03. Written before any new participant is recorded.** Freeze it by committing it
unchanged: the commit date is the evidence that the plan came before the data. Any later change is a
*deviation* and goes into `docs/decision_log.md` with its date and reason.

**Applies to** (a) new Group 3 participants (codes Y2-081 onward, if the instructor approves; see
`docs/next_steps/01_email_to_instructor.md`), and/or (b) a pooled dataset, if other groups share recordings
collected under the same protocol (same prompt, same 1–5 scales, one take, audio only, see
`docs/data_collection_checklist.md`).

## 1. Why this plan exists

The current analysis of 32 participants is the **pilot**. It found no model that survives Holm correction, a
strong lexical confound (the word chosen is decodable from the audio and associated with affect), and a
recording-configuration confound. The pilot was exploratory: hypotheses formed after seeing it cannot be
confirmed on the same data. This plan fixes in advance:

- the hypotheses and their direction,
- the features (few, theory-driven),
- the tests and the correction for multiple testing,
- the sample size,

so that a new sample can confirm or refute them. **Confirmatory tests use the new participants only.** The
pilot + new combined analysis is reported as secondary.

## 2. Hypotheses

| | Hypothesis | Direction | Basis |
|---|---|---|---|
| **H1** (confirmatory) | Self-reported arousal (1–5) correlates with the five H1 features (§4): RMS level, F0 median, F0 range, spectral centroid **positively**, and duration per syllable **negatively** | directional | Higher arousal is consistently linked to louder, higher, more variable, brighter and faster speech (Banse & Scherer, 1996; Juslin & Laukka, 2003; Scherer, 2003). In the pilot all five had the predicted sign (ρ = +0.11 to +0.34; duration per syllable −0.24), none significant (smallest p = 0.054); that is a reason to test, not evidence. |
| **H2** (exploratory) | Self-reported valence (1–5) correlates with the same features | none | Acoustic valence cues are weak and inconsistent in the literature, so no direction is predicted |
| **H3** (confirmatory) | Speaker-independent classification of arousal (3 derived levels) beats the stratified dummy baseline in balanced accuracy | above chance | The pilot could not show this at n = 32 |
| **H4** (confirmatory, conditional) | The H1 associations remain after removing word identity | same as H1 | The pilot showed that word choice is a confound |

**Support rules, fixed now.**

- H1 is supported if at least one H1 feature passes Holm correction **in the predicted direction**. A
  significant result in the opposite direction counts as *not supported* and is reported as such.
- A claim that the system can *recognise* affect needs H3 as well.
- No causal wording: the design is correlational, with post-class self-reports.

## 3. Sample size and power

Spearman correlation, two-sided, power 0.80. n = 3 + 1.06·[(z₁₋α/₂ + z₁₋β) / atanh ρ]²; the factor 1.06 is
the Spearman variance correction of Fieller, Hartley & Pearson (1957). Values were computed with SciPy, not
from a table.

| True correlation ρ | n for one test (α = 0.05) | n for the first Holm step of five tests (α = 0.01) |
|---|---|---|
| 0.2 | 206 | 305 |
| 0.3 | 90 | 133 |
| 0.4 | 50 | 72 |
| 0.5 | 31 | 45 |

| n | smallest ρ detectable (α = 0.05) | (α = 0.01) |
|---|---|---|
| 32 (pilot) | 0.49 | 0.57 |
| 72 | 0.33 | 0.40 |
| 90 | 0.30 | 0.36 |
| 120 | 0.26 | 0.31 |
| 160 | 0.23 | 0.27 |

**Honest conclusion.**

- The pilot could only have detected very large effects (ρ ≥ 0.49).
- A moderate effect (ρ = 0.3) needs about **90 new participants** for one pre-specified test, and about
  **133** to survive Holm across five.
- A single group cannot reach this. Pooling across groups can.

**Stopping rule:** n is fixed by what is collected before the agreed closing date. There is no peeking at
results and no "collect until significant".

## 4. Features

Fixed in advance, defined in `data/acoustic_features_dictionary.csv` or below. No feature selection on the
data.

### H1 family (five features, Holm-corrected)

| Feature | Predicted sign with arousal | Status |
|---|---|---|
| `rms_db_mean`: mean frame RMS level (dBFS) | + | exists. Only interpretable with one device setting and a fixed 20–30 cm distance (checklist) |
| `f0_median`: median F0 of voiced frames (Praat autocorrelation) | + | exists; Spearman is unaffected by the Hz-to-semitone transform |
| `f0_range_st`: F0 range, 5th–95th percentile, in semitones | + | exists |
| `spectral_centroid_hz_mean` | + | exists |
| `event_s_per_syllable`: located word without context padding ÷ syllables of the spoken word | − | computed in step 03 (QA table); to be added to the feature table |

### Secondary, exploratory (BH-FDR)

Each needs to be implemented, and its code reviewed for bugs on the pilot data only, never for effects:

- HNR (dB);
- jitter (local, %);
- shimmer (local, dB);
- alpha ratio and Hammarberg index (eGeMAPS; Eyben et al., 2016);
- F0 slope.

**Caution:** jitter and shimmer need several stable glottal periods, and single words of 0.3–0.6 s give
noisy values. Report how many participants have a valid value.

## 5. Normalisation and confounds

- **Word identity (H4).**
  - (i) If the most frequent word has ≥ 20 tokens, rerun H1 on that word alone.
  - (ii) Rerun H1 on features residualised on word identity. Words with < 5 tokens are pooled as "other".
    For correlations, ranks are residualised on all data. For machine learning, the residualisation is
    fitted inside each training fold.
- **Speaker.** With one utterance per person no speaker normalisation is possible, so differences in
  anatomy stay in the features: F0 differs by about an octave between typical male and female voices.
  - *If the instructor approves*, collect self-reported gender (optional field, consent form updated) and use
    it as a covariate for the F0 features.
  - Without it, this remains a stated limitation.
- **Recording configuration.** Use one device setting for everyone. If a minority configuration occurs
  anyway, report the results with and without it (as in the pilot sensitivity analysis).

## 6. Analyses

1. **H1.** Spearman ρ of each H1 feature with arousal. Two-sided permutation p-values (10,000 permutations),
   Holm at α = 0.05 across the five. 95% CIs by participant bootstrap (10,000 resamples).
2. **H2.** The same features plus the secondary set with valence, Benjamini–Hochberg q < 0.05, labelled
   exploratory.
3. **Ordinal secondary analysis.**
   - Model: a proportional-odds logistic regression of arousal on the standardised H1 features (F0 median in
     semitones re 100 Hz), with the word group as a covariate. Report odds ratios with 95% CIs.
   - Check the proportional-odds assumption against a partial-proportional-odds model.
   - Implementation: `statsmodels` `OrderedModel`, to be added to `requirements.txt` when the data exist.
4. **H3.** The pipeline already in use, unchanged:
   - nested StratifiedGroupKFold, 5 outer folds × 10 repeats, 3 inner folds;
   - stratified and majority dummy baselines, L2 logistic regression and random forest, `class_weight="balanced"`;
   - all preprocessing inside the pipeline;
   - balanced accuracy as the primary metric, macro-F1 as secondary;
   - a permutation test of the whole nested procedure with ≥ 1,000 permutations;
   - Holm correction across the two primary targets (arousal_3, valence_2).
5. **H4.** Analyses (i) and (ii) of §5, plus the word-identity probe from the pilot (`src/08_lexical_control.py`).
6. **Sensitivity.** Each of these is excluded in turn:
   - (a) participants with retakes;
   - (b) segmentation-critical flags that the listening review did not resolve;
   - (c) the minority recording configuration;
   - (d) LOW_SBR.

## 7. Data rules (same as the pilot pipeline)

- **Exclusion:**
  - no usable recording;
  - consent not documented in `consent/consent_log.csv`;
  - a struck-through or invalid rating (that rating only);
  - NO_SPEECH.
- **Takes and boundaries:** the complete-word rule, then the blind listening review (at least two
  listeners). The review is finished **before** the ratings are opened for analysis.
- **Missing F0** (no voiced frames): the participant drops out of the F0 tests only. There is no imputation
  in correlation tests, and median imputation inside the pipeline for machine learning.

## 8. Not planned, on purpose

- Feature selection or hyperparameter tuning on the full data.
- Oversampling (SMOTE) or any resampling outside training folds.
- Subgroup analyses by subject, activity, session or year, unless they vary in the pooled data and are added
  here *before* the analysis.
- Changing the derived class boundaries after seeing results.

## References

- Banse, R., & Scherer, K. R. (1996). Acoustic profiles in vocal emotion expression. *Journal of Personality
  and Social Psychology, 70*(3), 614–636.
- Eyben, F., et al. (2016). The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for voice research and
  affective computing. *IEEE Transactions on Affective Computing, 7*(2), 190–202.
- Fieller, E. C., Hartley, H. O., & Pearson, E. S. (1957). Tests for rank correlation coefficients. I.
  *Biometrika, 44*(3/4), 470–481.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of
  Statistics, 6*(2), 65–70.
- Juslin, P. N., & Laukka, P. (2003). Communication of emotions in vocal expression and music performance:
  Different channels, same code? *Psychological Bulletin, 129*(5), 770–814.
- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*, 239–281.
- Scherer, K. R. (2003). Vocal communication of emotion: A review of research paradigms. *Speech
  Communication, 40*(1–2), 227–256.
