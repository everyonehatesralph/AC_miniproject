# Presentation Technical Checking Bundle

Open this folder only during the live checking.

## Code order
1. `code/03_prepare_audio.py` — preprocess and locate the spoken word
2. `code/04_extract_features.py` — extract 108 acoustic features
3. `code/06_train_models.py` — train Logistic Regression, SVM, and Random Forest
4. `code/07_evaluate_models.py` — calculate model performance and statistics
5. `code/common.py` — shared settings/functions
6. `code/viz.py` — shared plotting functions

## Data / feature proof
- `data/analysis_manifest.csv`
- `data/acoustic_features.csv`
- `data/acoustic_features_with_labels.csv`

## Model proof
- `models/oof_predictions.csv`
- `models/cv_fold_scores.csv`
- `models/permutation_tests.csv`

## Final results
- `results/summary_metrics.csv`
- `results/model_evaluation.md`

### Demo flow
Audio preparation → 108 features → model training → model evaluation → final metrics.
