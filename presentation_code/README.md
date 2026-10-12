# Presentation Code

Use this folder for the live technical checking.

Order:
1. 03_prepare_audio.py — prepares and checks audio; finds the spoken-word segment.
2. 04_extract_features.py — converts each recording into 108 acoustic features.
3. 06_train_models.py — trains Logistic Regression, SVM, and Random Forest.
4. 07_evaluate_models.py — calculates balanced accuracy, precision, recall, F1, confidence intervals, p-values, and confusion matrices.
5. common.py — shared settings and helper functions.
6. viz.py — shared plotting functions.

Quick flow: 03 Prepare audio → 04 Extract 108 features → 06 Train models → 07 Evaluate results.
