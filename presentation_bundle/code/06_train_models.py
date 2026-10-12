"""
06_train_models.py -- PHASES 8-11: leakage-safe, speaker-independent model training.

Inputs   data/acoustic_features.csv (X), data/metadata_cleaned.csv (y), data/analysis_manifest.csv (subsets)
Outputs  models/oof_predictions.csv      out-of-fold prediction for every participant, repeat, model, task
         models/cv_fold_scores.csv       test-fold metrics + chosen hyper-parameters
         models/cv_splits.csv            outer-fold assignment (participant, task, repeat, fold)
         models/permutation_tests.csv    label-permutation test of every learned model vs chance
         models/sensitivity_*.csv        same design on pre-defined subsets
         models/final/*.joblib           each model refit on all data (for reuse; NOT the performance estimate)
         models/training_summary.json

Design (Protocol Sections XV-XVI)
    X       the 108 acoustic features only. The script refuses to run if any metadata column
            (spoken word, feeling, year, subject, session, activity, code, file name) is in X.
    y       derived classes of the participants' own ratings (common.TARGETS); a task runs only if
            every class has >= MIN_CLASS_SIZE participants -- otherwise it is reported, not forced.
    groups  participant_code. Outer CV = StratifiedGroupKFold(k = min(5, smallest class)) repeated
            10x with different shuffles; every split is checked to share no participant.
    tuning  inside each outer training fold only (GridSearchCV on an inner StratifiedGroupKFold);
            imputation and scaling are Pipeline steps fitted on training folds only.
    imbalance  class_weight='balanced'; no resampling, no synthetic samples.
    chance  DummyClassifier (majority, stratified) under the identical CV, plus a label-permutation
            test (500 permutations) of each learned model's pooled balanced accuracy.

Usage: .venv/Scripts/python.exe src/06_train_models.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (CLEAN_METADATA_CSV, DATA, FEATURES_CSV, MIN_CLASS_SIZE, MODELS, N_OUTER_FOLDS, N_PERMUTATIONS,
                    N_REPEATS, RANDOM_STATE, REPORTS, TARGETS, TASK_ROLE, derive_target, rel, setup_logging, sha256_of)

MANIFEST = DATA / "analysis_manifest.csv"
ID_COLS = ("participant_code", "segment_file", "segment_sha256")
FORBIDDEN = ("participant", "code", "file", "sha", "word", "feeling", "year", "subject", "session", "activity",
             "consent", "collector", "date", "notes", "valence", "arousal")
LEARNED = ("logistic_regression", "svm_rbf", "random_forest")
SCENARIOS = {  # sensitivity analyses: (manifest column whose True rows are removed, description)
    "no_retakes": ("subset_retake", "participants with retakes removed"),
    "no_config_B": ("subset_config_or_time_anomaly", "different recording configuration / capture-time outliers removed"),
    "high_confidence_segmentation": ("subset_segmentation_critical", "segmentation-critical flags removed"),
    "no_unverified_source": ("subset_no_original", "supplied files without a verified original recording removed"),
    "no_edited_or_clipped": ("subset_edited_or_clipped", "files with inserted digital silence or clipping removed"),
}  # a scenario that would remove nobody (e.g. no retakes in a supplied delivery) is skipped
SENSITIVITY_TASKS = ("arousal_3", "valence_2")

log = setup_logging("train", REPORTS / "logs" / "06_train_models.log")


def model_specs() -> dict:
    def pipe(clf, scale=True):
        steps = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps + [("clf", clf)])
    return {
        "dummy_majority": (DummyClassifier(strategy="most_frequent"), {}),
        "dummy_stratified": (DummyClassifier(strategy="stratified", random_state=RANDOM_STATE), {}),
        "logistic_regression": (pipe(LogisticRegression(class_weight="balanced", max_iter=10000,
                                                        random_state=RANDOM_STATE)),
                                {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
        "svm_rbf": (pipe(SVC(kernel="rbf", gamma="scale", class_weight="balanced")), {"clf__C": [0.1, 1.0, 10.0]}),
        "random_forest": (pipe(RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced",
                                                      random_state=RANDOM_STATE, n_jobs=1), scale=False),
                          {"clf__min_samples_leaf": [1, 3]}),
    }


def check_no_leakage(feature_cols: list[str], dictionary: set[str]) -> None:
    bad = [c for c in feature_cols if c not in dictionary or any(tok in c for tok in FORBIDDEN)]
    if bad:
        raise RuntimeError(f"non-acoustic columns in X (leakage): {bad}")


def outer_splits(y: np.ndarray, groups: np.ndarray, k: int, seed: int) -> list:
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    splits = list(cv.split(np.zeros(len(y)), y, groups))
    for tr, te in splits:
        if set(groups[tr]) & set(groups[te]):
            raise RuntimeError("a participant appears in both training and test folds")
        if set(y[te]) != set(y):
            raise RuntimeError(f"a test fold lacks a class (seed {seed}); stratification failed")
    if sorted(np.concatenate([te for _, te in splits]).tolist()) != list(range(len(y))):
        raise RuntimeError("outer folds do not partition the participants")
    return splits


def inner_cv(y_train: np.ndarray, groups_train: np.ndarray, seed: int):
    k = int(min(3, pd.Series(y_train).value_counts().min()))
    if k < 2:
        return None
    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(
        np.zeros(len(y_train)), y_train, groups_train))


def fit_predict(name: str, X, y, groups, tr, te, seed: int):
    est, grid = model_specs()[name]
    est = clone(est)
    params = {}
    if grid:
        cv = inner_cv(y[tr], groups[tr], seed)
        if cv is not None:
            gs = GridSearchCV(est, grid, cv=cv, scoring="balanced_accuracy", refit=True, n_jobs=1, error_score="raise")
            gs.fit(X[tr], y[tr])
            est, params = gs.best_estimator_, gs.best_params_
        else:
            est.fit(X[tr], y[tr])
    else:
        est.fit(X[tr], y[tr])
    return est.predict(X[te]), params


def fold_metrics(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred), "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def run_cv(task: str, name: str, X, y, codes, repeat: int, k: int, scenario: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        seed = RANDOM_STATE + repeat
        preds, scores = [], []
        for fold, (tr, te) in enumerate(outer_splits(y, codes, k, seed)):
            y_hat, params = fit_predict(name, X, y, codes, tr, te, seed)
            for c, t, p in zip(codes[te], y[te], y_hat):
                preds.append({"scenario": scenario, "task": task, "model": name, "repeat": repeat, "fold": fold,
                              "participant_code": c, "y_true": t, "y_pred": p})
            scores.append({"scenario": scenario, "task": task, "model": name, "repeat": repeat, "fold": fold,
                           "n_train": len(tr), "n_test": len(te), **fold_metrics(y[te], y_hat),
                           "best_params": json.dumps(params, sort_keys=True)})
        return preds, scores


def pooled_ba(task, name, X, y, codes, k, seed=RANDOM_STATE) -> float:
    """Balanced accuracy of the pooled out-of-fold predictions for one CV repeat."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        y_hat = np.empty(len(y), dtype=object)
        for tr, te in outer_splits(y, codes, k, seed):
            y_hat[te], _ = fit_predict(name, X, y, codes, tr, te, seed)
        return balanced_accuracy_score(y, y_hat)


def perm_one(task, name, X, y, codes, k, i) -> float:
    rng = np.random.default_rng(RANDOM_STATE * 1000 + i)
    return pooled_ba(task, name, X, rng.permutation(y), codes, k)


def task_data(task: str, feats: pd.DataFrame, meta: pd.DataFrame, feature_cols: list[str], drop: set | None = None):
    df = feats.merge(meta[["participant_code", TARGETS[task]["source"]]], on="participant_code",
                     validate="one_to_one").copy()
    if drop:
        df = df[~df["participant_code"].isin(drop)]
    df = df.assign(target=derive_target(df[TARGETS[task]["source"]], task)).dropna(subset=["target"])
    df = df.sort_values("participant_code").reset_index(drop=True)
    counts = df["target"].value_counts().reindex(TARGETS[task]["order"], fill_value=0)
    viable = bool(counts.min() >= MIN_CLASS_SIZE)
    k = int(min(N_OUTER_FOLDS, counts.min()))
    return (df[feature_cols].to_numpy(dtype=float), df["target"].to_numpy(dtype=object),
            df["participant_code"].to_numpy(dtype=object), counts, viable, k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 repeats, 20 permutations (smoke test only)")
    args = ap.parse_args()
    n_repeats = 2 if args.quick else N_REPEATS
    n_perm = 20 if args.quick else N_PERMUTATIONS
    t0 = time.time()
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / "final").mkdir(exist_ok=True)

    feats = pd.read_csv(FEATURES_CSV)
    meta = pd.read_csv(CLEAN_METADATA_CSV, encoding="utf-8-sig", usecols=["participant_code", "valence", "arousal"])
    manifest = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    dictionary = set(pd.read_csv(DATA / "acoustic_features_dictionary.csv")["feature"])
    feature_cols = [c for c in feats.columns if c not in ID_COLS]
    check_no_leakage(feature_cols, dictionary)
    if set(feats["participant_code"]) != set(manifest.loc[manifest["include_primary"] == "True", "participant_code"]):
        raise RuntimeError("feature rows do not match the included participants in the manifest")
    specs = model_specs()

    task_info, jobs = {}, []
    for task in TARGETS:
        X, y, codes, counts, viable, k = task_data(task, feats, meta, feature_cols)
        task_info[task] = {"role": TASK_ROLE[task], "n": int(len(y)), "class_counts": counts.to_dict(),
                           "viable": viable, "outer_folds": k if viable else None,
                           "reason": "" if viable else f"smallest class has {int(counts.min())} < {MIN_CLASS_SIZE} participants"}
        if not viable:
            log.info("task %s NOT modelled: %s", task, task_info[task]["reason"])
            continue
        log.info("task %s: n=%d classes=%s outer folds=%d", task, len(y), counts.to_dict(), k)
        for name in specs:
            for r in range(n_repeats):
                jobs.append((task, name, X, y, codes, r, k, "all"))

    # sensitivity scenarios (primary tasks, all models, same design)
    for scen, (col, _) in SCENARIOS.items():
        drop = set(manifest.loc[manifest[col] == "True", "participant_code"]) if col in manifest else set()
        if not drop:
            log.info("sensitivity %s: nobody to remove, skipped", scen)
            continue
        for task in SENSITIVITY_TASKS:
            X, y, codes, counts, viable, k = task_data(task, feats, meta, feature_cols, drop)
            task_info.setdefault("sensitivity", {}).setdefault(scen, {})[task] = {
                "n": int(len(y)), "removed": sorted(drop), "class_counts": counts.to_dict(), "viable": viable,
                "outer_folds": k if viable else None}
            if not viable:
                log.info("sensitivity %s/%s not viable: %s", scen, task, counts.to_dict())
                continue
            for name in specs:
                for r in range(n_repeats):
                    jobs.append((task, name, X, y, codes, r, k, scen))

    log.info("running %d CV jobs on %d workers", len(jobs), joblib.cpu_count())
    results = Parallel(n_jobs=-1, verbose=0)(delayed(run_cv)(*j) for j in jobs)
    preds = pd.DataFrame([p for pr, _ in results for p in pr])
    scores = pd.DataFrame([s for _, sc in results for s in sc])
    preds[preds["scenario"] == "all"].drop(columns="scenario").to_csv(MODELS / "oof_predictions.csv", index=False)
    scores[scores["scenario"] == "all"].drop(columns="scenario").to_csv(MODELS / "cv_fold_scores.csv", index=False)
    preds[preds["scenario"] != "all"].to_csv(MODELS / "sensitivity_oof_predictions.csv", index=False)
    scores[scores["scenario"] != "all"].to_csv(MODELS / "sensitivity_fold_scores.csv", index=False)
    log.info("CV done in %.0f s", time.time() - t0)

    # outer split assignments (for audit)
    split_rows = []
    for task, info in task_info.items():
        if task == "sensitivity" or not info["viable"]:
            continue
        X, y, codes, *_ = task_data(task, feats, meta, feature_cols)
        for r in range(n_repeats):
            for fold, (_, te) in enumerate(outer_splits(y, codes, info["outer_folds"], RANDOM_STATE + r)):
                split_rows += [{"task": task, "repeat": r, "fold": fold, "participant_code": c} for c in codes[te]]
    pd.DataFrame(split_rows).to_csv(MODELS / "cv_splits.csv", index=False)

    # permutation tests (repeat 0 split seed; statistic = pooled out-of-fold balanced accuracy)
    perm_rows = []
    for task, info in task_info.items():
        if task == "sensitivity" or not info["viable"]:
            continue
        X, y, codes, counts, viable, k = task_data(task, feats, meta, feature_cols)
        for name in LEARNED:
            observed = pooled_ba(task, name, X, y, codes, k)
            null = np.array(Parallel(n_jobs=-1)(delayed(perm_one)(task, name, X, y, codes, k, i) for i in range(n_perm)))
            p = (1 + int((null >= observed - 1e-12).sum())) / (1 + n_perm)
            perm_rows.append({"task": task, "model": name, "observed_pooled_ba_repeat0": observed,
                              "null_mean": float(null.mean()), "null_sd": float(null.std()),
                              "null_p95": float(np.percentile(null, 95)), "n_permutations": n_perm, "p_value": p})
            log.info("permutation %s/%s: observed %.3f, null mean %.3f, p = %.4f", task, name, observed, null.mean(), p)
    pd.DataFrame(perm_rows).to_csv(MODELS / "permutation_tests.csv", index=False)

    # final models refit on all data (inner-CV tuning on the full set), for reuse
    cards = []
    for task, info in task_info.items():
        if task == "sensitivity" or not info["viable"]:
            continue
        X, y, codes, *_ = task_data(task, feats, meta, feature_cols)
        for name in LEARNED:
            est, grid = specs[name]
            cv = inner_cv(y, codes, RANDOM_STATE)
            gs = GridSearchCV(clone(est), grid, cv=cv, scoring="balanced_accuracy", refit=True, n_jobs=-1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gs.fit(X, y)
            path = MODELS / "final" / f"{task}__{name}.joblib"
            joblib.dump({"model": gs.best_estimator_, "features": feature_cols, "classes": list(gs.classes_),
                         "task": task, "task_definition": {str(k_): v for k_, v in TARGETS[task]["map"].items()}},
                        path)
            cards.append({"task": task, "model": name, "file": rel(path), "best_params": gs.best_params_,
                          "inner_cv_balanced_accuracy": float(gs.best_score_), "n_train": int(len(y))})
    summary = {
        "sklearn_version": sklearn.__version__, "random_state": RANDOM_STATE, "n_repeats": n_repeats,
        "n_permutations": n_perm, "min_class_size": MIN_CLASS_SIZE, "n_features": len(feature_cols),
        "features_sha256": sha256_of(FEATURES_CSV), "tasks": task_info, "final_models": cards,
        "grids": {k: {p: v for p, v in g.items()} for k, (_, g) in specs.items()},
        "runtime_s": round(time.time() - t0, 1), "quick_mode": args.quick,
    }
    (MODELS / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("training finished in %.0f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
