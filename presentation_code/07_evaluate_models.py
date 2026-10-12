"""
07_evaluate_models.py -- PHASE 12: model evaluation from the saved out-of-fold predictions.

Inputs   models/oof_predictions.csv, models/cv_fold_scores.csv, models/permutation_tests.csv,
         models/sensitivity_*.csv, models/training_summary.json
Outputs  reports/model_evaluation.md (generated), reports/evaluation/*.csv + evaluation_summary.json,
         figures/evaluation/*.png

Reported for every task x model
    * fold-level mean +/- SD (and min-max) over all repeats x folds: accuracy, balanced accuracy,
      macro precision, macro recall, macro F1
    * pooled out-of-fold metrics (all participants predicted once per repeat), averaged over repeats,
      with a participant-level stratified bootstrap 95% interval (2000 resamples)
    * label-permutation p-value of the pooled balanced accuracy (from 06), Holm-adjusted over all
      model-vs-chance tests
    * per-class precision / recall / F1 and the confusion matrix (summed over repeats, shown per repeat)
    * pairwise model comparison: corrected repeated k-fold t-test (Nadeau & Bengio, 2003)

Usage: .venv/Scripts/python.exe src/07_evaluate_models.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

import viz
from viz import plt
from common import FIGURES, MODELS, RANDOM_STATE, REPORTS, TARGETS, setup_logging

OUT = REPORTS / "evaluation"
FIG = FIGURES / "evaluation"
MODEL_ORDER = ["dummy_majority", "dummy_stratified", "logistic_regression", "svm_rbf", "random_forest"]
MODEL_LABEL = {"dummy_majority": "Dummy (majority)", "dummy_stratified": "Dummy (stratified random)",
               "logistic_regression": "Logistic regression", "svm_rbf": "SVM (RBF)", "random_forest": "Random forest"}
LEARNED = ["logistic_regression", "svm_rbf", "random_forest"]
METRICS = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"]
N_BOOT = 2000

log = setup_logging("evaluate", REPORTS / "logs" / "07_evaluate_models.log")


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
            return "—"
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False))


def class_metrics(y, p, classes):
    """Per-class precision/recall/F1 and macro averages for vectors y (true) and p (pred)."""
    out = {}
    for c in classes:
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        out[c] = {"precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
                  "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0, "support": int(np.sum(y == c))}
    return out


def pooled_scores(y, P, classes):
    """y: (n,), P: (n, R) predictions -> per-repeat accuracy, balanced accuracy, macro-F1 arrays."""
    acc, ba, f1 = [], [], []
    for r in range(P.shape[1]):
        cm = class_metrics(y, P[:, r], classes)
        acc.append(np.mean(P[:, r] == y))
        ba.append(np.mean([cm[c]["recall"] for c in classes]))
        f1.append(np.mean([cm[c]["f1"] for c in classes]))
    return np.array(acc), np.array(ba), np.array(f1)


def bootstrap_ci(y, P, classes, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    idx_by_class = [np.flatnonzero(y == c) for c in classes]
    ba_b, f1_b = np.empty(N_BOOT), np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by_class])
        _, ba, f1 = pooled_scores(y[idx], P[idx], classes)
        ba_b[b], f1_b[b] = ba.mean(), f1.mean()
    return np.percentile(ba_b, [2.5, 97.5]), np.percentile(f1_b, [2.5, 97.5])


def corrected_ttest(a: np.ndarray, b: np.ndarray, n_train: float, n_test: float):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV."""
    d = a - b
    j = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return float(d.mean()), np.nan, np.nan
    t = d.mean() / np.sqrt((1 / j + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=j - 1)
    return float(d.mean()), float(t), float(p)


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(MODELS / "oof_predictions.csv", dtype={"y_true": str, "y_pred": str})
    folds = pd.read_csv(MODELS / "cv_fold_scores.csv")
    perm = pd.read_csv(MODELS / "permutation_tests.csv")
    train = json.loads((MODELS / "training_summary.json").read_text(encoding="utf-8"))
    sens_oof = pd.read_csv(MODELS / "sensitivity_oof_predictions.csv", dtype={"y_true": str, "y_pred": str})
    tasks = [t for t, i in train["tasks"].items() if t != "sensitivity" and i["viable"]]
    not_run = {t: i["reason"] for t, i in train["tasks"].items() if t != "sensitivity" and not i["viable"]}

    # identical outer splits for every model (required for paired comparisons)
    chk = oof.groupby(["task", "repeat", "participant_code"])["fold"].nunique()
    if (chk != 1).any():
        raise RuntimeError("models were not evaluated on identical outer folds")

    summary_rows, per_class_rows, cm_store, comp_rows, hp_rows = [], [], {}, [], []
    perm = perm.assign(p_holm=holm(perm["p_value"].tolist()))
    for task in tasks:
        classes = TARGETS[task]["order"]
        k_chance = 1 / len(classes)
        for model in MODEL_ORDER:
            d = oof[(oof["task"] == task) & (oof["model"] == model)]
            piv = d.pivot(index="participant_code", columns="repeat", values="y_pred").sort_index()
            y = d.drop_duplicates("participant_code").set_index("participant_code").loc[piv.index, "y_true"].to_numpy()
            P = piv.to_numpy()
            acc, ba, f1 = pooled_scores(y, P, classes)
            (ba_lo, ba_hi), (f1_lo, f1_hi) = bootstrap_ci(y, P, classes)
            fs = folds[(folds["task"] == task) & (folds["model"] == model)]
            row = {"task": task, "model": model, "n": int(len(y)), "repeats": int(P.shape[1]),
                   "folds_total": int(len(fs)), "chance_ba": k_chance,
                   "pooled_accuracy": acc.mean(), "pooled_ba": ba.mean(), "pooled_ba_ci_low": ba_lo,
                   "pooled_ba_ci_high": ba_hi, "pooled_ba_repeat_min": ba.min(), "pooled_ba_repeat_max": ba.max(),
                   "pooled_macro_f1": f1.mean(), "pooled_f1_ci_low": f1_lo, "pooled_f1_ci_high": f1_hi}
            for m in METRICS:
                row[f"fold_{m}_mean"] = fs[m].mean()
                row[f"fold_{m}_sd"] = fs[m].std(ddof=1)
            row["fold_ba_min"], row["fold_ba_max"] = fs["balanced_accuracy"].min(), fs["balanced_accuracy"].max()
            pr = perm[(perm["task"] == task) & (perm["model"] == model)]
            row["perm_p"] = float(pr["p_value"].iloc[0]) if len(pr) else np.nan
            row["perm_p_holm"] = float(pr["p_holm"].iloc[0]) if len(pr) else np.nan
            row["perm_observed_ba_repeat0"] = float(pr["observed_pooled_ba_repeat0"].iloc[0]) if len(pr) else np.nan
            summary_rows.append(row)
            # per-class (averaged over repeats) and confusion matrix (summed over repeats)
            pcs = [class_metrics(y, P[:, r], classes) for r in range(P.shape[1])]
            for c in classes:
                per_class_rows.append({"task": task, "model": model, "class": c, "support": pcs[0][c]["support"],
                                       **{m: float(np.mean([pc[c][m] for pc in pcs])) for m in ("precision", "recall", "f1")}})
            cm = np.zeros((len(classes), len(classes)))
            for r in range(P.shape[1]):
                for i, ct in enumerate(classes):
                    for j, cp in enumerate(classes):
                        cm[i, j] += np.sum((y == ct) & (P[:, r] == cp))
            cm_store[(task, model)] = cm / P.shape[1]
            for params, cnt in fs["best_params"].value_counts().items():
                hp_rows.append({"task": task, "model": model, "best_params": params, "outer_folds_chosen": int(cnt),
                                "share": cnt / len(fs)})
        # pairwise comparisons on fold-level balanced accuracy
        n_train = folds[(folds["task"] == task)]["n_train"].mean()
        n_test = folds[(folds["task"] == task)]["n_test"].mean()
        pairs = [(a, b) for i, a in enumerate(LEARNED) for b in LEARNED[i + 1:]] + [(m, "dummy_majority") for m in LEARNED]
        for a, b in pairs:
            fa = folds[(folds["task"] == task) & (folds["model"] == a)].sort_values(["repeat", "fold"])
            fb = folds[(folds["task"] == task) & (folds["model"] == b)].sort_values(["repeat", "fold"])
            diff, t, p = corrected_ttest(fa["balanced_accuracy"].to_numpy(), fb["balanced_accuracy"].to_numpy(),
                                         n_train, n_test)
            comp_rows.append({"task": task, "model_a": a, "model_b": b, "mean_ba_difference": diff, "corrected_t": t,
                              "p_value": p})
    summary = pd.DataFrame(summary_rows)
    per_class = pd.DataFrame(per_class_rows)
    comps = pd.DataFrame(comp_rows)
    hp = pd.DataFrame(hp_rows)
    summary.to_csv(OUT / "summary_metrics.csv", index=False)
    per_class.to_csv(OUT / "per_class_metrics.csv", index=False)
    comps.to_csv(OUT / "model_comparison.csv", index=False)
    hp.to_csv(OUT / "hyperparameter_choices.csv", index=False)
    pd.DataFrame([{"task": t, "model": m, "true_class": TARGETS[t]["order"][i], **{f"pred_{c}": cm[i, j] for j, c in
                                                                                 enumerate(TARGETS[t]["order"])}}
                  for (t, m), cm in cm_store.items() for i in range(cm.shape[0])]).to_csv(
        OUT / "confusion_matrices_mean_per_repeat.csv", index=False)

    # sensitivity
    sens_rows = []
    for (scen, task, model), d in sens_oof.groupby(["scenario", "task", "model"]):
        classes = TARGETS[task]["order"]
        piv = d.pivot(index="participant_code", columns="repeat", values="y_pred").sort_index()
        y = d.drop_duplicates("participant_code").set_index("participant_code").loc[piv.index, "y_true"].to_numpy()
        acc, ba, f1 = pooled_scores(y, piv.to_numpy(), classes)
        prim = summary[(summary["task"] == task) & (summary["model"] == model)]["pooled_ba"].iloc[0]
        sens_rows.append({"scenario": scen, "task": task, "model": model, "n": int(len(y)), "pooled_ba": ba.mean(),
                          "pooled_macro_f1": f1.mean(), "primary_pooled_ba": prim, "difference_vs_primary": ba.mean() - prim})
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(OUT / "sensitivity_summary.csv", index=False)

    # ---- figures ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6), sharey=True)
    for ax, task in zip(np.atleast_1d(axes), tasks):
        s = summary[summary["task"] == task].set_index("model").loc[MODEL_ORDER[::-1]]
        ypos = np.arange(len(s))
        for i, (m, r) in enumerate(s.iterrows()):
            col = viz.MUTED if m.startswith("dummy") else viz.SERIES[0]
            ax.plot([r["pooled_ba_ci_low"], r["pooled_ba_ci_high"]], [i, i], color=col, lw=2)
            ax.scatter(r["pooled_ba"], i, s=48, color=col, edgecolor=viz.SURFACE, linewidth=1.5, zorder=3)
            ax.text(r["pooled_ba_ci_high"] + 0.01, i, f"{r['pooled_ba']:.2f}", va="center", fontsize=7.5, color=viz.INK_2)
        ax.axvline(1 / len(TARGETS[task]["order"]), color=viz.INK_2, lw=1)
        ax.set_yticks(ypos, [MODEL_LABEL[m] for m in s.index])
        ax.set_xlim(0, 1)
        ax.set_xlabel("balanced accuracy (pooled out-of-fold)")
        ax.set_title(f"{task}  (n = {int(s['n'].iloc[0])}; chance = {1 / len(TARGETS[task]['order']):.2f})", fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Speaker-independent balanced accuracy, mean of 10 repeated 5-fold CVs with 95% bootstrap interval; "
                 "vertical line = chance", x=0.01, ha="left", fontsize=9.5, fontweight="semibold")
    fig.tight_layout()
    viz.save(fig, FIG / "fig_eval_balanced_accuracy.png")

    for task in tasks:
        classes = TARGETS[task]["order"]
        models = ["dummy_majority", "logistic_regression", "svm_rbf", "random_forest"]
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
        for ax, m in zip(axes, models):
            cm = cm_store[(task, m)]
            norm = cm / cm.sum(axis=1, keepdims=True)
            im = ax.imshow(norm, cmap=viz.SEQUENTIAL, vmin=0, vmax=1)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, f"{norm[i, j]:.0%}\n({cm[i, j]:.1f})", ha="center", va="center", fontsize=7.5,
                            color=viz.ink_for(im.cmap(im.norm(norm[i, j]))))
            ax.set_xticks(range(len(classes)), classes, fontsize=7.5)
            ax.set_yticks(range(len(classes)), classes, fontsize=7.5)
            ax.set_xlabel("predicted")
            ax.set_title(MODEL_LABEL[m], fontsize=9)
            ax.grid(False)
        axes[0].set_ylabel("true (self-report)")
        fig.suptitle(f"{task}: confusion matrices (row %, mean count per repeat in brackets)", x=0.01, ha="left",
                     fontsize=9.5, fontweight="semibold")
        fig.tight_layout()
        viz.save(fig, FIG / f"fig_eval_confusion_{task}.png")

    if not sens.empty:
        fig, axes = plt.subplots(1, sens["task"].nunique(), figsize=(11, 3.4), sharey=True)
        for ax, (task, d) in zip(np.atleast_1d(axes), sens.groupby("task")):
            scen = ["primary (all)"] + sorted(d["scenario"].unique())
            for k_, m in enumerate(["dummy_majority", "logistic_regression", "svm_rbf", "random_forest"]):
                vals = [summary[(summary["task"] == task) & (summary["model"] == m)]["pooled_ba"].iloc[0]] + \
                       [d[(d["scenario"] == s) & (d["model"] == m)]["pooled_ba"].iloc[0]
                        if len(d[(d["scenario"] == s) & (d["model"] == m)]) else np.nan for s in scen[1:]]
                # scenarios are categories, not a sequence: offset dots, no connecting lines
                ax.scatter(np.arange(len(scen)) + (k_ - 1.5) * 0.12, vals, s=42, zorder=3,
                           color=viz.MUTED if m.startswith("dummy") else viz.SERIES[k_ - 1],
                           edgecolor=viz.SURFACE, linewidth=1.2, label=MODEL_LABEL[m])
            ax.axhline(1 / len(TARGETS[task]["order"]), color=viz.INK_2, lw=1)
            labels = [s.replace("_", " ") + (f"\n(n = {int(d[d['scenario'] == s]['n'].iloc[0])})" if s in set(d['scenario']) else
                      f"\n(n = {int(summary[summary['task'] == task]['n'].iloc[0])})") for s in scen]
            ax.set_xticks(range(len(scen)), labels, fontsize=7)
            ax.set_title(task, fontsize=9)
            ax.set_ylim(0, 1)
            ax.grid(axis="x", visible=False)
        np.atleast_1d(axes)[0].set_ylabel("pooled balanced accuracy")
        np.atleast_1d(axes)[-1].legend(loc="upper right", fontsize=7)
        fig.suptitle("Sensitivity analyses: same CV design on pre-defined subsets (horizontal line = chance)", x=0.01,
                     ha="left", fontsize=9.5, fontweight="semibold")
        fig.tight_layout()
        viz.save(fig, FIG / "fig_eval_sensitivity.png")

    # ---- report ----------------------------------------------------------------------------
    any_sig = bool((summary["perm_p_holm"] < 0.05).any())
    sig_rows = summary[summary["perm_p_holm"] < 0.05]
    comp_sig = comps[(comps["p_value"] < 0.05) & comps["model_b"].ne("dummy_majority")]
    tab = summary.assign(
        model=summary["model"].map(MODEL_LABEL),
        **{"BA pooled [95% CI]": summary.apply(lambda r: f"{r.pooled_ba:.3f} [{r.pooled_ba_ci_low:.3f}, {r.pooled_ba_ci_high:.3f}]", axis=1),
           "macro-F1 pooled [95% CI]": summary.apply(lambda r: f"{r.pooled_macro_f1:.3f} [{r.pooled_f1_ci_low:.3f}, {r.pooled_f1_ci_high:.3f}]", axis=1),
           "accuracy pooled": summary["pooled_accuracy"],
           "BA folds mean ± SD (min–max)": summary.apply(lambda r: f"{r.fold_balanced_accuracy_mean:.3f} ± {r.fold_balanced_accuracy_sd:.3f} ({r.fold_ba_min:.2f}–{r.fold_ba_max:.2f})", axis=1),
           "perm. p": summary["perm_p"], "perm. p (Holm)": summary["perm_p_holm"]})
    fold_tab = summary.assign(model=summary["model"].map(MODEL_LABEL))[
        ["task", "model"] + [f"fold_{m}_mean" for m in METRICS] + [f"fold_{m}_sd" for m in METRICS]]
    lines = [
        "# Model Evaluation (Phases 8–12)",
        "",
        f"*Generated by `src/07_evaluate_models.py` from `models/` outputs of `src/06_train_models.py` "
        f"(scikit-learn {train['sklearn_version']}, random_state {train['random_state']}). Tables: `reports/evaluation/`.*",
        "",
        "## 1. Design",
        "",
        f"- **Input (X):** the {train['n_features']} acoustic/prosodic features only (`data/acoustic_features.csv`). "
        "The training script refuses to run if any metadata column (spoken word, feeling text, year, subject, "
        "session, activity, participant code, file name) is in X.",
        "- **Target (y):** derived classes of the participants' own 1–5 ratings (Phase 8; original ratings kept).",
        f"- **Speaker-independent evaluation:** StratifiedGroupKFold grouped by participant, {train['n_repeats']} "
        "repeats with different shuffles. Every split is verified to share no participant between training and "
        "test, and every test fold contains every class.",
        "- **No leakage:** median imputation and standardisation are Pipeline steps fitted on training folds only. "
        "Hyperparameters are tuned by GridSearchCV on an inner grouped CV inside each outer training fold. There is "
        "no resampling and no synthetic data; class imbalance is handled with `class_weight='balanced'`.",
        f"- **Grids (deliberately small):** {json.dumps(train['grids'])}.",
        f"- **Chance:** two DummyClassifiers under the identical CV, plus a label-permutation test "
        f"({train['n_permutations']} permutations, full nested procedure repeated on permuted labels; statistic: "
        "pooled out-of-fold balanced accuracy of repeat 0). p-values are Holm-adjusted over all "
        f"{len(perm)} model-vs-chance tests.",
        "",
        "## 2. Targets",
        "",
        md_table(pd.DataFrame([{"task": t, "role": i["role"], "n": i["n"], "classes": "; ".join(f"{k}: {v}" for k, v in i["class_counts"].items()),
                                "modelled": "yes" if i["viable"] else "no", "reason / outer folds": i["reason"] or i["outer_folds"]}
                               for t, i in train["tasks"].items() if t != "sensitivity"])),
        "",
        f"Not modelled: **{', '.join(not_run) or 'none'}**. Each has at least one class with fewer than 5 "
        "participants, so stratified test folds and per-class metrics would be meaningless. This is reported "
        "rather than forced (Phase 8).",
        "",
        "## 3. Results",
        "",
        md_table(tab[["task", "model", "BA pooled [95% CI]", "macro-F1 pooled [95% CI]", "accuracy pooled",
                      "BA folds mean ± SD (min–max)", "perm. p", "perm. p (Holm)"]]),
        "",
        "BA = balanced accuracy. *Pooled* means all participants' out-of-fold predictions within a repeat, "
        "averaged over repeats; the 95% interval is a participant-level stratified bootstrap. Fold-level metrics "
        "(mean ± SD over all test folds):",
        "",
        md_table(fold_tab),
        "",
        "![balanced accuracy](../figures/evaluation/fig_eval_balanced_accuracy.png)",
        "",
        *[(f"*Note on the majority baseline for {r.task}:* its fold-level balanced accuracy is exactly "
           f"{r.fold_balanced_accuracy_mean:.3f} (= 1/{len(TARGETS[r.task]['order'])}). Its pooled value "
           f"({r.pooled_ba:.3f}) differs because the majority class of the training folds is not the same in every "
           f"fold (class counts: {', '.join(f'{k}: {v}' for k, v in train['tasks'][r.task]['class_counts'].items())}). "
           "Pooled predictions therefore mix two constant guesses.\n")
          for r in summary[(summary["model"] == "dummy_majority")].itertuples()
          if abs(r.pooled_ba - r.fold_balanced_accuracy_mean) > 1e-9],
        "### Per-class metrics (mean over repeats)",
        "",
        md_table(per_class.assign(model=per_class["model"].map(MODEL_LABEL))),
        "",
        "### Confusion matrices",
        "",
        *[f"![{t}](../figures/evaluation/fig_eval_confusion_{t}.png)\n" for t in tasks],
        "## 4. Statistical comparison of models (RQ5)",
        "",
        "Corrected repeated k-fold t-test on fold-level balanced accuracy (Nadeau & Bengio, 2003). This accounts "
        "for the overlap between training sets across folds and repeats, which naive t-tests ignore.",
        "",
        md_table(comps.assign(model_a=comps["model_a"].map(MODEL_LABEL), model_b=comps["model_b"].map(MODEL_LABEL))),
        "",
        "### Hyperparameters chosen in the outer folds",
        "",
        md_table(hp.assign(model=hp["model"].map(MODEL_LABEL)), "{:.2f}"),
        "",
        "## 5. Sensitivity analyses",
        "",
        "Same design on pre-defined subsets: retake participants removed; recordings with a different "
        "configuration or capture-time outliers removed; recordings with segmentation-critical flags removed.",
        "",
        md_table(sens.assign(model=sens["model"].map(MODEL_LABEL))),
        "",
        "![sensitivity](../figures/evaluation/fig_eval_sensitivity.png)" if not sens.empty else "",
        "",
        "## 6. What the numbers support",
        "",
        ("- After Holm correction, **" + ", ".join(f"{MODEL_LABEL[r.model]} on {r.task}" for r in sig_rows.itertuples())
         + "** exceeded the permutation null (p < 0.05).") if any_sig else
        "- **No learned model exceeded the label-permutation null after Holm correction** (all adjusted p ≥ 0.05). "
        "Within this pilot dataset there is no reliable evidence that these acoustic features predict the "
        "self-reported affect of unseen speakers.",
        ("- Pairwise model differences with uncorrected p < 0.05: "
         + "; ".join(f"{r.task}: {MODEL_LABEL[r.model_a]} vs {MODEL_LABEL[r.model_b]} (Δ = {r.mean_ba_difference:+.3f})"
                     for r in comp_sig.itertuples()) + ".") if len(comp_sig) else
        "- **No pair of learned models differs significantly** (corrected t-test, all p ≥ 0.05). Differences in "
        "the table are within resampling noise, so no algorithm can be called best for this dataset.",
        "- Bootstrap intervals are wide (n ≈ 30), and fold-level scores vary widely between folds. Any single "
        "point estimate is fragile.",
        "",
    ]
    (REPORTS / "model_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    ev = {"tasks_modelled": tasks, "tasks_not_modelled": not_run, "any_model_significant_after_holm": any_sig,
          "significant_after_holm": sig_rows[["task", "model", "pooled_ba", "perm_p", "perm_p_holm"]].to_dict("records"),
          "pairwise_uncorrected_p_lt_05": comp_sig.to_dict("records"),
          "summary": summary.to_dict("records"), "sensitivity": sens.to_dict("records")}
    (OUT / "evaluation_summary.json").write_text(json.dumps(ev, indent=2, default=float), encoding="utf-8")
    log.info("evaluation written; any model significant after Holm: %s", any_sig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
