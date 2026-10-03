"""
08_lexical_control.py -- PHASE 13: lexically controlled analysis (Protocol Section XVII).

Question: do acoustic/prosodic differences track self-reported affect when the spoken word is held
constant? The word is used only to SELECT recordings; it is never a model input.

Inputs   data/acoustic_features.csv, data/metadata_cleaned.csv, models/oof_predictions.csv,
         reports/eda/spearman_feature_vs_rating.csv
Outputs  reports/lexical_control_analysis.md (generated), reports/lexical/*.csv, figures/lexical/*.png

Analyses
    A. Feasibility: recordings per spoken word and the affect spread within each word.
    B. Within-word association for every word with >= MIN_WORD_N recordings and rating variation:
       Spearman correlation of the pre-specified prosodic features with valence / arousal,
       permutation p-values (20,000 permutations), BH-FDR; direction compared with the full sample.
    C. Lexical-confound probe: can the same acoustic features (same leakage-safe pipeline) tell the
       word 'Kapoy' from other words? Repeated stratified grouped 5-fold CV + 500-permutation test.
    D. Word-affect association (Fisher exact / permutation chi-square) and the primary models'
       out-of-fold accuracy for Kapoy vs other recordings.

Usage: .venv/Scripts/python.exe src/08_lexical_control.py
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import viz
from viz import plt
from common import (CLEAN_METADATA_CSV, FEATURES_CSV, FIGURES, MODELS, RANDOM_STATE, REPORTS, derive_target,
                    setup_logging)

OUT = REPORTS / "lexical"
FIG = FIGURES / "lexical"
ID_COLS = ("participant_code", "segment_file", "segment_sha256")
PRESPECIFIED = ["f0_median", "f0_range_st", "rms_db_mean", "utterance_duration_s", "spectral_centroid_hz_mean",
                "voiced_ratio"]  # same six features fixed in advance in 05_eda.py
MIN_WORD_N = 8
N_PERM_CORR = 20000
N_PERM_PROBE = 500
N_REPEATS = 10

log = setup_logging("lexical", REPORTS / "logs" / "08_lexical_control.log")


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
            return "—"
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False))


def spearman_perm(x, y, n_perm=N_PERM_CORR, seed=RANDOM_STATE):
    rho = stats.spearmanr(x, y)[0]
    rng = np.random.default_rng(seed)
    null = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)])
    return float(rho), float((1 + np.sum(np.abs(null) >= abs(rho) - 1e-12)) / (1 + n_perm))


def probe_pipeline():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=10000))])


def probe_ba(X, y, groups, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        y_hat = np.empty(len(y), dtype=object)
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, groups):
            if set(groups[tr]) & set(groups[te]):
                raise RuntimeError("participant in both train and test")
            y_hat[te] = probe_pipeline().fit(X[tr], y[tr]).predict(X[te])
        return balanced_accuracy_score(y, y_hat)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(FEATURES_CSV)
    meta = pd.read_csv(CLEAN_METADATA_CSV, encoding="utf-8-sig", keep_default_na=False, na_values=[""],
                       dtype={"spoken_word": str})
    meta["spoken_word"] = meta["spoken_word"].fillna("")
    feature_cols = [c for c in feats.columns if c not in ID_COLS]
    data = feats.merge(meta[["participant_code", "spoken_word", "valence", "arousal"]], on="participant_code",
                       validate="one_to_one").copy()
    data["arousal_3"] = derive_target(data["arousal"], "arousal_3")
    data["valence_2"] = derive_target(data["valence"], "valence_2")
    full_sp = pd.read_csv(REPORTS / "eda" / "spearman_feature_vs_rating.csv")

    # A. feasibility
    known = data[data["spoken_word"] != ""]
    feas = (known.groupby("spoken_word").agg(n=("participant_code", "size"),
                                              valence_distinct=("valence", "nunique"),
                                              arousal_distinct=("arousal", "nunique"),
                                              valence_values=("valence", lambda s: ", ".join(map(str, sorted(s.astype(int))))),
                                              arousal_values=("arousal", lambda s: ", ".join(map(str, sorted(s.astype(int))))))
            .sort_values("n", ascending=False).reset_index())
    feas["eligible"] = (feas["n"] >= MIN_WORD_N) & ((feas["valence_distinct"] > 1) | (feas["arousal_distinct"] > 1))
    feas.to_csv(OUT / "word_feasibility.csv", index=False)
    eligible = feas.loc[feas["eligible"], "spoken_word"].tolist()
    log.info("words eligible for within-word analysis (n >= %d): %s", MIN_WORD_N, eligible)

    # B. within-word associations
    rows = []
    for w in eligible:
        sub = data[data["spoken_word"] == w]
        for rating in ("valence", "arousal"):
            if sub[rating].nunique() < 2:
                continue
            for f in PRESPECIFIED:
                rho, p = spearman_perm(sub[f].to_numpy(), sub[rating].to_numpy())
                full = full_sp[(full_sp["rating"] == rating) & (full_sp["feature"] == f)].iloc[0]
                rows.append({"word": w, "n": int(len(sub)), "rating": rating, "feature": f, "rho_within_word": rho,
                             "p_perm": p, "rho_full_sample": float(full["spearman_rho"]),
                             "same_direction": bool(np.sign(rho) == np.sign(full["spearman_rho"]))})
    within = pd.DataFrame(rows)
    if not within.empty:
        within["q_bh"] = stats.false_discovery_control(within["p_perm"].to_numpy(), method="bh")
    within.to_csv(OUT / "within_word_correlations.csv", index=False)

    # figure: Kapoy recordings, each pre-specified feature vs each rating
    for w in eligible:
        sub = data[data["spoken_word"] == w].sort_values("participant_code")
        fig, axes = plt.subplots(2, len(PRESPECIFIED), figsize=(15, 5.4))
        rng = np.random.default_rng(1)
        for row_i, rating in enumerate(("valence", "arousal")):
            for col_i, f in enumerate(PRESPECIFIED):
                ax = axes[row_i, col_i]
                x = sub[rating].to_numpy() + rng.uniform(-0.08, 0.08, len(sub))
                ax.scatter(x, sub[f], s=40, color=viz.SERIES[0], edgecolor=viz.SURFACE, linewidth=1.2, zorder=3)
                ax.set_xticks(range(1, 6))
                ax.set_xlim(0.5, 5.5)
                r = within[(within["word"] == w) & (within["rating"] == rating) & (within["feature"] == f)]
                ttl = f"ρ = {r['rho_within_word'].iloc[0]:+.2f}, p = {r['p_perm'].iloc[0]:.2f}" if len(r) else "no variation"
                ax.set_title(f"{f}\n{ttl}", fontsize=8)
                ax.tick_params(labelsize=7)
                if col_i == 0:
                    ax.set_ylabel(f"vs {rating}", fontsize=8)
                ax.set_xlabel(f"self-reported {rating} (1–5)", fontsize=7)
        fig.suptitle(f"Lexically controlled view: only recordings of '{w}' (n = {len(sub)}); permutation p-values, "
                     "uncorrected", x=0.01, ha="left", fontsize=10, fontweight="semibold")
        fig.tight_layout()
        viz.save(fig, FIG / f"fig_lexical_{w.lower()}_features_vs_affect.png")

    # C. lexical-confound probe: Kapoy vs other words, acoustic features only
    probe = known.assign(is_kapoy=np.where(known["spoken_word"] == "Kapoy", "Kapoy", "other")).reset_index(drop=True)
    Xp = probe[feature_cols].to_numpy(dtype=float)
    yp = probe["is_kapoy"].to_numpy(dtype=object)
    gp = probe["participant_code"].to_numpy(dtype=object)
    probe_scores = [probe_ba(Xp, yp, gp, RANDOM_STATE + r) for r in range(N_REPEATS)]
    obs = probe_ba(Xp, yp, gp, RANDOM_STATE)
    null = np.array(Parallel(n_jobs=-1)(delayed(probe_ba)(Xp, np.random.default_rng(10_000 + i).permutation(yp), gp,
                                                         RANDOM_STATE) for i in range(N_PERM_PROBE)))
    probe_p = float((1 + np.sum(null >= obs - 1e-12)) / (1 + N_PERM_PROBE))
    probe_res = {"n": int(len(yp)), "n_kapoy": int((yp == "Kapoy").sum()), "n_other": int((yp == "other").sum()),
                 "balanced_accuracy_mean_10_repeats": float(np.mean(probe_scores)),
                 "balanced_accuracy_min": float(np.min(probe_scores)), "balanced_accuracy_max": float(np.max(probe_scores)),
                 "observed_repeat0": float(obs), "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                 "permutation_p": probe_p, "n_permutations": N_PERM_PROBE,
                 "model": "LogisticRegression(C=0.1, class_weight='balanced') after median imputation + standardisation"}
    log.info("lexical probe: BA %.3f (10 repeats), p = %.4f", probe_res["balanced_accuracy_mean_10_repeats"], probe_p)

    # D. word-affect association and model accuracy by word group
    assoc = {}
    for t in ("valence_2", "arousal_3"):
        tab = pd.crosstab(probe["is_kapoy"], probe[t])
        if tab.shape == (2, 2):
            p = float(stats.fisher_exact(tab.to_numpy())[1])
            test = "Fisher exact"
        else:
            obs_chi = stats.chi2_contingency(tab.to_numpy(), correction=False)[0]
            rng = np.random.default_rng(RANDOM_STATE)
            nulls = [stats.chi2_contingency(pd.crosstab(probe["is_kapoy"], rng.permutation(probe[t].to_numpy())).to_numpy(),
                                            correction=False)[0] for _ in range(20000)]
            p = float((1 + np.sum(np.array(nulls) >= obs_chi - 1e-12)) / 20001)
            test = "permutation chi-square (20,000)"
        chi = stats.chi2_contingency(tab.to_numpy(), correction=False)[0]
        cramers_v = float(np.sqrt(chi / (tab.to_numpy().sum() * (min(tab.shape) - 1))))
        assoc[t] = {"table": {r: tab.loc[r].to_dict() for r in tab.index}, "test": test, "p_value": p,
                    "cramers_v": cramers_v}
    oof = pd.read_csv(MODELS / "oof_predictions.csv", dtype={"y_true": str, "y_pred": str})
    word_of = data.set_index("participant_code")["spoken_word"]
    acc_rows = []
    for (task, model), d in oof[oof["model"].isin(["logistic_regression", "svm_rbf", "random_forest"])].groupby(["task", "model"]):
        d = d.assign(group=np.where(d["participant_code"].map(word_of) == "Kapoy", "Kapoy",
                                    np.where(d["participant_code"].map(word_of) == "", "word unknown", "other words")))
        for g, dd in d.groupby("group"):
            acc_rows.append({"task": task, "model": model, "word_group": g, "participants": dd["participant_code"].nunique(),
                             "accuracy_over_repeats": float((dd["y_true"] == dd["y_pred"]).mean())})
    acc_by_word = pd.DataFrame(acc_rows)
    acc_by_word.to_csv(OUT / "oof_accuracy_by_word_group.csv", index=False)

    summary = {"eligible_words": eligible, "probe": probe_res, "word_affect_association": assoc,
               "within_word_significant_q05": int((within["q_bh"] < 0.05).sum()) if not within.empty else 0,
               "within_word_tests": int(len(within)),
               "within_word_same_direction_share": float(within["same_direction"].mean()) if not within.empty else None}
    (OUT / "lexical_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    sig_within = summary["within_word_significant_q05"]
    lines = [
        "# Lexically Controlled Analysis (Phase 13, Protocol §XVII)",
        "",
        "*Generated by `src/08_lexical_control.py`. The spoken word is used only to **select** recordings; it is "
        "never a predictor. Tables: `reports/lexical/`.*",
        "",
        "## A. Which words allow a within-word comparison?",
        "",
        md_table(feas),
        "",
        f"A word is eligible when it has ≥ {MIN_WORD_N} recordings and variation in at least one rating. "
        f"**Eligible: {', '.join(eligible) if eligible else 'none'}.** No other word has more than "
        f"{int(feas.loc[~feas['spoken_word'].isin(eligible), 'n'].max())} recordings, so the protocol's example "
        "comparison (for example 'Grabe' + positive vs 'Grabe' + negative) is possible only for *Kapoy*, and only "
        "descriptively.",
        "",
        "## B. Same word, different self-reported affect",
        "",
        f"Spearman correlation between the six pre-specified prosodic features and the ratings, inside each "
        f"eligible word. Permutation p-values ({N_PERM_CORR:,} permutations), Benjamini–Hochberg across the "
        f"{len(within)} tests. **{sig_within} of {len(within)} pass q < 0.05.** In "
        f"{summary['within_word_same_direction_share']:.0%} of tests the within-word correlation has the same sign "
        "as in the full sample.",
        "",
        md_table(within),
        "",
        *[f"![{w}](../figures/lexical/fig_lexical_{w.lower()}_features_vs_affect.png)\n" for w in eligible],
        f"Within *Kapoy* (n = {int(feas.loc[feas['spoken_word'] == 'Kapoy', 'n'].iloc[0]) if 'Kapoy' in feas['spoken_word'].values else 0}): "
        + (lambda sub: f"{(sub['valence'] == sub['valence'].mode().iloc[0]).sum()} of {len(sub)} report valence "
           f"{int(sub['valence'].mode().iloc[0])} and {(sub['arousal'] == sub['arousal'].mode().iloc[0]).sum()} report "
           f"arousal {int(sub['arousal'].mode().iloc[0])}")(data[data['spoken_word'] == 'Kapoy'])
        + ". One or two participants at the extremes can therefore determine a correlation. These estimates are "
        "descriptive only.",
        "",
        "## C. Does the acoustic signal carry the word's identity? (lexical-confound probe)",
        "",
        f"Same leakage-safe pipeline (median imputation, standardisation, L2 logistic regression; participant-"
        f"grouped stratified 5-fold CV, {N_REPEATS} repeats) predicting *Kapoy* vs any other word from the acoustic "
        f"features alone (n = {probe_res['n']}: {probe_res['n_kapoy']} Kapoy, {probe_res['n_other']} other).",
        "",
        md_table(pd.DataFrame([{"balanced accuracy (mean of 10 repeats)": probe_res["balanced_accuracy_mean_10_repeats"],
                                "range over repeats": f"{probe_res['balanced_accuracy_min']:.3f}–{probe_res['balanced_accuracy_max']:.3f}",
                                "permutation null mean": probe_res["null_mean"], "null 95th percentile": probe_res["null_p95"],
                                "permutation p": probe_res["permutation_p"]}])),
        "",
        "## D. Is word choice itself associated with the ratings?",
        "",
        md_table(pd.DataFrame([{"target": t, "test": a["test"], "p_value": a["p_value"], "Cramér's V": a["cramers_v"],
                                "counts": "; ".join(f"{r}: {v}" for r, v in a["table"].items())} for t, a in assoc.items()])),
        "",
        "Out-of-fold accuracy of the affect models by word group (all repeats pooled):",
        "",
        md_table(acc_by_word),
        "",
        "## Interpretation rules applied",
        "",
        "- If the probe in C separates *Kapoy* from other words above chance, **and** word choice in D is "
        "associated with the ratings, an \"acoustic-only\" affect model can partly learn word identity instead of "
        "affective prosody. The protocol (§XV) warns of exactly this risk.",
        "- Only within-word comparisons (B) hold lexical content constant. With a single eligible word and n ≈ 10, "
        "they can show the direction of an effect at best, not establish one.",
        "",
    ]
    (REPORTS / "lexical_control_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("lexical analysis written (eligible words: %s)", eligible)
    return 0


if __name__ == "__main__":
    sys.exit(main())
