"""
05_eda.py -- PHASE 7: exploratory data analysis (before any model is trained).

Inputs   data/metadata_cleaned.csv, data/analysis_manifest.csv, data/acoustic_features.csv,
         reports/audio_quality_table.csv
Outputs  figures/eda/*.png, reports/eda/*.csv, reports/eda/eda_summary.json,
         reports/exploratory_analysis.md (generated; numbers come from the data)

Analyses
    RQ1  distribution of self-reported valence/arousal (all metadata rows and the analysis set)
    RQ2  context variables -- reported as NOT analysable when constant (no variation in Group 3)
    RQ3  Spearman correlation of every acoustic feature with the raw 1-5 ratings, Benjamini-Hochberg
         FDR per rating; six interpretable features fixed in advance are plotted by derived class
    Data quality  recording-configuration effect on features (Mann-Whitney, BH-FDR) and its
                  association with the labels; utterance-to-background ratio vs ratings
All p-values are exploratory. Associations are not causal.

Usage: .venv/Scripts/python.exe src/05_eda.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

import viz
from viz import plt
from common import (CLEAN_METADATA_CSV, DATA, FEATURES_CSV, FIGURES, MIN_CLASS_SIZE, PROTOCOL_WORDS, RANDOM_STATE,
                    REPORTS, TARGETS, derive_target, rel, setup_logging)

FIG = FIGURES / "eda"
OUT = REPORTS / "eda"
MANIFEST = DATA / "analysis_manifest.csv"
QA_TABLE = REPORTS / "audio_quality_table.csv"
ID_COLS = ("participant_code", "segment_file", "segment_sha256")
KEY_FEATURES = ["utterance_duration_s", "voiced_ratio", "f0_median", "f0_std_st", "f0_range_st", "f0_slope_st_per_s",
                "rms_db_mean", "rms_db_std", "zcr_mean", "spectral_centroid_hz_mean", "spectral_rolloff_hz_mean",
                "mfcc_1_mean"]
# fixed BEFORE looking at any feature-affect statistic: classic prosodic correlates of affect
PRESPECIFIED = ["f0_median", "f0_range_st", "rms_db_mean", "utterance_duration_s", "spectral_centroid_hz_mean",
                "voiced_ratio"]
LABELS = {"utterance_duration_s": "utterance duration (s)", "voiced_ratio": "voiced ratio", "f0_median": "F0 median (Hz)",
          "f0_std_st": "F0 std (semitones)", "f0_range_st": "F0 range p5–p95 (semitones)",
          "f0_slope_st_per_s": "F0 slope (st/s)", "rms_db_mean": "RMS level mean (dBFS)", "rms_db_std": "RMS level std (dB)",
          "zcr_mean": "zero-crossing rate mean", "spectral_centroid_hz_mean": "spectral centroid mean (Hz)",
          "spectral_rolloff_hz_mean": "spectral roll-off mean (Hz)", "mfcc_1_mean": "MFCC 1 (c0) mean"}

log = setup_logging("eda", REPORTS / "logs" / "05_eda.log")


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
            return "—"
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False))


def perm_chi2(a, b, n_perm=20000, seed=RANDOM_STATE) -> float:
    """Permutation p-value for association of two categorical variables (chi-square statistic)."""
    a, b = np.asarray(a), np.asarray(b)
    def chi2(x, y):
        ct = pd.crosstab(x, y).to_numpy()
        return stats.chi2_contingency(ct, correction=False)[0] if min(ct.shape) > 1 else 0.0
    obs = chi2(a, b)
    rng = np.random.default_rng(seed)
    null = np.array([chi2(a, rng.permutation(b)) for _ in range(n_perm)])
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + n_perm))


def strip_by_class(ax, values, classes, order, title, ylabel):
    rng = np.random.default_rng(0)
    for i, c in enumerate(order):
        v = values[classes == c]
        x = i + rng.uniform(-0.12, 0.12, size=len(v))
        ax.scatter(x, v, s=34, color=viz.SERIES[i], edgecolor=viz.SURFACE, linewidth=1.2, zorder=3,
                   label=f"{c} (n = {len(v)})")
        if len(v):
            ax.hlines(np.median(v), i - 0.28, i + 0.28, color=viz.INK, lw=1.6, zorder=4)
    ax.set_xticks(range(len(order)), order)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="x", visible=False)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(CLEAN_METADATA_CSV, keep_default_na=False, na_values=[""], encoding="utf-8-sig",
                       dtype={"spoken_word": str, "self_reported_feeling": str})
    meta["spoken_word"] = meta["spoken_word"].fillna("")
    meta["self_reported_feeling"] = meta["self_reported_feeling"].fillna("")
    manifest = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    feats = pd.read_csv(FEATURES_CSV)
    qa = pd.read_csv(QA_TABLE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    feature_cols = [c for c in feats.columns if c not in ID_COLS]
    data = feats.merge(meta, on="participant_code", how="left", validate="one_to_one").copy()
    sel_qa = qa[qa["selected"] == "True"].set_index("participant_code")
    data["config"] = np.where(data["participant_code"].map(
        lambda c: "RECORDING_CONFIG_DIFFERS" in sel_qa.loc[c, "flags"]), "B: stereo + 2nd track", "A: mono")
    data["sbr_db"] = data["participant_code"].map(sel_qa["sbr_db"].astype(float))
    for t in TARGETS:
        data[t] = derive_target(data[TARGETS[t]["source"]], t)
    n = len(data)
    summary: dict = {"n_metadata": int(len(meta)), "n_analysis": int(n), "n_features": len(feature_cols)}

    # ---- RQ1: affect distributions --------------------------------------------------
    dist_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, key in zip(axes, ("valence", "arousal")):
        m_counts = meta[key].dropna().astype(int).value_counts().reindex(range(1, 6), fill_value=0)
        a_counts = data[key].dropna().astype(int).value_counts().reindex(range(1, 6), fill_value=0)
        x = np.arange(1, 6)
        b1 = ax.bar(x - 0.19, m_counts.values, width=0.36, color=viz.SERIES[0],
                    label=f"all metadata rows (n = {int(m_counts.sum())})")
        b2 = ax.bar(x + 0.19, a_counts.values, width=0.36, color=viz.SERIES[1],
                    label=f"analysis set with recordings (n = {int(a_counts.sum())})")
        viz.bar_labels(ax, b1)
        viz.bar_labels(ax, b2)
        tick_labels = {"valence": ["1 Very\nunpleasant", "2", "3 Neutral", "4", "5 Very\npleasant"],
                       "arousal": ["1 Very\nlow", "2", "3 Moderate", "4", "5 Very\nhigh"]}[key]
        ax.set_xticks(x, tick_labels, fontsize=7.5)
        ax.set_title(f"Self-reported {key}")
        ax.grid(axis="x", visible=False)
        for r in range(1, 6):
            dist_rows.append({"rating": key, "value": r, "metadata_n": int(m_counts[r]),
                              "metadata_pct": 100 * m_counts[r] / m_counts.sum(), "analysis_n": int(a_counts[r]),
                              "analysis_pct": 100 * a_counts[r] / a_counts.sum()})
    axes[0].set_ylabel("participants")
    # one shared legend above both panels, clear of the bars and their value labels
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               frameon=False)
    viz.save(fig, FIG / "fig01_affect_distributions.png")
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(OUT / "affect_distributions.csv", index=False)

    # valence x arousal
    ct = pd.crosstab(data["valence"].astype(int), data["arousal"].astype(int)).reindex(
        index=range(1, 6), columns=range(1, 6), fill_value=0)
    rho_va, p_va = stats.spearmanr(data["valence"], data["arousal"])
    mv = meta.dropna(subset=["valence", "arousal"])
    rho_va_m, p_va_m = stats.spearmanr(mv["valence"], mv["arousal"])
    summary["valence_arousal_spearman"] = {"analysis": [float(rho_va), float(p_va), int(n)],
                                           "metadata": [float(rho_va_m), float(p_va_m), int(len(mv))]}
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(ct.values, cmap=viz.SEQUENTIAL, vmin=0, vmax=max(1, ct.values.max()), origin="lower")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, int(ct.values[i, j]), ha="center", va="center", fontsize=9,
                    color=viz.ink_for(im.cmap(im.norm(ct.values[i, j]))))
    ax.set_xticks(range(5), range(1, 6))
    ax.set_yticks(range(5), range(1, 6))
    ax.set_xlabel("arousal (1 very low – 5 very high)")
    ax.set_ylabel("valence (1 very unpleasant – 5 very pleasant)")
    ax.set_title(f"Valence × arousal, analysis set (n = {n})")
    ax.grid(False)
    viz.save(fig, FIG / "fig02_valence_arousal_heatmap.png")

    # derived targets + viability
    target_rows = []
    tasks = ["valence_3", "valence_2", "arousal_3", "arousal_2"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(12, 3.0), sharey=True)
    for ax, t in zip(axes, tasks):
        order = TARGETS[t]["order"]
        counts = pd.Series(data[t]).value_counts().reindex(order, fill_value=0)
        bars = ax.bar(range(len(order)), counts.values, width=0.6, color=viz.SERIES[0])
        viz.bar_labels(ax, bars)
        ax.axhline(MIN_CLASS_SIZE, color=viz.INK_2, lw=1, label=f"minimum {MIN_CLASS_SIZE} per class")
        ax.set_xticks(range(len(order)), order, fontsize=7.5)
        ax.set_title(t, fontsize=9)
        ax.grid(axis="x", visible=False)
    for t in TARGETS:
        counts = pd.Series(data[t]).value_counts().reindex(TARGETS[t]["order"], fill_value=0)
        target_rows.append({"task": t, "classes": " / ".join(f"{k}: {v}" for k, v in counts.items()),
                            "min_class": int(counts.min()), "viable": bool(counts.min() >= MIN_CLASS_SIZE)})
    axes[0].set_ylabel("participants")
    axes[-1].legend(loc="upper right")
    viz.save(fig, FIG / "fig03_derived_target_classes.png")
    targets_df = pd.DataFrame(target_rows)
    targets_df.to_csv(OUT / "derived_target_viability.csv", index=False)

    # ---- context variables (RQ2) -------------------------------------------------------
    ctx = []
    for col, label in (("year_level", "Year Level"), ("class_activity_raw", "Class Activity"),
                       ("session", "Session"), ("subject", "Subject")):
        vc = meta[col].fillna("(blank)").astype(str).replace("", "(blank)").value_counts()
        ctx.append({"variable": label, "values": "; ".join(f"{k}: {v}" for k, v in vc.items()),
                    "distinct_non_blank": int((vc.index != "(blank)").sum())})
    ctx_df = pd.DataFrame(ctx)

    # ---- words and feelings ----------------------------------------------------------------
    words = meta.loc[meta["spoken_word"] != "", "spoken_word"].value_counts()
    on_list = {w.casefold() for w in PROTOCOL_WORDS}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    order = words.sort_values().index
    colors = [viz.SERIES[0] if w.casefold() in on_list else viz.SERIES[1] for w in order]
    bars = ax.barh(order, words[order].values, color=colors, height=0.6)
    viz.bar_labels(ax, bars, horizontal=True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=viz.SERIES[0], label="on the protocol's 7-word list"),
                       Patch(color=viz.SERIES[1], label="not on the list")], loc="lower right")
    ax.set_title(f"Spoken word (metadata rows with a word, n = {int(words.sum())}; missing: {int((meta['spoken_word'] == '').sum())})")
    ax.set_xlabel("participants")
    ax.grid(axis="y", visible=False)
    viz.save(fig, FIG / "fig04_spoken_word_frequency.png")
    feelings = meta.loc[meta["self_reported_feeling"] != "", "self_reported_feeling"].value_counts()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    order = feelings.sort_values().index
    bars = ax.barh(order, feelings[order].values, color=viz.SERIES[0], height=0.6)
    viz.bar_labels(ax, bars, horizontal=True)
    ax.set_title(f"Self-reported feeling text (n = {int(feelings.sum())}; missing: {int((meta['self_reported_feeling'] == '').sum())})")
    ax.set_xlabel("participants")
    ax.grid(axis="y", visible=False)
    viz.save(fig, FIG / "fig05_self_reported_feeling_frequency.png")
    word_affect = (meta[meta["spoken_word"] != ""].assign(
        valence_2=lambda d: derive_target(d["valence"], "valence_2"),
        arousal_3=lambda d: derive_target(d["arousal"], "arousal_3"))
        .groupby("spoken_word").agg(n=("participant_code", "size"),
                                    valence_values=("valence", lambda s: ", ".join(str(int(v)) for v in sorted(s.dropna()))),
                                    arousal_values=("arousal", lambda s: ", ".join(str(int(v)) for v in sorted(s.dropna()))))
        .sort_values("n", ascending=False).reset_index())
    word_affect["on_protocol_list"] = word_affect["spoken_word"].str.casefold().isin(on_list)
    word_affect.to_csv(OUT / "spoken_word_affect.csv", index=False)

    # ---- acoustic feature distributions and correlations ----------------------------------
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.2))
    for ax, f in zip(axes.flat, KEY_FEATURES):
        ax.hist(data[f], bins=10, color=viz.SERIES[0], edgecolor=viz.SURFACE, linewidth=1.5)
        ax.set_title(LABELS[f], fontsize=8.5)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Distributions of selected acoustic features (n = {n} located utterances)", x=0.01, ha="left",
                 fontsize=10, fontweight="semibold")
    fig.tight_layout()
    viz.save(fig, FIG / "fig06_feature_distributions.png")
    prosodic = [c for c in feature_cols if not c.startswith(("mfcc", "delta"))]
    corr = data[prosodic].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    im = ax.imshow(corr.values, cmap=viz.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(prosodic)), prosodic, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(prosodic)), prosodic, fontsize=6.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Spearman ρ")
    ax.set_title(f"Correlation among duration, F0, energy, ZCR and spectral features (n = {n})")
    viz.save(fig, FIG / "fig07_feature_correlation_matrix.png")
    high_corr = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(prosodic) for b in prosodic[i + 1:]
                 if abs(corr.loc[a, b]) >= 0.9]

    # ---- RQ3: features vs ratings ---------------------------------------------------------
    rows = []
    for key in ("valence", "arousal"):
        for f in feature_cols:
            rho, p = stats.spearmanr(data[f], data[key])
            rows.append({"rating": key, "feature": f, "spearman_rho": float(rho), "p_value": float(p)})
    sp = pd.DataFrame(rows)
    sp["q_value_bh"] = np.nan
    for key in ("valence", "arousal"):
        idx = sp["rating"] == key
        sp.loc[idx, "q_value_bh"] = stats.false_discovery_control(sp.loc[idx, "p_value"].to_numpy(), method="bh")
    sp.to_csv(OUT / "spearman_feature_vs_rating.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, key in zip(axes, ("valence", "arousal")):
        d = sp[sp["rating"] == key]
        sig = d["q_value_bh"] < 0.05
        ax.scatter(d.loc[~sig, "spearman_rho"], -np.log10(d.loc[~sig, "p_value"]), s=26, color=viz.SERIES[0],
                   edgecolor=viz.SURFACE, linewidth=1, label="q ≥ 0.05 (BH)")
        if sig.any():
            ax.scatter(d.loc[sig, "spearman_rho"], -np.log10(d.loc[sig, "p_value"]), s=34, color=viz.SERIES[1],
                       edgecolor=viz.SURFACE, linewidth=1, label="q < 0.05 (BH)")
            for _, r in d[sig].iterrows():
                ax.annotate(r["feature"], (r["spearman_rho"], -np.log10(r["p_value"])), fontsize=6.5,
                            color=viz.INK_2, xytext=(3, 3), textcoords="offset points")
        top = d.reindex(d["p_value"].sort_values().index[:3])
        for rank, (_, r) in enumerate(top.iterrows()):
            if not (r["q_value_bh"] < 0.05):
                ax.annotate(r["feature"], (r["spearman_rho"], -np.log10(r["p_value"])), fontsize=6.5,
                            color=viz.INK_2, xytext=(4, 6 - 9 * rank), textcoords="offset points",
                            ha="left" if r["spearman_rho"] < 0.55 else "right")
        ax.axhline(-np.log10(0.05), color=viz.INK_2, lw=1, label="uncorrected p = 0.05")
        ax.set_xlim(-0.75, 0.75)
        ax.set_xlabel(f"Spearman ρ with {key} (1–5)")
        ax.set_title(f"{key}: {int(sig.sum())} of {len(d)} features pass FDR q < 0.05")
        ax.legend(loc="upper left", fontsize=7)
    axes[0].set_ylabel("−log10 p")
    viz.save(fig, FIG / "fig08_feature_rating_associations.png")

    for t, fname in (("arousal_3", "fig09_prespecified_features_by_arousal.png"),
                     ("valence_2", "fig10_prespecified_features_by_valence.png")):
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
        classes = np.asarray(data[t], dtype=object)
        for ax, f in zip(axes.flat, PRESPECIFIED):
            strip_by_class(ax, data[f].to_numpy(), classes, TARGETS[t]["order"], LABELS[f], "")
        axes.flat[0].legend(loc="best", fontsize=7)
        fig.suptitle(f"Pre-specified prosodic features by {t} (derived from the participants' ratings); "
                     "black line = median", x=0.01, ha="left", fontsize=10, fontweight="semibold")
        fig.tight_layout()
        viz.save(fig, FIG / fname)

    # ---- recording-configuration confound ---------------------------------------------------
    is_b = data["config"].str.startswith("B")
    cap = pd.to_datetime(sel_qa.loc[data.loc[is_b, "participant_code"], "capture_time"], format="%Y-%m-%dT%H:%M:%S%z")
    med_cap = pd.to_datetime(sel_qa["capture_time"], format="%Y-%m-%dT%H:%M:%S%z").sort_values().iloc[len(sel_qa) // 2]
    late_h = ((cap - med_cap).dt.total_seconds() / 3600)
    late = late_h[late_h > 1]
    mw = []
    for f in feature_cols:
        u, p = stats.mannwhitneyu(data.loc[is_b, f], data.loc[~is_b, f], alternative="two-sided")
        rbc = 2 * u / (is_b.sum() * (~is_b).sum()) - 1  # rank-biserial correlation (B vs A)
        mw.append({"feature": f, "median_A": float(data.loc[~is_b, f].median()), "median_B": float(data.loc[is_b, f].median()),
                   "rank_biserial_B_vs_A": float(rbc), "p_value": float(p)})
    mw = pd.DataFrame(mw)
    mw["q_value_bh"] = stats.false_discovery_control(mw["p_value"].to_numpy(), method="bh")
    mw = mw.sort_values("p_value")
    mw.to_csv(OUT / "recording_configuration_effect.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    cfg_order = ["A: mono", "B: stereo + 2nd track"]
    for ax, f in zip(axes, ["zcr_mean", "spectral_centroid_hz_mean", "mfcc_1_mean", "rms_db_mean"]):
        strip_by_class(ax, data[f].to_numpy(), data["config"].to_numpy(), cfg_order, LABELS[f], "")
        ax.set_xticks(range(2), ["A: mono", "B: stereo\n+ 2nd track"], fontsize=7.5)
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle("Recording configuration changes the acoustics (not affect): config B = 6 files with a different "
                 "audio setup", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    viz.save(fig, FIG / "fig11_recording_configuration_effect.png")
    assoc = {}
    for t in ("valence_2", "arousal_2", "arousal_3"):
        tab = pd.crosstab(data["config"], data[t])
        if tab.shape == (2, 2):
            _, p = stats.fisher_exact(tab.to_numpy())
            test = "Fisher exact"
        else:
            p = perm_chi2(data["config"], data[t])
            test = "permutation chi-square (20,000)"
        assoc[t] = {"table": {r: tab.loc[r].to_dict() for r in tab.index}, "test": test, "p_value": float(p)}
    sbr = {key: [float(v) for v in stats.spearmanr(data["sbr_db"], data[key])] for key in ("valence", "arousal")}

    # ---- summary + report ---------------------------------------------------------------------
    sig_counts = {key: int((sp[(sp["rating"] == key)]["q_value_bh"] < 0.05).sum()) for key in ("valence", "arousal")}
    summary.update({
        "significant_features_fdr05": sig_counts,
        "top_features": {key: sp[sp["rating"] == key].sort_values("p_value").head(10)[
            ["feature", "spearman_rho", "p_value", "q_value_bh"]].to_dict("records") for key in ("valence", "arousal")},
        "config_effect_features_fdr05": int((mw["q_value_bh"] < 0.05).sum()),
        "config_label_association": assoc, "sbr_vs_rating_spearman": sbr,
        "derived_targets": target_rows, "context": ctx, "n_prosodic_pairs_abs_rho_ge_0_9": len(high_corr),
    })
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    top = lambda key: sp[sp["rating"] == key].sort_values("p_value").head(10)[  # noqa: E731
        ["feature", "spearman_rho", "p_value", "q_value_bh"]]
    va = summary["valence_arousal_spearman"]
    lines = [
        "# Exploratory Data Analysis (Phase 7)",
        "",
        f"*Generated by `src/05_eda.py`. Analysis set: the {n} participants with a usable recording "
        f"(of {len(meta)} metadata rows). Tables: `reports/eda/`. Figures: `figures/eda/`. All p-values are "
        "exploratory, and associations do not imply causation.*",
        "",
        "## 1. Self-reported affect (RQ1)",
        "",
        "![affect](../figures/eda/fig01_affect_distributions.png)",
        "",
        md_table(dist.assign(metadata_pct=dist["metadata_pct"].round(1), analysis_pct=dist["analysis_pct"].round(1)),
                 "{:.1f}"),
        "",
        f"Metadata arousal counts exclude {', '.join(meta.loc[meta['arousal_status'] == 'STRUCK_THROUGH', 'participant_code'])}, "
        f"whose struck-through rating is not used. In the analysis set, "
        f"valence and arousal are positively associated: Spearman ρ = {va['analysis'][0]:.2f}, p = "
        f"{va['analysis'][1]:.3g}, n = {va['analysis'][2]}. Across all metadata rows with both ratings, "
        f"ρ = {va['metadata'][0]:.2f}, p = {va['metadata'][1]:.3g}, n = {va['metadata'][2]}.",
        "",
        "![valence x arousal](../figures/eda/fig02_valence_arousal_heatmap.png)",
        "",
        "## 2. Classification targets (derived) and their viability",
        "",
        "The original 1–5 ratings are kept. Derived groupings are *not* collected variables. A task is modelled "
        f"only if every class has at least {MIN_CLASS_SIZE} participants, so each test fold of a 5-fold "
        "cross-validation can contain every class.",
        "",
        md_table(targets_df),
        "",
        "![classes](../figures/eda/fig03_derived_target_classes.png)",
        "",
        "## 3. Educational context (RQ2): not analysable within Group 3",
        "",
        md_table(ctx_df),
        "",
        "Year level, class activity and session each take **one value** for every Group 3 participant, and subject "
        "is one value or blank. Their association with affect (Protocol §XIII: Year Level × Affect, Class Activity "
        "× Affect, Session × Affect) **cannot be estimated from Group 3 data**. It needs the consolidated class "
        "dataset, where these variables vary.",
        "",
        "## 4. Spoken words and feeling descriptions",
        "",
        "![words](../figures/eda/fig04_spoken_word_frequency.png)",
        "",
        md_table(word_affect),
        "",
        f"Words with ≥ 8 occurrences (needed for a within-word comparison): "
        f"{', '.join(f'{w} ({c})' for w, c in words.items() if c >= 8) or 'none'}; every other word occurs at most "
        f"{int(words[words < 8].max())} times (see `reports/lexical_control_analysis.md`).",
        "",
        "![feelings](../figures/eda/fig05_self_reported_feeling_frequency.png)",
        "",
        "## 5. Acoustic features",
        "",
        "![distributions](../figures/eda/fig06_feature_distributions.png)",
        "",
        f"![correlations](../figures/eda/fig07_feature_correlation_matrix.png)",
        "",
        f"{len(high_corr)} pairs of duration/F0/energy/ZCR/spectral features have |ρ| ≥ 0.9; the feature set is "
        "partly redundant. That is one reason models use regularisation.",
        "",
        "## 6. Acoustic features vs self-reported affect (RQ3)",
        "",
        f"Spearman correlation of each of the {len(feature_cols)} features with the raw ratings (n = {n}), "
        f"Benjamini–Hochberg FDR within each rating. **Features passing q < 0.05: valence {sig_counts['valence']}, "
        f"arousal {sig_counts['arousal']}.**",
        "",
        "![associations](../figures/eda/fig08_feature_rating_associations.png)",
        "",
        "Ten smallest p-values, valence:",
        "",
        md_table(top("valence")),
        "",
        "Ten smallest p-values, arousal:",
        "",
        md_table(top("arousal")),
        "",
        "Six features fixed in advance (pitch level and range, loudness, duration, spectral balance, voicing), by "
        "derived class:",
        "",
        "![arousal](../figures/eda/fig09_prespecified_features_by_arousal.png)",
        "",
        "![valence](../figures/eda/fig10_prespecified_features_by_valence.png)",
        "",
        "F0 in Hz mostly reflects voice type (for example sex), and no speaker baseline exists because each "
        "participant gives one utterance. Semitone measures (range, std, slope) are scale-free.",
        "",
        "## 7. Recording condition as a confound",
        "",
        f"{int(is_b.sum())} selected recordings (config B) were made with a different audio setup (stereo plus a second track; "
        f"{len(late)} captured {late.min():.1f}–{late.max():.1f} h after the median capture time). **{summary['config_effect_features_fdr05']} of "
        f"{len(feature_cols)} features differ between configurations at FDR q < 0.05** (Mann–Whitney):",
        "",
        md_table(mw.head(10)),
        "",
        "![configuration](../figures/eda/fig11_recording_configuration_effect.png)",
        "",
        "Association of configuration with the labels:",
        "",
        md_table(pd.DataFrame([{"target": t, "test": a["test"], "p_value": a["p_value"],
                                "counts": "; ".join(f"{r}: {v}" for r, v in a["table"].items())}
                               for t, a in assoc.items()])),
        "",
        f"Utterance-to-background ratio vs ratings (Spearman): valence ρ = {sbr['valence'][0]:.2f} "
        f"(p = {sbr['valence'][1]:.3g}); arousal ρ = {sbr['arousal'][0]:.2f} (p = {sbr['arousal'][1]:.3g}).",
        "",
        "Any acoustic feature that separates the configurations can produce a spurious affect association "
        "whenever configuration and labels are unevenly distributed. The model evaluation therefore includes a "
        "sensitivity analysis without config-B recordings.",
        "",
    ]
    (REPORTS / "exploratory_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("EDA done: n=%d; FDR-significant features valence=%d arousal=%d; config-affected features=%d",
             n, sig_counts["valence"], sig_counts["arousal"], summary["config_effect_features_fdr05"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
