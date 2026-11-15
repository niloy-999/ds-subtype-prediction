"""
Stage 03 - leakage-free repeated nested cross-validation on the 304 T21 samples.

* outer: stratified (group-aware) 5-fold x R repeats; inner: stratified 5-fold
* residualiser fitted ONCE per outer fold (outer-training samples only), drop_first=True dummies
* panel size = selector__k INSIDE the GridSearchCV grid (no Python loop over panel sizes)
* one-SE parsimony refit (smallest panel within one SE) inside each GridSearchCV
* models: elastic net (primary), L2 logistic, SVM, random forest, XGBoost (if installed)
* stability table (selection frequency over outer folds) -> final panel = genes with frequency >= 70 %
  (final estimator is fitted directly on these pre-selected genes; no fresh SelectKBest on the full data)
* sensitivity: stability thresholds 60/70/80 %, leave-source-out validation
"""
from __future__ import annotations

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, auc, confusion_matrix, precision_recall_curve, roc_curve)
from sklearn.preprocessing import label_binarize

import config as C
import cv_engine as E
import data_access as DA
from ml_components import make_splits, get_model_specs, XGBOOST_AVAILABLE
from utils import (SUBTYPE_COLORS, get_logger, run_stage, save_fig, save_json, save_table)

log = get_logger()
CKPT = C.PROCESSED / "cv_checkpoints"


def run_all_folds(inp, models):
    CKPT.mkdir(exist_ok=True)
    splits = list(make_splits(inp["y"], inp["groups"], C.OUTER_SPLITS, C.OUTER_REPEATS, C.SEED))
    force = os.environ.get("DS_FORCE") == "1"
    key = f"{'-'.join(models)}_R{C.OUTER_REPEATS}_k{'-'.join(map(str, C.PANEL_SIZES))}_p{C.PREFILTER_TOP}"

    def job(r, f, tr, te):
        path = CKPT / f"{key}_r{r}f{f}.joblib"
        if path.exists() and not force:
            try:
                return joblib.load(path)
            except Exception:  # noqa: BLE001
                pass
        out = E.run_outer_fold(r, f, tr, te, inp, models)
        joblib.dump(out, path)
        return out

    log.info("nested CV: %d outer folds x %d models, grids include selector__k=%s, n_jobs=%d",
             len(splits), len(models), C.PANEL_SIZES, C.N_JOBS)
    results = Parallel(n_jobs=C.N_JOBS, verbose=5)(delayed(job)(r, f, tr, te) for r, f, tr, te in splits)
    cat = lambda k: pd.concat([r[k] for r in results if len(r[k])], ignore_index=True)
    return cat("perf"), cat("pred"), cat("sel"), cat("curve"), len(splits)


def plot_model_comparison(perf, summ):
    ok = perf.dropna(subset=["macro_f1"])
    order = summ.model.tolist()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, col, ttl in [(ax[0], "macro_f1", "Outer-fold macro-F1"), (ax[1], "roc_auc_ovr_macro", "Outer-fold ROC-AUC (OvR macro)")]:
        a.boxplot([ok.loc[ok.model == m, col] for m in order], labels=order, showmeans=True)
        a.set_title(ttl); a.tick_params(axis="x", rotation=30)
    ax[2].bar(order, summ.macro_f1_mean, yerr=summ.macro_f1_se, color="#4c72b0", capsize=3)
    ax[2].set_title("Mean macro-F1 +/- SE (over repeats)"); ax[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    save_fig("Fig5_model_comparison.png", fig)


def plot_panel_size_and_stability(curve, perf, stab, chosen):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
    for m, d in curve.groupby("model"):
        g = d.groupby("k")["mean_test_score"].agg(["mean", "std"]).reset_index()
        ax[0].errorbar(g.k, g["mean"], yerr=g["std"] / np.sqrt(d.fold_id.nunique()), marker="o", label=m, capsize=2)
    ax[0].set_xscale("log"); ax[0].set_xticks(C.PANEL_SIZES); ax[0].set_xticklabels(C.PANEL_SIZES)
    ax[0].set_xlabel("panel size (selector__k)"); ax[0].set_ylabel("inner-CV macro-F1 (best other params)")
    ax[0].set_title("Panel size vs performance (inner CV)"); ax[0].legend(fontsize=7)
    ks = perf.dropna(subset=["macro_f1"]).pivot_table(index="chosen_k", columns="model", values="fold_id", aggfunc="count").fillna(0)
    ks.plot(kind="bar", ax=ax[1], width=.8); ax[1].set_title("Panel size chosen by one-SE rule (outer folds)"); ax[1].set_xlabel("k")
    top = stab.head(30).iloc[::-1]
    ax[2].barh(top.label, top.selection_frequency, color=np.where(top.selection_frequency >= C.STABILITY_THRESHOLD, "#d62728", "#999999"))
    ax[2].axvline(C.STABILITY_THRESHOLD, ls="--", c="k"); ax[2].set_xlabel("selection frequency (outer folds)")
    ax[2].set_title(f"Stability selection ({chosen})"); ax[2].tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    save_fig("Fig6_panel_size_and_stability.png", fig)


def plot_final_performance(agg, chosen):
    pc = [f"prob_{c}" for c in C.CLASSES]
    yb = label_binarize(agg.true, classes=C.CLASSES)
    P = agg[pc].to_numpy()
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ConfusionMatrixDisplay(confusion_matrix(agg.true, agg.pred, labels=C.CLASSES), display_labels=C.CLASSES).plot(ax=ax[0, 0], colorbar=False)
    ax[0, 0].set_title("Aggregated outer-fold confusion matrix")
    rows = []
    for j, c in enumerate(C.CLASSES):
        fpr, tpr, _ = roc_curve(yb[:, j], P[:, j]); ax[0, 1].plot(fpr, tpr, color=SUBTYPE_COLORS[c], label=f"{c} (AUC {auc(fpr, tpr):.2f})")
        pr, rc, _ = precision_recall_curve(yb[:, j], P[:, j]); ax[0, 2].plot(rc, pr, color=SUBTYPE_COLORS[c], label=c)
        frac, mean_p = calibration_curve(yb[:, j], P[:, j], n_bins=6, strategy="quantile")
        ax[1, 0].plot(mean_p, frac, "o-", color=SUBTYPE_COLORS[c], label=c)
        bins = np.quantile(P[:, j], np.linspace(0, 1, 7)); idx = np.clip(np.digitize(P[:, j], bins[1:-1]), 0, 5)
        ece = np.sum([(idx == b).mean() * abs(yb[idx == b, j].mean() - P[idx == b, j].mean()) for b in range(6) if (idx == b).any()])
        rows.append({"class": c, "brier_ovr": float(np.mean((P[:, j] - yb[:, j]) ** 2)), "ece_ovr_6bin": float(ece)})
    ax[0, 1].plot([0, 1], [0, 1], "k:"); ax[0, 1].legend(); ax[0, 1].set_title("One-vs-rest ROC (out-of-fold)")
    ax[0, 2].legend(); ax[0, 2].set_title("One-vs-rest precision-recall"); ax[0, 2].set_xlabel("recall"); ax[0, 2].set_ylabel("precision")
    ax[1, 0].plot([0, 1], [0, 1], "k:"); ax[1, 0].legend(); ax[1, 0].set_title("Calibration (reliability)")
    ax[1, 0].set_xlabel("mean predicted probability"); ax[1, 0].set_ylabel("observed fraction")
    conf = P.max(axis=1); ok = (agg.pred == agg.true).to_numpy()
    ax[1, 1].hist([conf[ok], conf[~ok]], bins=15, label=["correct", "misclassified"], color=["#55a868", "#c44e52"], stacked=True)
    ax[1, 1].legend(); ax[1, 1].set_title("Prediction uncertainty (max class probability)")
    ax[1, 2].axis("off"); ax[1, 2].text(0, .6, f"Model: {chosen}\nPooled out-of-fold predictions: each participant's\nprobabilities are averaged over the outer repeats\n(an ensemble of independently trained models).\nThe single-model fold-level mean is in Table 5\n(macro_f1_mean).", fontsize=10)
    fig.tight_layout()
    save_fig("Fig7_final_model_performance.png", fig)
    save_table(pd.DataFrame(rows), "calibration_summary.csv", index=False)
    mis = agg[~ok].assign(max_prob=conf[~ok]).sort_values("max_prob", ascending=False)
    save_table(mis, "misclassified_samples.csv")


def leave_source_out(inp, model):
    src = inp["cov"]["source"].astype(str)
    big = src.value_counts()[lambda s: s >= 15].index.tolist()
    if len(big) < 2:
        log.info("leave-source-out skipped (<2 sources with >=15 T21 samples)")
        return
    rows = []
    for i, s in enumerate(big):
        te = np.where(src == s)[0]; tr = np.where(src != s)[0]
        if inp["y"].iloc[te].nunique() < 3:
            continue
        out = E.run_outer_fold(-1, i, tr, te, inp, [model], tag="LSO_")
        if len(out["perf"]) and "macro_f1" in out["perf"]:
            r = out["perf"].iloc[0].to_dict(); r["held_out_source"] = s; r["n_test"] = len(te); rows.append(r)
    if rows:
        save_table(pd.DataFrame(rows), "sensitivity_leave_source_out.csv", index=False)


def main() -> None:
    warnings.filterwarnings("ignore")
    inp = DA.load_ml_inputs()
    ann = DA.load_annotation()
    y = inp["y"]
    log.info("ML matrix: %d T21 samples x %d non-HSA21 genes | classes %s", *inp["X"].shape, y.value_counts().to_dict())
    assert (inp["manifest"].karyotype == "T21").all(), "controls must not enter the predictive matrix"
    assert len(inp["X"]) == y.notna().sum()

    models = list(get_model_specs(C.P["models"]).keys())
    if "xgboost" in C.P["models"] and not XGBOOST_AVAILABLE:
        log.warning("xgboost not installed -> XGBoost comparator skipped")
    perf, pred, sel, curve, n_folds = run_all_folds(inp, models)
    save_table(perf, "nested_cv_performance_by_fold.csv", index=False)
    save_table(pred, "nested_cv_oof_predictions.csv", index=False)
    save_table(sel, "nested_cv_selected_features.csv", index=False)
    if "error" in perf and perf["error"].notna().any():
        log.warning("some fits failed: %s", perf.dropna(subset=["error"])[["model", "error"]].drop_duplicates().to_dict("records")[:3])

    summ = E.summarize(perf)
    ci_rows = []
    for m in summ.model:
        ci_rows.append({"model": m, **E.bootstrap_ci(E.aggregate_oof(pred, m), C.P["n_boot"])})
    table5 = summ.merge(pd.DataFrame(ci_rows), on="model")
    ren = {}
    for k in ("macro_f1", "balanced_accuracy", "roc_auc_ovr_macro"):          # make the two estimators unambiguous
        for suf in ("", "_ci_low", "_ci_high"):
            ren[k + suf] = k + "_pooled_oof_repeat_avg" + suf
    table5 = table5.rename(columns=ren)
    save_table(table5, "Table5_outer_cv_performance_with_CI.csv", index=False)
    chosen_row = E.choose_model(summ)
    chosen = str(chosen_row.model)
    log.info("best mean macro-F1: %s (%.3f). ONE-SE-selected model: %s (median panel size %.0f, macro-F1 %.3f)",
             summ.iloc[0].model, summ.iloc[0].macro_f1_mean, chosen, chosen_row.chosen_k_median, chosen_row.macro_f1_mean)
    save_table(E.compare_models(perf, chosen), "model_comparison_tests.csv", index=False)

    # ------------------------------------------------ stability + final panel (rule: >=70 % frequency table)
    stab = E.stability_table(sel, chosen, n_folds)
    stab["gene_symbol"] = ann.reindex(stab.gene_id)["gene_symbol"].values
    stab["chromosome"] = ann.reindex(stab.gene_id)["chromosome"].values
    stab["label"] = np.where(stab.gene_symbol.fillna("") != "", stab.gene_symbol, stab.gene_id)
    save_table(stab, f"stability_selection_{chosen}.csv", index=False)
    sens = {f"n_genes_freq>={t:.2f}": int((stab.selection_frequency >= t).sum()) for t in [*C.STABILITY_SENSITIVITY, C.STABILITY_THRESHOLD]}
    log.info("stable genes by threshold: %s", sens)
    save_table(pd.DataFrame([sens]), "stability_threshold_sensitivity.csv", index=False)
    panel, panel_min, pinfo = E.derive_panels(stab, inp, C.STABILITY_THRESHOLD)
    sig60 = E.signature_at(stab, min(C.STABILITY_SENSITIVITY))
    log.info("final panel: %d genes (ALL genes with frequency >= %.0f%%%s); redundancy-pruned minimal panel: %d genes "
             "(%d pruned for |r|>%.2f); 60%% sensitivity signature: %d genes",
             len(panel), 100 * C.STABILITY_THRESHOLD, ", topped up to minimum size" if pinfo["topped_up"] else "",
             len(panel_min), len(pinfo["pruned_correlated"]), C.REDUNDANCY_R, len(sig60))

    def panel_table(genes, model):
        pt = pd.DataFrame({"gene_id": genes}).merge(
            stab[["gene_id", "gene_symbol", "chromosome", "selection_frequency", "mean_rank"]], on="gene_id")
        est = model.estimator
        if hasattr(est, "coef_") and est.coef_.shape[0] == len(model.classes_):
            for j, c in enumerate(model.classes_):
                pt[f"coef_{c}"] = est.coef_[j]
        elif hasattr(est, "feature_importances_"):
            pt["feature_importance"] = est.feature_importances_
        assert not (pt.chromosome.astype(str) == "21").any(), "HSA21 gene in panel"
        return pt

    fm = E.fit_final_model(inp, panel, chosen)
    joblib.dump(fm, C.MODELS / "final_subtype_model.joblib")
    save_table(panel_table(panel, fm), "Table6_final_gene_panel.csv", index=False)
    if len(panel_min) < len(panel):
        fm_min = E.fit_final_model(inp, panel_min, chosen)
        joblib.dump(fm_min, C.MODELS / "final_subtype_model_minimal.joblib")
        save_table(panel_table(panel_min, fm_min), "Table6b_minimal_nonredundant_panel.csv", index=False)
    save_json({"model": chosen, "params": fm.params, "panel": panel, "panel_minimal": panel_min, "signature_60": sig60,
               "classes": list(fm.classes_), "covariates": list(inp["cov"].columns),
               "stability_threshold": C.STABILITY_THRESHOLD, "panel_info": pinfo,
               "chosen_k_median": float(chosen_row.chosen_k_median), "outer_folds": n_folds, "mode": C.MODE,
               "seed": C.SEED,
               "performance_reporting": ("Primary single-model estimate = mean over the outer folds (macro_f1_mean, SE over repeats). "
                                "*_pooled_oof_repeat_avg = pooled out-of-fold predictions with probabilities averaged over the "
                                "outer repeats (ensemble of independently trained models; bootstrap 95% CI over participants); "
                                "expect it to be higher than the fold-level mean."),
               "note": ("Nested-CV performance describes the whole selection+fitting procedure (about "
                        f"{chosen_row.chosen_k_median:.0f} genes chosen inside each fold), NOT the final fixed panel; quote it "
                        "as such. The final panel is a stability-derived signature.")},
              C.MODELS / "final_model_metadata.json")

    # ------------------------------------------------ figures
    agg = E.aggregate_oof(pred, chosen)
    save_table(agg, "chosen_model_oof_predictions_aggregated.csv")
    plot_model_comparison(perf, summ)
    plot_panel_size_and_stability(curve, perf, stab, chosen)
    plot_final_performance(agg, chosen)
    leave_source_out(inp, chosen)


if __name__ == "__main__":
    run_stage("03 nested cross-validation", main)
