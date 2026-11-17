"""
Stage 07 - nested validation of the STABLE-PANEL PROCEDURE.

Stage 03 estimates the performance of a pipeline that re-selects ~k genes inside every fold; the final deployable panel
(genes with >=70 % selection frequency across outer folds) is then derived from ALL folds, so its own performance is not
estimated by stage 03. This stage closes that gap. For every outer fold (identical splits to stage 03), using ONLY the
outer-training samples:
    1. covariate residualisation + variance screen (as in stage 03),
    2. stability selection: B random 80 % subsamples; in each, the top-k genes by ANOVA F; keep genes selected in >= 70 %
       of subsamples (at least MIN_FINAL_PANEL, at most MAX_FINAL_PANEL),
    3. fit the classifier (inner-CV tuned) on the stable genes only,
    4. predict the untouched outer-test fold.
The result is an approximately unbiased estimate of the performance obtained by the full "stable panel -> classifier"
procedure that generated the final signature, plus the stability of the panel across folds.
"""
from __future__ import annotations

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C
import cv_engine as E
import data_access as DA
from ml_components import align_proba, evaluate_predictions, get_model_specs, make_splits, prepare_fold
from utils import get_logger, load_json, run_stage, save_fig, save_table

log = get_logger()
CKPT = C.PROCESSED / "stable_panel_checkpoints"
N_SUBSAMPLES = 50
MODELS = ["elastic_net", "svm"]          # elastic net = pre-specified primary model; SVM = best model under the one-SE rule


def one_fold(rep, fold, tr, te, inp, k, models):
    classes = np.array(C.CLASSES)
    Xtr, Xte, keep, _ = prepare_fold(inp["X"], inp["expressed"], inp["cov"], tr, te, C.PREFILTER_TOP, C.MIN_SAMPLE_FRACTION)
    ytr, yte = inp["y"].iloc[tr].to_numpy(), inp["y"].iloc[te].to_numpy()
    sss = StratifiedShuffleSplit(n_splits=N_SUBSAMPLES, train_size=0.8, random_state=C.SEED + 7919 * (rep + 1) + fold)
    counts = np.zeros(Xtr.shape[1])
    for a, _ in sss.split(Xtr, ytr):
        F, _ = f_classif(Xtr[a], ytr[a])
        F = np.nan_to_num(F)
        counts[np.argpartition(-F, k - 1)[:k]] += 1
    freq = counts / N_SUBSAMPLES
    sel = np.where(freq >= C.STABILITY_THRESHOLD)[0]
    topped_up = len(sel) < C.MIN_FINAL_PANEL
    if topped_up:
        sel = np.argsort(-freq)[:C.MIN_FINAL_PANEL]
    sel = sel[np.argsort(-freq[sel], kind="stable")][:C.MAX_FINAL_PANEL]
    genes = [keep[i] for i in sel]
    inner = E._inner_splits(inp["y"].iloc[tr], inp["groups"].iloc[tr], C.SEED + 1000 * (rep + 1) + fold)
    fid = f"r{rep}f{fold}"
    perf, pred = [], []
    for name in models:
        spec = get_model_specs([name])[name]
        grids = spec["grid"] if isinstance(spec["grid"], list) else [spec["grid"]]
        grids = [{kk: v for kk, v in g.items() if kk != "selector__k"} for g in grids]
        pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(spec["estimator"]))])
        gs = GridSearchCV(pipe, grids, scoring="f1_macro", cv=inner, n_jobs=1, refit=True, error_score=np.nan)
        gs.fit(Xtr[:, sel], ytr)
        p = align_proba(gs.predict_proba(Xte[:, sel]), gs.best_estimator_.named_steps["model"].classes_, classes)
        yhat = classes[p.argmax(1)]
        perf.append({"fold_id": fid, "repeat": rep, "fold": fold, "model": name, "panel_size": len(sel), "topped_up": topped_up,
                     "n_test": len(te), **evaluate_predictions(yte, yhat, p, classes)})
        for i, sid in enumerate(inp["y"].index[te]):
            pred.append({"fold_id": fid, "repeat": rep, "sample_id": sid, "model": name, "true": yte[i], "pred": yhat[i],
                         **{f"prob_{c}": p[i, j] for j, c in enumerate(classes)}})
    return {"perf": pd.DataFrame(perf), "pred": pd.DataFrame(pred), "genes": genes}


def main() -> None:
    warnings.filterwarnings("ignore")
    inp = DA.load_ml_inputs()
    ann = DA.load_annotation()
    meta = load_json(C.MODELS / "final_model_metadata.json")
    k = int(np.clip(round(meta.get("chosen_k_median", 50)), 5, 100))
    k = min(C.PANEL_SIZES, key=lambda x: abs(x - k))
    models = [m for m in MODELS if m in get_model_specs()]
    splits = list(make_splits(inp["y"], inp["groups"], C.OUTER_SPLITS, C.OUTER_REPEATS, C.SEED))
    CKPT.mkdir(exist_ok=True)
    force = os.environ.get("DS_FORCE") == "1"
    key = f"k{k}_B{N_SUBSAMPLES}_thr{C.STABILITY_THRESHOLD}_R{C.OUTER_REPEATS}_{'-'.join(models)}_p{C.PREFILTER_TOP}"

    def job(r, f, tr, te):
        path = CKPT / f"{key}_r{r}f{f}.joblib"
        if path.exists() and not force:
            try:
                return joblib.load(path)
            except Exception:  # noqa: BLE001
                pass
        out = one_fold(r, f, tr, te, inp, k, models)
        joblib.dump(out, path)
        return out

    log.info("stable-panel nested validation: %d outer folds, stability selection with %d subsamples per fold (top-%d genes, threshold %.0f%%), models %s",
             len(splits), N_SUBSAMPLES, k, 100 * C.STABILITY_THRESHOLD, models)
    res = Parallel(n_jobs=C.N_JOBS, verbose=5)(delayed(job)(r, f, tr, te) for r, f, tr, te in splits)
    perf = pd.concat([r["perf"] for r in res], ignore_index=True)
    pred = pd.concat([r["pred"] for r in res], ignore_index=True)
    save_table(perf, "Table12_stable_panel_nested_performance_by_fold.csv", index=False)

    rows = []
    for m, d in perf.groupby("model"):
        rep_means = d.groupby("repeat")["macro_f1"].mean()
        agg = E.aggregate_oof(pred, m)
        ci = E.bootstrap_ci(agg, C.P["n_boot"])
        rows.append({"model": m, "n_outer_folds": len(d), "macro_f1_mean": d.macro_f1.mean(), "macro_f1_sd": d.macro_f1.std(ddof=1),
                     "macro_f1_se_over_repeats": rep_means.std(ddof=1) / np.sqrt(len(rep_means)),
                     "balanced_accuracy_mean": d.balanced_accuracy.mean(), "roc_auc_ovr_mean": d.roc_auc_ovr_macro.mean(),
                     "mcc_mean": d.mcc.mean(), "panel_size_mean": d.panel_size.mean(), "panel_size_min": int(d.panel_size.min()),
                     "panel_size_max": int(d.panel_size.max()), "folds_topped_up_to_minimum": int(d.topped_up.sum()),
                     "macro_f1_pooled_oof_repeat_avg": ci["macro_f1"], "macro_f1_pooled_ci_low": ci["macro_f1_ci_low"],
                     "macro_f1_pooled_ci_high": ci["macro_f1_ci_high"], "roc_auc_pooled_oof_repeat_avg": ci["roc_auc_ovr_macro"],
                     "roc_auc_pooled_ci_low": ci["roc_auc_ovr_macro_ci_low"], "roc_auc_pooled_ci_high": ci["roc_auc_ovr_macro_ci_high"]})
    summ = pd.DataFrame(rows)

    # paired comparison with the stage-03 pipeline (same outer folds)
    s3 = pd.read_csv(C.TABLES / "nested_cv_performance_by_fold.csv")
    cmp_rows = []
    for m in models:
        a = perf[perf.model == m].set_index("fold_id")["macro_f1"]
        b = s3[s3.model == m].set_index("fold_id")["macro_f1"]
        j = pd.concat([a.rename("stable_panel"), b.rename("pipeline_k_tuned")], axis=1).dropna()
        d = j.stable_panel - j.pipeline_k_tuned
        cmp_rows.append({"model": m, "n_folds": len(j), "macro_f1_stable_panel": j.stable_panel.mean(), "macro_f1_pipeline_k_tuned": j.pipeline_k_tuned.mean(),
                         "mean_difference": d.mean(), "wilcoxon_p": stats.wilcoxon(d)[1] if np.any(d != 0) else 1.0})
    save_table(pd.DataFrame(cmp_rows), "Table12b_stable_panel_vs_pipeline_paired.csv", index=False)

    # panel stability across outer folds + overlap with the final signature
    sets = [set(r["genes"]) for r in res]
    jac = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    final = list(meta["panel"])
    freq = pd.Series({g: np.mean([g in s for s in sets]) for g in final})
    symbols = ann.reindex(final)["gene_symbol"].fillna("").astype(str).values
    save_table(pd.DataFrame({"gene_id": final, "gene_symbol": symbols, "fraction_of_outer_fold_panels_containing_gene": freq.values}),
               "Table12c_final_panel_genes_recurrence_in_nested_folds.csv", index=False)
    summ["mean_pairwise_jaccard_fold_panels"] = float(np.mean(jac))
    summ["mean_overlap_with_final_panel"] = float(np.mean([len(s & set(final)) / len(final) for s in sets]))
    save_table(summ, "Table12a_stable_panel_nested_summary.csv", index=False)
    log.info("stable-panel nested performance:\n%s", summ[["model", "macro_f1_mean", "macro_f1_se_over_repeats", "roc_auc_ovr_mean", "panel_size_mean",
                                                          "macro_f1_pooled_oof_repeat_avg", "mean_pairwise_jaccard_fold_panels"]].round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    pos, labels, data = 0, [], []
    for m in models:
        a = perf[perf.model == m].macro_f1
        b = s3[s3.model == m].macro_f1
        data += [b.to_numpy(), a.to_numpy()]
        labels += [f"{m}\npipeline (k tuned)", f"{m}\nstable panel"]
    ax[0].boxplot(data, labels=labels, showmeans=True)
    ax[0].set_ylabel("outer-fold macro-F1"); ax[0].set_title("Stable-panel procedure vs stage-03 pipeline"); ax[0].tick_params(axis="x", labelsize=7)
    ax[1].hist(perf[perf.model == models[0]].panel_size, bins=range(int(perf.panel_size.min()), int(perf.panel_size.max()) + 2), color="#4c72b0")
    ax[1].set_xlabel("stable panel size in outer-training fold"); ax[1].set_title("Panel size across 50 outer folds")
    ax[2].barh(symbols[::-1], freq.values[::-1], color="#55a868"); ax[2].set_xlabel("fraction of outer-fold panels containing the gene")
    ax[2].set_title("Recurrence of the 16 final genes"); ax[2].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    save_fig("Fig13_stable_panel_nested_validation.png", fig)


if __name__ == "__main__":
    run_stage("07 nested validation of the stable-panel procedure", main)
