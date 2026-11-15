"""
Nested cross-validation engine (importable so joblib workers can unpickle it).

Per OUTER fold (fitted on the outer-training samples only):
    prepare_fold()  -> expression filter, variance screen, covariate residualiser  (fitted ONCE per outer fold)
    for each model: ONE GridSearchCV whose grid contains  selector__k in {5,10,20,50,100}  plus the model's own
                    hyper-parameters; the refit callable applies the one-SE parsimony rule (smallest panel
                    within one SE of the best inner-CV macro-F1).
The outer-test fold is touched exactly once, for prediction.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from ml_components import (CovariateResidualizer, PanelModel, align_proba, build_pipeline, evaluate_predictions,
                           get_model_specs, make_one_se_refit, make_splits, prepare_fold, selected_features)
from utils import bh_fdr


def _inner_splits(y: pd.Series, groups: pd.Series | None, seed: int):
    yy = y.reset_index(drop=True)
    gg = groups.reset_index(drop=True) if groups is not None else None
    return [(a, b) for _, _, a, b in make_splits(yy, gg, C.INNER_SPLITS, 1, seed)]


def run_outer_fold(rep: int, fold: int, tr: np.ndarray, te: np.ndarray, inp: Dict[str, Any],
                   model_names: Sequence[str], tag: str = "") -> Dict[str, pd.DataFrame]:
    X, expressed, cov, y, groups = inp["X"], inp["expressed"], inp["cov"], inp["y"], inp["groups"]
    classes = np.array(C.CLASSES)
    Xtr, Xte, keep, _ = prepare_fold(X, expressed, cov, tr, te, C.PREFILTER_TOP, C.MIN_SAMPLE_FRACTION)
    ytr, yte = y.iloc[tr], y.iloc[te]
    inner = _inner_splits(ytr, groups.iloc[tr] if groups is not None else None, C.SEED + 1000 * (rep + 1) + fold)
    specs = get_model_specs(model_names)
    perf, pred, sel, curve = [], [], [], []
    fid = f"{tag}r{rep}f{fold}"
    for name, spec in specs.items():
        try:
            gs = GridSearchCV(build_pipeline(spec["estimator"]), spec["grid"], scoring="f1_macro", cv=inner,
                              refit=make_one_se_refit(), n_jobs=1, error_score=np.nan)
            gs.fit(Xtr, ytr.to_numpy())
            pipe: Pipeline = gs.best_estimator_
            p = align_proba(pipe.predict_proba(Xte), pipe.named_steps["model"].classes_, classes)
            yhat = classes[p.argmax(1)]
            k = int(gs.best_params_["selector__k"])
            cvr = pd.DataFrame(gs.cv_results_)
            cvr["k"] = [q["selector__k"] for q in gs.cv_results_["params"]]
            best_i = gs.best_index_
            perf.append({"fold_id": fid, "repeat": rep, "fold": fold, "model": name, "chosen_k": k,
                         "inner_macro_f1": cvr.loc[best_i, "mean_test_score"],
                         "inner_macro_f1_se": cvr.loc[best_i, "std_test_score"] / np.sqrt(len(inner)),
                         "best_params": json.dumps({a: b for a, b in gs.best_params_.items()}, default=str),
                         "n_test": len(te), **evaluate_predictions(yte, yhat, p, classes)})
            for i, sid in enumerate(yte.index):
                pred.append({"fold_id": fid, "repeat": rep, "sample_id": sid, "model": name, "true": yte.iloc[i],
                             "pred": yhat[i], **{f"prob_{c}": p[i, j] for j, c in enumerate(classes)}})
            s = selected_features(pipe, keep)
            s.insert(0, "model", name); s.insert(0, "fold_id", fid); s["chosen_k"] = k
            sel.append(s)
            best_by_k = cvr.groupby("k")["mean_test_score"].max().reset_index()
            best_by_k.insert(0, "model", name); best_by_k.insert(0, "fold_id", fid)
            curve.append(best_by_k)
        except Exception as exc:  # noqa: BLE001
            perf.append({"fold_id": fid, "repeat": rep, "fold": fold, "model": name, "error": repr(exc)})
    fr = lambda rows: pd.DataFrame(rows) if len(rows) else pd.DataFrame()
    return {"perf": fr(perf), "pred": fr(pred), "sel": pd.concat(sel) if sel else pd.DataFrame(),
            "curve": pd.concat(curve) if curve else pd.DataFrame()}


# ----------------------------------------------------------------------------- summaries
def summarize(perf: pd.DataFrame) -> pd.DataFrame:
    ok = perf.dropna(subset=["macro_f1"])
    rows = []
    for m, d in ok.groupby("model"):
        rep_means = d.groupby("repeat")["macro_f1"].mean()          # repeats are the ~independent units
        se = rep_means.std(ddof=1) / np.sqrt(len(rep_means)) if len(rep_means) > 1 else d.macro_f1.std(ddof=1) / np.sqrt(len(d))
        rows.append({"model": m, "n_outer_folds": len(d), "macro_f1_mean": d.macro_f1.mean(), "macro_f1_sd": d.macro_f1.std(ddof=1),
                     "macro_f1_se": se, "balanced_accuracy_mean": d.balanced_accuracy.mean(),
                     "roc_auc_ovr_mean": d.roc_auc_ovr_macro.mean(), "pr_auc_mean": d.pr_auc_macro.mean(),
                     "mcc_mean": d.mcc.mean(), "log_loss_mean": d.log_loss.mean(), "brier_mean": d.brier.mean(),
                     "chosen_k_median": float(d.chosen_k.median()), "chosen_k_mean": d.chosen_k.mean()})
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def choose_model(summary: pd.DataFrame) -> pd.Series:
    """One-SE rule across models: within one SE of the best mean macro-F1, prefer the smallest panel, then the
    simplest model family (elastic net < logistic < SVM < RF < XGBoost)."""
    best = summary.iloc[0]
    elig = summary[summary.macro_f1_mean >= best.macro_f1_mean - best.macro_f1_se].copy()
    elig["cx"] = elig.model.map(C.MODEL_COMPLEXITY).fillna(9)
    elig = elig.sort_values(["chosen_k_median", "cx", "macro_f1_mean"], ascending=[True, True, False])
    return elig.iloc[0]


def aggregate_oof(pred: pd.DataFrame, model: str) -> pd.DataFrame:
    d = pred[pred.model == model]
    pc = [c for c in d.columns if c.startswith("prob_")]
    agg = d.groupby("sample_id").agg(true=("true", "first"), **{c: (c, "mean") for c in pc})
    agg["pred"] = np.array([c[5:] for c in pc])[agg[pc].to_numpy().argmax(1)]
    return agg


def bootstrap_ci(agg: pd.DataFrame, n_boot: int, seed: int = C.SEED) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    pc = [f"prob_{c}" for c in C.CLASSES]
    n = len(agg)
    vals: Dict[str, List[float]] = {"macro_f1": [], "balanced_accuracy": [], "roc_auc_ovr_macro": []}
    for _ in range(n_boot):
        s = agg.iloc[rng.integers(0, n, n)]
        if s.true.nunique() < 3:
            continue
        m = evaluate_predictions(s.true, s.pred, s[pc].to_numpy(), C.CLASSES)
        for k in vals:
            vals[k].append(m[k])
    point = evaluate_predictions(agg.true, agg.pred, agg[pc].to_numpy(), C.CLASSES)
    out = {}
    for k, v in vals.items():
        lo, hi = np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)
        out[k] = point[k]; out[f"{k}_ci_low"] = lo; out[f"{k}_ci_high"] = hi
    return out


def compare_models(perf: pd.DataFrame, ref_model: str) -> pd.DataFrame:
    """Friedman test over all models + paired Wilcoxon vs the chosen model (BH-corrected) on outer-fold macro-F1."""
    piv = perf.dropna(subset=["macro_f1"]).pivot(index="fold_id", columns="model", values="macro_f1").dropna()
    rows = []
    fr_p = stats.friedmanchisquare(*[piv[c] for c in piv.columns])[1] if piv.shape[1] >= 3 else np.nan
    for m in piv.columns:
        if m == ref_model:
            continue
        diff = piv[ref_model] - piv[m]
        try:
            p = stats.wilcoxon(diff)[1] if np.any(diff != 0) else 1.0
        except ValueError:
            p = np.nan
        rows.append({"reference": ref_model, "comparator": m, "mean_diff_macro_f1": diff.mean(), "wilcoxon_p": p,
                     "friedman_p_all_models": fr_p})
    out = pd.DataFrame(rows)
    if len(out):
        out["wilcoxon_q_BH"] = bh_fdr(out.wilcoxon_p)
    return out


def stability_table(sel: pd.DataFrame, model: str, n_folds: int) -> pd.DataFrame:
    d = sel[sel.model == model]
    t = d.groupby("gene_id").agg(selection_count=("fold_id", "nunique"), mean_rank=("rank", "mean"),
                                 mean_f_score=("f_score", "mean")).reset_index()
    t["selection_frequency"] = t.selection_count / n_folds
    return t.sort_values(["selection_frequency", "mean_rank"], ascending=[False, True]).reset_index(drop=True)


def derive_panels(stab: pd.DataFrame, inp: Dict[str, Any], threshold: float) -> tuple[list[str], list[str], dict]:
    """Panels derived from the outer-fold stability table (NOT a fresh SelectKBest on the full data).

    full    : ALL genes with selection frequency >= threshold (topped up to MIN_FINAL_PANEL, capped at
              MAX_FINAL_PANEL). This is the primary signature and the primary final model.
    minimal : `full` after redundancy pruning (|r| > REDUNDANCY_R on covariate-residualised expression keeps the
              more stable gene). Reported as a separate, smaller model; it was NOT what nested CV evaluated and
              is typically too small for network-level analyses.
    """
    info: Dict[str, Any] = {"threshold": threshold, "topped_up": False, "pruned_correlated": []}
    cand = stab[stab.selection_frequency >= threshold]
    if len(cand) < C.MIN_FINAL_PANEL:
        cand = stab.head(C.MIN_FINAL_PANEL)
        info["topped_up"] = True
    genes = cand.head(C.MAX_FINAL_PANEL).gene_id.tolist()
    frame = pd.concat([inp["X"][genes], inp["cov"]], axis=1)
    resid = pd.DataFrame(CovariateResidualizer(genes, list(inp["cov"].columns)).fit(frame).transform(frame), columns=genes)
    corr = resid.corr().abs()
    kept: list[str] = []
    for g in genes:                                                   # genes already ordered by stability
        clash = [k for k in kept if corr.loc[g, k] > C.REDUNDANCY_R]
        if clash:
            info["pruned_correlated"].append({"dropped": g, "kept": clash[0], "abs_r": float(corr.loc[g, clash[0]])})
        else:
            kept.append(g)
    return genes, kept, info


def signature_at(stab: pd.DataFrame, threshold: float) -> list[str]:
    """All genes with selection frequency >= threshold (capped), used for sensitivity analyses."""
    return stab[stab.selection_frequency >= threshold].head(C.MAX_FINAL_PANEL).gene_id.tolist()


def fit_final_model(inp: Dict[str, Any], panel: Sequence[str], model_name: str) -> PanelModel:
    """Fit the deployable model directly on the pre-selected stable panel (all 304 T21 samples)."""
    X, cov, y = inp["X"], inp["cov"], inp["y"]
    panel = list(panel)
    frame = pd.concat([X[panel], cov], axis=1)
    resid = CovariateResidualizer(panel, list(cov.columns)).fit(frame)
    R = resid.transform(frame)
    spec = get_model_specs([model_name])[model_name]
    grids = spec["grid"] if isinstance(spec["grid"], list) else [spec["grid"]]
    grids = [{k: v for k, v in g.items() if k != "selector__k"} for g in grids]      # panel is fixed: no selector
    pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(spec["estimator"]))])
    cv = _inner_splits(y, inp["groups"], C.SEED + 99)
    gs = GridSearchCV(pipe, grids, scoring="f1_macro", cv=cv, refit=True, n_jobs=max(1, C.N_JOBS), error_score=np.nan)
    gs.fit(R, y.to_numpy())
    best = gs.best_estimator_
    return PanelModel(panel, resid, best.named_steps["scaler"], best.named_steps["model"], model_name,
                      {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in gs.best_params_.items()})
