"""
Stage 05 - explainability and biological validation of the final panel.

  A. SHAP: TreeExplainer for XGBoost / Random Forest; LinearExplainer for Elastic Net / Logistic Regression
     (whichever wins the one-SE rule); beeswarm + dependence plots per subtype. SVM (no exact SHAP) falls back to
     permutation importance. Coefficients / permutation importance are always produced as cross-checks.
  B. HSA21 validation: exact polygenic z-scores for Donovan et al.'s Cluster 1 (97 genes) and Cluster 2 (29 genes),
     Spearman correlation (BH-FDR) with the non-HSA21 panel genes; MS1-arm genes should track Cluster 1, MS3-arm
     genes Cluster 2.
  C. Cell-composition benchmark: Gene-only vs Cell-only vs Combined models under IDENTICAL outer folds
     (ROC-AUC, macro-F1, paired Wilcoxon) + per-gene subtype association after adjusting for cell scores.
"""
from __future__ import annotations

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C
import data_access as DA
from hsa21_clusters import adjust_expression, load_clusters, polygenic_zscores
from immune import immune_scores
from ml_components import (ElasticNetClassifier, align_proba, evaluate_predictions, make_splits, prepare_fold)
from utils import (SUBTYPE_COLORS, bh_fdr, get_logger, heatmap, load_json, optional_import, run_stage, save_fig,
                   save_table)

log = get_logger()


# ============================================================================ A. SHAP
def _per_class(sv, n_classes: int):
    """Normalise SHAP output (list | (n,f,c) | (c,n,f) | (n,f)) to a list of (n, f) arrays."""
    if isinstance(sv, list):
        return [np.asarray(a) for a in sv]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        if sv.shape[-1] == n_classes:
            return [sv[:, :, k] for k in range(n_classes)]
        if sv.shape[0] == n_classes:
            return [sv[k] for k in range(n_classes)]
    if sv.ndim == 2:
        return [sv]
    raise ValueError(f"Unexpected SHAP output shape {sv.shape}")


def _compute_shap(shap, est, name: str, Xs: pd.DataFrame, n_classes: int):
    """SHAP values per class for the fitted estimator. Returns (list of (n, f) arrays, explainer name) or (None, None).
    LinearExplainer: elastic net / logistic regression (exact). TreeExplainer: random forest / XGBoost (exact).
    KernelExplainer: everything else (e.g. the SVM) -- model-agnostic, applied to predict_proba with a k-means
    background; approximate, but valid for any classifier."""
    X = Xs.to_numpy()
    model = getattr(est, "model_", est)                                   # unwrap our sklearn/XGB wrappers
    if name in C.LINEAR_MODELS:
        return _per_class(shap.LinearExplainer(model, X).shap_values(X), n_classes), "shap.LinearExplainer"
    if name in C.TREE_MODELS:
        return _per_class(shap.TreeExplainer(model).shap_values(X), n_classes), "shap.TreeExplainer"
    background = shap.kmeans(X, min(10, len(X))) if hasattr(shap, "kmeans") else X[:10]
    explainer = shap.KernelExplainer(est.predict_proba, background)
    # l1_reg=0 matters: with shap's default (feature selection when the coalition space is under-sampled) only ~10
    # features per participant receive non-zero SHAP values, which silently zeroes the rest.
    nsamples = int(2 * X.shape[1] + 2048)
    try:
        values = explainer.shap_values(X, nsamples=nsamples, l1_reg=0, silent=True)
    except TypeError:                                                      # shap version without these keywords
        try:
            values = explainer.shap_values(X, nsamples=nsamples, l1_reg=0)
        except TypeError:
            values = explainer.shap_values(X, nsamples=nsamples)
    return _per_class(values, n_classes), f"shap.KernelExplainer (k-means background, nsamples={nsamples}, l1_reg=0)"


def _shap_outputs(shap, sv, explainer_name: str, est, name: str, Xs: pd.DataFrame, labels, classes, y,
                  fig_prefix: str, table_name: str, note: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Beeswarm (per class), dependence (top-3 genes per class) and local-explanation figures + mean|SHAP| table.
    Returns (mean|SHAP| table, per-class direction table = Spearman(expression, SHAP))."""
    imp = pd.DataFrame({c: np.abs(sv[k]).mean(axis=0) for k, c in enumerate(classes)}, index=labels)
    imp["mean_abs_shap_all_classes"] = imp.mean(axis=1)
    imp["model"] = name
    save_table(imp.sort_values("mean_abs_shap_all_classes", ascending=False), table_name)
    direction = pd.DataFrame({c: [stats.spearmanr(Xs[g], sv[k][:, i])[0] for i, g in enumerate(labels)]
                              for k, c in enumerate(classes)}, index=labels)
    for k, c in enumerate(classes):
        plt.figure(figsize=(6.5, max(3.5, 0.28 * len(labels) + 1.5)))
        shap.summary_plot(sv[k], Xs, feature_names=labels, show=False, max_display=min(25, len(labels)))
        plt.title(f"SHAP beeswarm - {c} ({explainer_name.split(' ')[0]}, {name}{note})", fontsize=9)
        save_fig(f"{fig_prefix}_shap_beeswarm_{c}.png")
    top = {c: imp[c].nlargest(3).index.tolist() for c in classes}
    fig, axes = plt.subplots(len(classes), 3, figsize=(13, 3.6 * len(classes)))
    for k, c in enumerate(classes):
        for j, g in enumerate(top[c]):
            a_ = axes[k, j]; gi = list(labels).index(g)
            a_.scatter(Xs[g], sv[k][:, gi], s=9, alpha=.7, color=SUBTYPE_COLORS.get(c, "#4c72b0"))
            a_.axhline(0, color="#999999", lw=.7)
            a_.set_xlabel(f"{g} (standardised expression)", fontsize=8); a_.set_ylabel(f"SHAP value ({c})", fontsize=8)
            a_.set_title(f"{c}: {g}", fontsize=9)
    fig.tight_layout()
    save_fig(f"{fig_prefix}_shap_dependence.png", fig)
    proba = est.predict_proba(Xs.to_numpy())
    fig, axes = plt.subplots(1, len(classes), figsize=(14, 4.5))
    for k, c in enumerate(classes):
        idx = np.where(y.to_numpy() == c)[0]
        i = idx[np.argmax(proba[idx, k])]
        order = np.argsort(np.abs(sv[k][i]))[-10:]
        axes[k].barh(np.array(labels)[order], sv[k][i][order], color=np.where(sv[k][i][order] > 0, "#d62728", "#1f77b4"))
        axes[k].set_title(f"{c}: representative participant (p={proba[i, k]:.2f})", fontsize=9); axes[k].tick_params(labelsize=7)
    fig.tight_layout()
    save_fig(f"{fig_prefix}_shap_local_examples.png", fig)
    return imp, direction


def _shap_agreement(imp_a, dir_a, name_a, imp_b, dir_b, name_b, classes) -> None:
    """Quantify whether two models explain the data with the same genes and the same directions."""
    rows = []
    rho, p = stats.spearmanr(imp_a["mean_abs_shap_all_classes"], imp_b.loc[imp_a.index, "mean_abs_shap_all_classes"])
    rows.append({"metric": "Spearman of mean|SHAP| across genes (all classes)", "value": rho, "p": p})
    for c in classes:
        r, p = stats.spearmanr(imp_a[c], imp_b.loc[imp_a.index, c])
        rows.append({"metric": f"Spearman of mean|SHAP| across genes ({c})", "value": r, "p": p})
    for n in (3, 5):
        n = min(n, len(imp_a))
        ta = set(imp_a["mean_abs_shap_all_classes"].nlargest(n).index)
        tb = set(imp_b["mean_abs_shap_all_classes"].nlargest(n).index)
        rows.append({"metric": f"top-{n} gene overlap (count)", "value": len(ta & tb), "p": np.nan})
    same = (np.sign(dir_a) == np.sign(dir_b.loc[dir_a.index])).to_numpy()
    rows.append({"metric": "direction agreement (sign of Spearman(expression, SHAP)), all gene x class", "value": float(same.mean()), "p": np.nan})
    out = pd.DataFrame(rows)
    out.insert(0, "model_A", name_a); out.insert(1, "model_B", name_b)
    save_table(out, "Table6c_shap_model_agreement.csv", index=False)
    both = dir_a.add_prefix(f"{name_a}_").join(dir_b.add_prefix(f"{name_b}_"))
    save_table(both, "Table6d_shap_direction_by_model.csv")
    log.info("SHAP agreement %s vs %s: Spearman(mean|SHAP|)=%.2f, direction agreement=%.0f%%", name_a, name_b, rho, 100 * same.mean())


def explain(fm, inp, ann, labels) -> None:
    Xs = pd.DataFrame(fm.transform(inp["X"], inp["cov"]), index=inp["X"].index, columns=labels)
    est, name = fm.estimator, fm.model_name
    classes = list(fm.classes_)
    y = inp["y"].reindex(Xs.index)
    shap = optional_import("shap")

    # permutation importance (model-agnostic cross-check, in-sample on the final fit; report as such)
    pi = permutation_importance(est, Xs.to_numpy(), y.to_numpy(), scoring="f1_macro", n_repeats=30, random_state=C.SEED, n_jobs=1)
    pi_df = pd.DataFrame({"gene": labels, "perm_importance_mean": pi.importances_mean, "perm_importance_sd": pi.importances_std})
    if hasattr(est, "coef_"):
        for j, c in enumerate(classes):
            pi_df[f"coef_{c}"] = est.coef_[j]
    save_table(pi_df.sort_values("perm_importance_mean", ascending=False), "Fig8_permutation_importance_and_coefficients.csv", index=False)

    imp_a = dir_a = None
    if shap is None:
        log.warning("shap is not installed (pip install shap) -> SHAP figures skipped; permutation / coefficient plots produced instead")
    else:
        try:
            sv, ename = _compute_shap(shap, est, name, Xs, len(classes))
            if len(sv) != len(classes):
                raise ValueError(f"expected {len(classes)} per-class SHAP arrays, got {len(sv)}")
            log.info("primary model '%s' explained with %s", name, ename)
            imp_a, dir_a = _shap_outputs(shap, sv, ename, est, name, Xs, labels, classes, y, "Fig8", "Table6b_mean_abs_SHAP.csv")
        except Exception as exc:  # noqa: BLE001
            log.error("SHAP for the primary model failed (%r) -> permutation / coefficient plots only", exc)

        # Secondary, clearly-labelled interpretability cross-check: elastic net (LinearExplainer) fitted on the SAME panel.
        if name not in C.LINEAR_MODELS:
            try:
                import cv_engine as E
                fm2 = E.fit_final_model(inp, fm.genes, "elastic_net")
                joblib.dump(fm2, C.MODELS / "final_subtype_model_elastic_net_secondary.joblib")
                Xs2 = pd.DataFrame(fm2.transform(inp["X"], inp["cov"]), index=inp["X"].index, columns=labels)
                sv2, ename2 = _compute_shap(shap, fm2.estimator, "elastic_net", Xs2, len(classes))
                imp_b, dir_b = _shap_outputs(shap, sv2, ename2, fm2.estimator, "elastic_net", Xs2, labels, classes, y,
                                             "Fig8S_elastic_net", "Table6b_mean_abs_SHAP_elastic_net_secondary.csv",
                                             note="; SECONDARY cross-check")
                pred1, pred2 = est.predict(Xs.to_numpy()), fm2.estimator.predict(Xs2.to_numpy())
                log.info("secondary elastic net (same %d-gene panel): agrees with the primary model's in-sample "
                         "predictions for %.1f%% of participants", len(labels), 100 * float((pred1 == pred2).mean()))
                if imp_a is not None:
                    _shap_agreement(imp_a, dir_a, name, imp_b, dir_b, "elastic_net", classes)
            except Exception as exc:  # noqa: BLE001
                log.error("secondary elastic-net SHAP failed (%r) -> skipped (primary outputs unaffected)", exc)

    # coefficient / permutation-importance overview (always)
    has_coef = hasattr(est, "coef_")
    fig, ax = plt.subplots(1, 2 if has_coef else 1, figsize=(14 if has_coef else 7, max(4, 0.25 * len(labels) + 1.5)))
    ax = np.atleast_1d(ax)
    p2 = pi_df.sort_values("perm_importance_mean")
    ax[0].barh(p2.gene, p2.perm_importance_mean, xerr=p2.perm_importance_sd, color="#4c72b0"); ax[0].set_title("Permutation importance (macro-F1 drop)")
    ax[0].tick_params(axis="y", labelsize=7)
    if hasattr(est, "coef_"):
        cm = pd.DataFrame(est.coef_, index=classes, columns=labels).T
        cm = cm.loc[cm.abs().max(axis=1).sort_values().index]
        heatmap(ax[1], cm.to_numpy(), classes, cm.index, vmin=-np.abs(cm.to_numpy()).max(), vmax=np.abs(cm.to_numpy()).max(), cbar_label="coefficient")
        ax[1].tick_params(axis="y", labelsize=7); ax[1].set_title("Class-specific model coefficients")
    fig.tight_layout()
    save_fig("Fig8_coefficients_and_permutation_importance.png", fig)


# ============================================================================ B. HSA21 validation
def hsa21_validation(inp, ann, panel_ids, labels, tag: str = "") -> None:
    man = inp["manifest"]
    hs = pd.read_csv(C.PROCESSED / "hsa21_protein_coding_genes.csv")
    hs = hs[hs.gene_symbol.notna() & ~hs.gene_symbol.duplicated()]
    lc = DA.load_logcpm()
    hs = hs[hs.gene_id.isin(lc.columns)]
    cov = inp["cov"]
    adj21 = adjust_expression(lc.loc[cov.index, hs.gene_id.tolist()], cov)
    adj21.columns = hs.gene_symbol.tolist()
    clusters, source = load_clusters(adj21)
    scores = polygenic_zscores(adj21, clusters)
    save_table(scores.join(inp["y"].rename("subtype")), "HSA21_cluster_polygenic_zscores.csv")
    save_table(pd.DataFrame([{"cluster": k, "n_genes_used": len([g for g in v if g in adj21.columns]), "published_n": {1: 97, 2: 29}[k],
                              "source": source} for k, v in clusters.items()]), "HSA21_cluster_definitions.csv", index=False)
    genes_adj = pd.DataFrame(adjust_expression(lc.loc[cov.index, panel_ids], cov).values, index=cov.index, columns=labels)
    rows = []
    for g in labels:
        for k in (1, 2):
            rho, p = stats.spearmanr(genes_adj[g], scores[f"cluster{k}_z"])
            rows.append({"gene": g, "cluster": f"HSA21_cluster{k}", "rho": rho, "pvalue": p})
    corr = pd.DataFrame(rows)
    corr["padj_BH"] = bh_fdr(corr.pvalue)
    # arm assignment (MS3-vs-MS1 effect on the adjusted panel expression)
    y = inp["y"]
    d = {g: (genes_adj.loc[y == "MS3", g].mean() - genes_adj.loc[y == "MS1", g].mean()) /
            np.sqrt((genes_adj.loc[y == "MS3", g].var() + genes_adj.loc[y == "MS1", g].var()) / 2) for g in labels}
    corr["arm"] = corr.gene.map(lambda g: "MS3-up" if d[g] > 0 else "MS1-up")
    corr["cohens_d_MS3_vs_MS1"] = corr.gene.map(d)
    corr["cluster_score_source"] = source
    save_table(corr, f"Table9_panel_vs_HSA21_cluster_correlations{tag}.csv", index=False)

    w = corr.pivot(index="gene", columns="cluster", values="rho")
    arm = corr.drop_duplicates("gene").set_index("gene")["arm"].reindex(w.index)
    summ = []
    for a, c, expect in [("MS1-up", "HSA21_cluster1", ">0"), ("MS3-up", "HSA21_cluster2", ">0"),
                         ("MS1-up", "HSA21_cluster2", "<0"), ("MS3-up", "HSA21_cluster1", "<0")]:
        v = w.loc[arm == a, c].dropna()
        if len(v) >= 3:
            wp = stats.wilcoxon(v, alternative="greater" if expect == ">0" else "less")[1]
            summ.append({"panel_arm": a, "cluster": c, "expected_sign": expect, "n_genes": len(v), "median_rho": v.median(),
                         "fraction_expected_sign": float((v > 0).mean() if expect == ">0" else (v < 0).mean()), "wilcoxon_p": wp})
    save_table(pd.DataFrame(summ), f"HSA21_expected_direction_tests{tag}.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(16, max(4.5, 0.26 * len(labels) + 2)))
    wo = w.loc[w.abs().max(axis=1).sort_values().index]
    heatmap(ax[0], wo.to_numpy(), ["Cluster 1\n(97 genes)", "Cluster 2\n(29 genes)"], wo.index, cbar_label="Spearman rho")
    ax[0].tick_params(axis="y", labelsize=7)
    sig = corr.pivot(index="gene", columns="cluster", values="padj_BH").reindex(wo.index)
    for i in range(sig.shape[0]):
        for j in range(2):
            if sig.iloc[i, j] < 0.05: ax[0].text(j, i, "*", ha="center", va="center", fontsize=9)
    ax[0].set_title(f"Panel genes vs HSA21 clusters\n(* BH q<0.05; cluster lists: {source.split(', sheet')[0].replace('published (', 'published, ').rstrip(')')})", fontsize=8)
    for k, c in enumerate(C.CLASSES):
        ax[1].boxplot([scores.loc[y == c, f"cluster{i}_z"] for i in (1, 2)], positions=np.array([1, 2]) + (k - 1) * .25, widths=.22,
                      patch_artist=True, boxprops=dict(facecolor=SUBTYPE_COLORS[c], alpha=.7), medianprops=dict(color="k"))
    ax[1].set_xticks([1, 2]); ax[1].set_xticklabels(["Cluster 1 z", "Cluster 2 z"]); ax[1].set_title("Polygenic cluster z-scores\n(red MS1, green MS2, blue MS3)")
    for c in C.CLASSES:
        k = y == c; ax[2].scatter(scores.loc[k, "cluster1_z"], scores.loc[k, "cluster2_z"], s=10, color=SUBTYPE_COLORS[c], label=c, alpha=.7)
    ax[2].set_xlabel("HSA21 Cluster 1 z"); ax[2].set_ylabel("HSA21 Cluster 2 z"); ax[2].legend(); ax[2].set_title("Participants in cluster-score space")
    fig.tight_layout()
    save_fig(f"Fig9_HSA21_cluster_validation{tag}.png", fig)


# ============================================================================ C. cell-composition benchmark
class GenesPlusCells(BaseEstimator, TransformerMixin):
    """SelectKBest on the first n_genes columns (fit on training data only); remaining columns pass through."""
    def __init__(self, n_genes: int, k: int = 20, use_cells: bool = True):
        self.n_genes, self.k, self.use_cells = n_genes, k, use_cells

    def fit(self, X, y):
        self.sel_ = SelectKBest(f_classif, k=min(self.k, self.n_genes)).fit(X[:, :self.n_genes], y)
        return self

    def transform(self, X):
        g = self.sel_.transform(X[:, :self.n_genes])
        return np.hstack([g, X[:, self.n_genes:]]) if self.use_cells else g


def _fit_eval(name, Xtr, ytr, Xte, yte, inner, n_genes, k, classes):
    if name == "Cell-only":
        pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000, class_weight="balanced"))])
        grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    else:
        pipe = Pipeline([("gc", GenesPlusCells(n_genes, k, use_cells=(name == "Combined"))), ("scaler", StandardScaler()),
                         ("model", ElasticNetClassifier(random_state=C.SEED))])
        grid = {"model__C": [0.01, 0.1, 1.0], "model__l1_ratio": [0.25, 0.75]}
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner, n_jobs=1, refit=True).fit(Xtr, ytr)
    p = align_proba(gs.predict_proba(Xte), gs.best_estimator_.named_steps["model"].classes_, classes)
    return evaluate_predictions(yte, np.array(classes)[p.argmax(1)], p, classes)


def _bench_fold(r, f, tr, te, inp, cells, k):
    classes = C.CLASSES
    Xtr, Xte, _, _ = prepare_fold(inp["X"], inp["expressed"], inp["cov"], tr, te, C.PREFILTER_TOP, C.MIN_SAMPLE_FRACTION)
    ytr, yte = inp["y"].iloc[tr].to_numpy(), inp["y"].iloc[te].to_numpy()
    ctr, cte = cells.iloc[tr].to_numpy(float), cells.iloc[te].to_numpy(float)
    inner = [(a, b) for _, _, a, b in make_splits(pd.Series(ytr), None, C.INNER_SPLITS, 1, C.SEED + 500 + 10 * r + f)]
    out = []
    for name in ["Gene-only", "Cell-only", "Combined"]:
        if name == "Cell-only":
            m = _fit_eval(name, ctr, ytr, cte, yte, inner, 0, k, classes)
        elif name == "Gene-only":
            m = _fit_eval(name, Xtr, ytr, Xte, yte, inner, Xtr.shape[1], k, classes)
        else:
            m = _fit_eval(name, np.hstack([Xtr, ctr]), ytr, np.hstack([Xte, cte]), yte, inner, Xtr.shape[1], k, classes)
        out.append({"fold_id": f"r{r}f{f}", "model": name, **m})
    return out


def cell_benchmark(inp, ann, meta) -> None:
    cells, src = immune_scores(DA.load_logcpm(), ann, inp["X"].index)
    save_table(cells.join(inp["y"].rename("subtype")), "immune_scores_T21.csv")
    tests = []
    for c in cells.columns:
        h = stats.kruskal(*[cells.loc[inp["y"] == s, c] for s in C.CLASSES])
        tests.append({"cell_type": c, "kruskal_p": h.pvalue, **{f"median_{s}": cells.loc[inp["y"] == s, c].median() for s in C.CLASSES}})
    t = pd.DataFrame(tests); t["padj_BH"] = bh_fdr(t.kruskal_p); t["score_source"] = src
    save_table(t, "immune_scores_by_subtype.csv", index=False)

    k = int(np.clip(round(meta.get("chosen_k_median", 20)), 5, 100))
    k = min(C.PANEL_SIZES, key=lambda x: abs(x - k))
    splits = list(make_splits(inp["y"], inp["groups"], C.OUTER_SPLITS, C.OUTER_REPEATS, C.SEED))
    log.info("3-way benchmark: %d outer folds (identical to stage 03), panel size k=%d, cell scores: %s", len(splits), k, src)
    res = Parallel(n_jobs=C.N_JOBS, verbose=0)(delayed(_bench_fold)(r, f, tr, te, inp, cells, k) for r, f, tr, te in splits)
    perf = pd.DataFrame([x for r in res for x in r])
    save_table(perf, "cell_benchmark_by_fold.csv", index=False)
    order = ["Gene-only", "Cell-only", "Combined"]
    summ = perf.groupby("model").agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                                     roc_auc_mean=("roc_auc_ovr_macro", "mean"), roc_auc_sd=("roc_auc_ovr_macro", "std"),
                                     balanced_accuracy_mean=("balanced_accuracy", "mean")).reindex(order).reset_index()
    piv_f1 = perf.pivot(index="fold_id", columns="model", values="macro_f1"); piv_auc = perf.pivot(index="fold_id", columns="model", values="roc_auc_ovr_macro")
    cmp_rows = []
    for a, b in [("Gene-only", "Cell-only"), ("Combined", "Gene-only"), ("Combined", "Cell-only")]:
        for nm, piv in [("macro_f1", piv_f1), ("roc_auc_ovr_macro", piv_auc)]:
            d = piv[a] - piv[b]
            p = stats.wilcoxon(d)[1] if np.any(d != 0) else 1.0
            cmp_rows.append({"comparison": f"{a} - {b}", "metric": nm, "mean_difference": d.mean(), "wilcoxon_p": p})
    cmp_ = pd.DataFrame(cmp_rows); cmp_["padj_BH"] = bh_fdr(cmp_.wilcoxon_p)
    summ["cell_score_source"] = src
    save_table(summ, "Table8_cell_composition_benchmark.csv", index=False)
    save_table(cmp_, "Table8b_cell_benchmark_paired_tests.csv", index=False)
    log.info("cell benchmark:\n%s", summ[["model", "macro_f1_mean", "roc_auc_mean"]].to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for a, col, ttl in [(ax[0], "roc_auc_ovr_macro", "ROC-AUC (OvR macro)"), (ax[1], "macro_f1", "Macro-F1")]:
        a.boxplot([perf.loc[perf.model == m, col] for m in order], labels=order, showmeans=True, patch_artist=True,
                  boxprops=dict(facecolor="#cfe0f3")); a.set_title(ttl + " - outer folds")
    # per-gene subtype association after adjusting for cell scores
    ids = inp["X"].columns.intersection(pd.Index(meta["panel"]))
    Y = inp["X"][meta["panel"]].to_numpy(float)
    covd = pd.concat([inp["cov"][C.CONTINUOUS_COVARIATES].astype(float),
                      pd.get_dummies(inp["cov"][C.CATEGORICAL_COVARIATES].astype(str), drop_first=True, dtype=float)], axis=1)
    base = np.hstack([np.ones((len(Y), 1)), covd.to_numpy(), StandardScaler().fit_transform(cells.to_numpy(float))])
    sub = pd.get_dummies(inp["y"].astype(str), drop_first=True, dtype=float).to_numpy()
    def rss(Xd): b = np.linalg.lstsq(Xd, Y, rcond=None)[0]; return ((Y - Xd @ b) ** 2).sum(0)
    r0, r1 = rss(base), rss(np.hstack([base, sub]))
    df1, df2 = sub.shape[1], len(Y) - base.shape[1] - sub.shape[1]
    F = ((r0 - r1) / df1) / (r1 / df2); p = stats.f.sf(F, df1, df2)
    tot = ((Y - Y.mean(0)) ** 2).sum(0)
    r2_cells = 1 - rss(np.hstack([np.ones((len(Y), 1)), covd.to_numpy(), cells.to_numpy(float)])) / tot
    pg = pd.DataFrame({"gene_id": meta["panel"], "gene_symbol": ann.reindex(meta["panel"])["gene_symbol"].values,
                       "subtype_F_adj_cells": F, "p_adj_cells": p, "R2_explained_by_cells_and_covariates": r2_cells})
    pg["q_BH"] = bh_fdr(pg.p_adj_cells)
    save_table(pg, "Table8c_panel_genes_subtype_association_adjusted_for_cells.csv", index=False)
    ax[2].hist(r2_cells, bins=12, color="#dd8452"); ax[2].set_xlabel("R2 of gene explained by cell scores + covariates")
    ax[2].set_title(f"Panel genes: {int((pg.q_BH < .05).sum())}/{len(pg)} keep subtype signal\nafter cell adjustment (q<0.05)", fontsize=9)
    fig.tight_layout()
    save_fig("Fig10c_cell_composition_benchmark.png", fig)


def main() -> None:
    warnings.filterwarnings("ignore")
    meta = load_json(C.MODELS / "final_model_metadata.json")
    fm = joblib.load(C.MODELS / "final_subtype_model.joblib")
    inp = DA.load_ml_inputs()
    ann = DA.load_annotation()
    panel = meta["panel"]
    sym = ann.reindex(panel)["gene_symbol"].fillna("").astype(str)
    labels = [s if s and s not in list(sym[:i]) else panel[i] for i, s in enumerate(sym)]
    log.info("explaining %s with %d panel genes (%s)", meta["model"], len(panel), ", ".join(labels[:8]))
    explain(fm, inp, ann, labels)
    hsa21_validation(inp, ann, panel, labels)
    sig60 = meta.get("signature_60") or []
    if sig60 and set(sig60) != set(panel):                       # sensitivity: more genes -> populates the MS3 arm
        sym60 = ann.reindex(sig60)["gene_symbol"].fillna("").astype(str).tolist()
        labels60, seen = [], set()
        for g, sy in zip(sig60, sym60):
            lab = sy if sy and sy not in seen else g
            seen.add(lab); labels60.append(lab)
        log.info("HSA21 validation repeated for the 60%% signature (%d genes)", len(sig60))
        hsa21_validation(inp, ann, sig60, labels60, tag="_freq60")
    cell_benchmark(inp, ann, meta)


if __name__ == "__main__":
    run_stage("05 explainability, HSA21 validation, cell benchmark", main)
