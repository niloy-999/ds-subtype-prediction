"""
Stage 04 - independent external validation on GSE35665 (15 DS vs 22 controls, Affymetrix microarray).

GSE35665 has NO MS1-MS3 labels, so no subtype accuracy/AUC is (or can be) reported. Instead:
  (a) CO-EXPRESSION NETWORK PRESERVATION: correlation matrices of the panel genes in the discovery T21 cohort
      vs the independent cohort (Mantel-type Spearman of off-diagonal correlations, gene-label permutation
      p-value, sign concordance, mean |difference|). "Preserved" is what is tested - identical matrices are
      neither expected (different platform/age/tissue processing) nor required.
  (b) COMPOSITE SIGNATURE SCORES: direction weights come ONLY from the discovery cohort (MS3-vs-MS1 effect size
      of every panel gene); genes are z-scored within GSE35665; the DS/control labels are used only for
      comparing the resulting score distributions.
  (c) MULTIMODAL VARIANCE: Gaussian-mixture BIC (1 vs 2 vs 3 components), Brown-Forsythe variance tests and the
      MS1-arm/MS3-arm antagonism within the independent DS population.
Sample size (n=15 DS) limits power; results are reported as supportive, not confirmatory.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

import config as C
import data_access as DA
from hsa21_clusters import adjust_expression
from utils import (SUBTYPE_COLORS, bh_fdr, get_logger, heatmap, load_json, parse_platform_table, parse_series_matrix,
                   probes_to_symbols, run_stage, save_fig, save_table, optional_import)

log = get_logger()


def infer_groups(meta: pd.DataFrame) -> pd.Series:
    for col in meta.columns:
        t = meta[col].astype(str).str.lower()
        ds = t.str.contains(r"down|trisomy|\bds\b|t21")
        ct = t.str.contains(r"control|normal|healthy|euploid|disomy|\bctl\b|\bd21\b") & ~ds
        if ds.sum() >= 3 and ct.sum() >= 3 and (ds | ct).all():
            log.info("GSE35665 phenotype column: %r (DS=%d, control=%d)", col, ds.sum(), ct.sum())
            return pd.Series(np.where(ds, "DS", "Control"), index=meta.index)
    raise RuntimeError("Could not infer DS/Control from GSE35665 metadata; inspect columns: " + str(list(meta.columns)))


def load_external() -> tuple[pd.DataFrame, pd.Series]:
    expr, meta, gpl = parse_series_matrix(C.GSE35665_MATRIX)
    groups = infer_groups(meta)
    if np.nanmax(expr.to_numpy()) > 40:
        log.warning("GSE35665 values look un-logged (max>40) -> log2 transform applied")
        expr = np.log2(expr.clip(lower=1))
    sym = probes_to_symbols(parse_platform_table(C.GSE35665_PLATFORM))
    e = expr.loc[expr.index.isin(sym.index)].copy()
    e["symbol"] = sym.reindex(e.index).values
    gene = e.groupby("symbol").median()                       # predefined rule: median over probes per gene
    log.info("GSE35665: %d probes -> %d gene symbols, %d samples (%s)", len(expr), len(gene), gene.shape[1], gpl)
    return gene, groups.reindex(gene.columns)


def offdiag(M: np.ndarray) -> np.ndarray:
    return M[np.triu_indices_from(M, 1)]


def preservation(disc: pd.DataFrame, ext: pd.DataFrame, n_perm: int, seed: int = C.SEED) -> dict:
    """Preservation of the correlation structure of `disc` in `ext` (same genes, same order)."""
    a, b = offdiag(disc.to_numpy()), offdiag(ext.to_numpy())
    rho_obs = stats.spearmanr(a, b)[0]
    rng = np.random.default_rng(seed)
    E = ext.to_numpy(); iu = np.triu_indices_from(E, 1); ra = stats.rankdata(a)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(E.shape[0])
        null[i] = np.corrcoef(ra, stats.rankdata(E[np.ix_(p, p)][iu]))[0, 1]
    return {"n_genes": disc.shape[0], "n_pairs": len(a), "spearman_of_correlations": rho_obs,
            "pearson_of_correlations": stats.pearsonr(a, b)[0], "permutation_p": (1 + (null >= rho_obs).sum()) / (n_perm + 1),
            "sign_concordance": float((np.sign(a) == np.sign(b)).mean()), "mean_abs_difference": float(np.abs(a - b).mean())}


def differential_preservation(disc: pd.DataFrame, E: pd.DataFrame, grp: pd.Series, n_perm: int, seed: int = C.SEED) -> dict:
    """Is the discovery co-expression pattern MORE preserved in DS than in controls?
    Statistic = Spearman(discovery pairwise rho, DS pairwise rho) - Spearman(discovery pairwise rho, control pairwise rho).
    Null: permute the DS/Control labels across the independent cohort's samples (group sizes fixed), recompute both
    correlation matrices. Two-sided. This directly addresses whether preservation is DS-specific or simply reflects
    gene co-variation that exists in blood in general."""
    X = E.to_numpy()
    is_ds = (grp.reindex(E.columns) == "DS").to_numpy()
    iu = np.triu_indices(X.shape[0], 1)
    ref = stats.rankdata(offdiag(disc.to_numpy()))

    def pres(cols: np.ndarray) -> float:
        r = stats.rankdata(X[:, cols], axis=1)
        c = np.corrcoef(r)[iu]
        return float(np.corrcoef(ref, stats.rankdata(np.nan_to_num(c)))[0, 1])

    idx = np.arange(X.shape[1])
    obs_ds, obs_ct = pres(idx[is_ds]), pres(idx[~is_ds])
    obs = obs_ds - obs_ct
    rng = np.random.default_rng(seed)
    n_ds = int(is_ds.sum())
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(idx)
        null[i] = pres(perm[:n_ds]) - pres(perm[n_ds:])
    return {"n_genes": X.shape[0], "n_pairs": len(iu[0]), "n_DS": n_ds, "n_Control": int((~is_ds).sum()),
            "preservation_DS": obs_ds, "preservation_Control": obs_ct, "difference_DS_minus_Control": obs,
            "permutation_p_two_sided": (1 + int((np.abs(null) >= abs(obs)).sum())) / (n_perm + 1)}


def gmm_bic(x: np.ndarray, kmax: int) -> dict:
    out = {}
    for k in range(1, kmax + 1):
        if len(x) <= k * (x.shape[1] + 1):
            break
        gm = GaussianMixture(k, covariance_type="full" if x.shape[1] == 1 else "diag", n_init=10, random_state=C.SEED, reg_covar=1e-3).fit(x)
        out[k] = gm.bic(x)
    return out


def main() -> None:
    meta = load_json(C.MODELS / "final_model_metadata.json")
    ann = DA.load_annotation()
    inp = DA.load_ml_inputs()
    ext, grp = load_external()
    validate(meta["panel"], ann, inp, ext, grp, tag="", label="stable genes, frequency >= %.0f%%" % (100 * meta["stability_threshold"]))
    sig60 = meta.get("signature_60") or []
    if sig60 and set(sig60) != set(meta["panel"]):
        validate(sig60, ann, inp, ext, grp, tag="_freq60", label="SENSITIVITY: stable genes, frequency >= 60%")


def validate(panel, ann, inp, ext, grp, tag: str, label: str) -> None:
    log.info("---- external validation: %s (%d genes) ----", label, len(panel))
    sym = ann.loc[panel, "gene_symbol"].astype(str)
    sym = sym[(sym != "") & ~sym.duplicated()]
    common = [g for g in sym.values if g in ext.index]
    save_table(pd.DataFrame({"gene_id": sym.index, "gene_symbol": sym.values, "on_GSE35665": [g in ext.index for g in sym.values]}),
               f"GSE35665_panel_gene_mapping{tag}.csv", index=False)
    log.info("signature genes present in GSE35665: %d / %d (pairs for the preservation test: %d)", len(common), len(sym),
             len(common) * (len(common) - 1) // 2)
    if len(common) < 4:
        log.warning("Fewer than 4 signature genes are measurable in GSE35665 -> co-expression/composite validation not possible.")
        save_table(pd.DataFrame([{"status": "insufficient overlap", "n_common": len(common)}]), f"GSE35665_validation_status{tag}.csv", index=False)
        return
    if len(common) < 8:
        log.warning("Only %d measurable genes (%d correlation pairs): the permutation test cannot reach conventional "
                    "significance with so few genes (only %d distinct gene-label permutations exist). Treat as underpowered.",
                    len(common), len(common) * (len(common) - 1) // 2, math.factorial(len(common)))
    id_of = {s: i for i, s in sym.items()}
    ids = [id_of[s] for s in common]

    # ---------------------------------------------------------------- discovery quantities (T21, adjusted)
    adj = adjust_expression(inp["X"][ids], inp["cov"])
    adj.columns = common
    disc_corr = adj.corr(method="spearman")
    y = inp["y"]
    d = {}
    for g in common:                                             # MS3 - MS1 standardised effect (Cohen's d)
        a, b = adj.loc[y == "MS3", g], adj.loc[y == "MS1", g]
        sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
        d[g] = (a.mean() - b.mean()) / sp
    d = pd.Series(d)
    w = d / d.abs().sum()
    ms3_genes, ms1_genes = d.index[d > 0].tolist(), d.index[d < 0].tolist()
    if len(ms3_genes) < 2 or len(ms1_genes) < 2:
        log.warning("An arm of the signature has fewer than 2 measurable genes (MS3-up: %d, MS1-up: %d): the "
                    "per-arm signature scores and the two-arm plot are not meaningful for this signature.",
                    len(ms3_genes), len(ms1_genes))
    save_table(pd.DataFrame({"gene_symbol": d.index, "cohens_d_MS3_vs_MS1": d.values, "weight": w.values,
                             "arm": np.where(d > 0, "MS3-up", "MS1-up")}), f"GSE35665_signature_weights_from_discovery{tag}.csv", index=False)

    # ---------------------------------------------------------------- (a) co-expression preservation
    from scipy.spatial.distance import squareform
    dmat = np.clip(1 - disc_corr.abs().to_numpy(), 0, None).copy(); np.fill_diagonal(dmat, 0)
    order = leaves_list(linkage(squareform(dmat, checks=False), "average"))
    genes_o = [common[i] for i in order]
    disc_o = disc_corr.loc[genes_o, genes_o]
    E = ext.loc[genes_o]
    subsets = {"DS": E.loc[:, grp == "DS"], "Control": E.loc[:, grp == "Control"], "All": E}
    pres, ext_corrs = [], {}
    for name, sub in subsets.items():
        ec = sub.T.corr(method="spearman").fillna(0)
        ext_corrs[name] = ec
        pres.append({"external_subset": name, "n_samples": sub.shape[1], **preservation(disc_o, ec, C.P["n_perm"])})
    pres = pd.DataFrame(pres)
    pres["permutation_q_BH"] = bh_fdr(pres.permutation_p)
    save_table(pres, f"GSE35665_coexpression_preservation{tag}.csv", index=False)
    dp = differential_preservation(disc_o, E, grp, C.P["n_perm"])
    save_table(pd.DataFrame([dp]), f"GSE35665_DS_vs_Control_differential_preservation{tag}.csv", index=False)
    log.info("DS vs Control differential preservation: DS %.2f vs Control %.2f (difference %+.2f, permutation p=%.3g, two-sided)",
             dp["preservation_DS"], dp["preservation_Control"], dp["difference_DS_minus_Control"], dp["permutation_p_two_sided"])
    log.info("co-expression preservation:\n%s", pres[["external_subset", "n_samples", "spearman_of_correlations", "permutation_p", "sign_concordance"]].to_string(index=False))

    fig, ax = plt.subplots(1, 4, figsize=(25, 6.4), gridspec_kw={"width_ratios": [1, 1, 1, 0.85]})
    heatmap(ax[0], disc_o.to_numpy(), genes_o, genes_o, cbar_label="Spearman rho"); ax[0].set_title("Discovery: GSE190125 T21 (n=%d)" % len(adj))
    heatmap(ax[1], ext_corrs["DS"].to_numpy(), genes_o, genes_o, cbar_label="Spearman rho"); ax[1].set_title("GSE35665 DS (n=%d)" % (grp == "DS").sum())
    heatmap(ax[2], ext_corrs["Control"].to_numpy(), genes_o, genes_o, cbar_label="Spearman rho"); ax[2].set_title("GSE35665 controls (n=%d)" % (grp == "Control").sum())
    for ttl in range(3):
        ax[ttl].tick_params(labelsize=6.5)
    a_, b_ = offdiag(disc_o.to_numpy()), offdiag(ext_corrs["DS"].to_numpy())
    ax[3].scatter(a_, b_, s=10, alpha=.6); ax[3].axhline(0, c="k", lw=.5); ax[3].axvline(0, c="k", lw=.5)
    r = pres.iloc[0]; ax[3].set_title(f"Pairwise correlations: DS vs discovery\nrho={r.spearman_of_correlations:.2f}, perm p={r.permutation_p:.3g}")
    ax[3].set_xlabel("discovery rho"); ax[3].set_ylabel("GSE35665 DS rho")
    fig.tight_layout(w_pad=4.0)
    save_fig(f"Fig10a_coexpression_preservation{tag}.png", fig)

    # ---------------------------------------------------------------- (b) composite scores
    def scores(M: pd.DataFrame) -> pd.DataFrame:                  # M: genes x samples, dataset-specific z-scoring
        z = M.T.apply(lambda c: (c - c.mean()) / (c.std(ddof=0) or 1.0))
        return pd.DataFrame({"composite_MS3_axis": z[common] @ w.reindex(common),
                             "MS1_signature": z[ms1_genes].mean(axis=1) if ms1_genes else np.nan,
                             "MS3_signature": z[ms3_genes].mean(axis=1) if ms3_genes else np.nan}, index=z.index)
    s_ext = scores(ext.loc[common]).assign(group=grp.reindex(ext.columns).values)
    s_dis = scores(adj.T).assign(group=y.reindex(adj.index).values)
    save_table(s_ext, f"GSE35665_composite_scores{tag}.csv")

    # ---------------------------------------------------------------- (c) multimodal variance
    tests = []
    dip = optional_import("diptest")
    for sc in ["composite_MS3_axis", "MS1_signature", "MS3_signature"]:
        ds, ct = s_ext.loc[s_ext.group == "DS", sc].dropna(), s_ext.loc[s_ext.group == "Control", sc].dropna()
        bf = stats.levene(ds, ct, center="median")
        bic = gmm_bic(ds.to_numpy().reshape(-1, 1), 2)
        row = {"score": sc, "n_DS": len(ds), "n_Control": len(ct), "sd_DS": ds.std(), "sd_Control": ct.std(),
               "variance_ratio_DS_over_Control": ds.var() / ct.var(), "brown_forsythe_p": bf.pvalue,
               "BIC_1comp_DS": bic.get(1, np.nan), "BIC_2comp_DS": bic.get(2, np.nan),
               "deltaBIC_1minus2_DS(>0 favours 2 modes)": bic.get(1, np.nan) - bic.get(2, np.nan),
               "mannwhitney_p_DS_vs_Control(descriptive)": stats.mannwhitneyu(ds, ct)[1]}
        if dip is not None and len(ds) >= 4:
            row["hartigan_dip_p_DS"] = dip.diptest(ds.to_numpy())[1]
        tests.append(row)
    ds2 = s_ext.loc[s_ext.group == "DS", ["MS1_signature", "MS3_signature"]].dropna()
    ct2 = s_ext.loc[s_ext.group == "Control", ["MS1_signature", "MS3_signature"]].dropna()
    bic2d = gmm_bic(ds2.to_numpy(), 3)
    rho_ds = stats.spearmanr(ds2.MS1_signature, ds2.MS3_signature) if len(ds2) > 3 else (np.nan, np.nan)
    rho_ct = stats.spearmanr(ct2.MS1_signature, ct2.MS3_signature) if len(ct2) > 3 else (np.nan, np.nan)
    rho_dis = stats.spearmanr(s_dis.MS1_signature, s_dis.MS3_signature)
    tests.append({"score": "2D(MS1_signature,MS3_signature) DS", "n_DS": len(ds2), **{f"BIC_{k}comp": v for k, v in bic2d.items()},
                  "spearman_MS1arm_vs_MS3arm_DS": rho_ds[0], "spearman_p_DS": rho_ds[1],
                  "spearman_MS1arm_vs_MS3arm_Control": rho_ct[0], "spearman_MS1arm_vs_MS3arm_discovery_T21": rho_dis[0]})
    save_table(pd.DataFrame(tests), f"GSE35665_multimodality_tests{tag}.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    bins = np.linspace(min(s_ext.composite_MS3_axis.min(), s_dis.composite_MS3_axis.min()), max(s_ext.composite_MS3_axis.max(), s_dis.composite_MS3_axis.max()), 22)
    for g, col in [("Control", "#7f7f7f"), ("DS", "#ff7f0e")]:
        ax[0].hist(s_ext.loc[s_ext.group == g, "composite_MS3_axis"], bins=bins, alpha=.6, color=col, label=f"GSE35665 {g}")
    ax[0].set_title("Composite panel score (MS1 <-> MS3 axis)"); ax[0].legend(fontsize=7, loc="best"); ax[0].set_xlabel("weighted composite z-score")
    for g in C.CLASSES:
        ax[1].hist(s_dis.loc[s_dis.group == g, "composite_MS3_axis"], bins=bins, alpha=.55, color=SUBTYPE_COLORS[g], label=f"discovery {g}")
    ax[1].set_title("Discovery T21 (published subtypes)"); ax[1].legend(fontsize=7, loc="best"); ax[1].set_xlabel("weighted composite z-score")
    ax[2].scatter(s_dis.MS1_signature, s_dis.MS3_signature, s=6, c="#dddddd", label="discovery T21")
    for g, col in [("Control", "#7f7f7f"), ("DS", "#ff7f0e")]:
        k = s_ext.group == g
        ax[2].scatter(s_ext.MS1_signature[k], s_ext.MS3_signature[k], s=40, c=col, edgecolor="k", label=f"GSE35665 {g}")
    ax[2].set_xlabel("MS1-arm signature (z)"); ax[2].set_ylabel("MS3-arm signature (z)"); ax[2].legend(fontsize=7)
    ax[2].set_title("Two-arm signature space")
    fig.tight_layout()
    save_fig(f"Fig10b_composite_signature_scores{tag}.png", fig)


if __name__ == "__main__":
    run_stage("04 external validation (GSE35665)", main)
