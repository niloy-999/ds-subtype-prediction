"""
Stage 06 - (A) concordance of this pipeline's differential expression with the PUBLISHED DESeq2 results (Supplementary
Data 2, sheet c), and (B) per-subtype expression profile of the final non-HSA21 panel genes relative to euploid controls.

(A) is a reproducibility check on the same participants and labels (PyDESeq2 here vs DESeq2 in the original study), not an
    independent-cohort validation.
(B) describes what the panel genes do in each subtype (log2 fold-change vs controls, from the stage-02 DE tables, adjusted
    for age and sex). It is the recommended way to describe gene-level direction: per-class SHAP / softmax coefficients are
    only identified relative to the other classes and should not be read one class at a time.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config as C
from utils import (find_column, get_logger, heatmap, load_json, read_excel_skip_banner, run_stage, save_fig, save_table)

log = get_logger()
# (our contrast file, published column prefix, sign to apply to the published log2FC so that directions match ours)
PAIRS = [("MS1_vs_Control", "MS1 vs D21", 1), ("MS2_vs_Control", "MS2 vs D21", 1), ("MS3_vs_Control", "MS3 vs D21", 1),
         ("MS1_vs_MS2", "MS2 vs MS1", -1), ("MS1_vs_MS3", "MS3 vs MS1", -1), ("MS2_vs_MS3", "MS3 vs MS2", -1)]


def load_de(name: str) -> pd.DataFrame | None:
    path = C.TABLES / f"DE_{name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path).drop_duplicates("gene_id").set_index("gene_id")


def published_de() -> pd.DataFrame | None:
    if not C.SUPP2_PATH.exists():
        return None
    for name, df in read_excel_skip_banner(C.SUPP2_PATH, sheet_name=None).items():
        gcol = find_column(df.columns, ["EnsemblID", "ensembl id", "gene_id"], allow_partial=False)
        if gcol is not None and any("l2FC" in str(c) for c in df.columns):
            df = df.copy()
            df["gid"] = df[gcol].astype(str).str.replace(r"\.\d+$", "", regex=True)
            for c in df.columns:
                if "l2FC" in str(c) or "padj" in str(c):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            log.info("published DESeq2 table found in sheet %r (%d genes)", name, len(df))
            return df.drop_duplicates("gid").set_index("gid")
    return None


def concordance() -> None:
    pub = published_de()
    if pub is None:
        log.warning("published DESeq2 table (Supplementary Data 2) not found -> concordance analysis skipped")
        return
    rows, fig_data = [], []
    for ours, theirs, sgn in PAIRS:
        d = load_de(ours)
        if d is None or f"{theirs} l2FC" not in pub.columns:
            continue
        j = d[["log2FoldChange", "padj", "chromosome"]].join(pub[[f"{theirs} l2FC", f"{theirs} padj"]], how="inner")
        j = j.dropna(subset=["log2FoldChange", f"{theirs} l2FC"])
        x, y = j["log2FoldChange"], sgn * j[f"{theirs} l2FC"]
        sig_o, sig_p = j.padj < C.FDR_THRESHOLD, j[f"{theirs} padj"] < C.FDR_THRESHOLD
        both = sig_o & sig_p
        h = j.chromosome.astype(str) == "21"
        rows.append({"contrast": ours, "published_contrast": theirs, "n_genes_compared": len(j),
                     "spearman_log2FC": stats.spearmanr(x, y)[0], "pearson_log2FC": stats.pearsonr(x, y)[0],
                     "n_sig_ours_FDR05": int(sig_o.sum()), "n_sig_published_FDR05": int(sig_p.sum()), "n_sig_both": int(both.sum()),
                     "jaccard_sig": both.sum() / max((sig_o | sig_p).sum(), 1),
                     "sign_agreement_among_sig_in_both": float((np.sign(x[both]) == np.sign(y[both])).mean()) if both.any() else np.nan,
                     "n_hsa21_genes": int(h.sum()), "spearman_log2FC_hsa21": stats.spearmanr(x[h], y[h])[0] if h.sum() > 3 else np.nan})
        fig_data.append((ours, x, y, h))
    if not rows:
        log.warning("no comparable contrasts found")
        return
    save_table(pd.DataFrame(rows), "Table10_DE_concordance_with_published.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (name, x, y, h) in zip(axes.ravel(), fig_data):
        ax.scatter(x[~h], y[~h], s=2, alpha=.25, color="#888888")
        ax.scatter(x[h], y[h], s=8, color="#ff7f0e", label="HSA21")
        lim = float(np.nanpercentile(np.abs(np.r_[x, y]), 99.8)) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k:", lw=.8); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"{name}\nSpearman = {stats.spearmanr(x, y)[0]:.3f}", fontsize=9)
        ax.set_xlabel("log2FC, this pipeline (PyDESeq2)"); ax.set_ylabel("log2FC, published (DESeq2)")
    axes.ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    save_fig("Fig11_DE_concordance_with_published.png", fig)


def panel_profile() -> None:
    meta_path = C.MODELS / "final_model_metadata.json"
    if not meta_path.exists():
        log.warning("final_model_metadata.json missing -> panel profile skipped")
        return
    panel = load_json(meta_path)["panel"]
    cols = {}
    for c in ["MS1_vs_Control", "MS2_vs_Control", "MS3_vs_Control", "MS1_vs_MS3", "MS2_vs_MS3", "MS1_vs_MS2"]:
        d = load_de(c)
        if d is None:
            log.warning("DE table for %s missing -> panel profile skipped", c)
            return
        cols[f"log2FC_{c}"] = d["log2FoldChange"]
        cols[f"padj_{c}"] = d["padj"]
    ann = pd.read_csv(C.PROCESSED / "gene_annotation_matched.csv", index_col=0)
    t = pd.DataFrame({"gene_id": panel}).set_index("gene_id").join(ann[["gene_symbol", "chromosome", "gene_type"]]).join(pd.DataFrame(cols))
    l1, l2, l3 = (t[f"log2FC_MS{k}_vs_Control"] for k in (1, 2, 3))
    t["higher_in"] = np.where(t.log2FC_MS1_vs_MS3 > 0, "MS1 (vs MS3)", "MS3 (vs MS1)")
    t["pattern_vs_control"] = np.select(
        [(l1 > l2) & (l2 > l3), (l1 < l2) & (l2 < l3), (l2 < l1) & (l3 < l1) & (l2 < 0) & (l3 < 0), (l2 > l1) & (l3 > l1) & (l2 > 0) & (l3 > 0)],
        ["MS1 > MS2 > MS3 (monotone)", "MS1 < MS2 < MS3 (monotone)", "MS2 and MS3 below both MS1 and control", "MS2 and MS3 above both MS1 and control"], "other")
    save_table(t.reset_index(), "Table11_panel_gene_subtype_profile.csv", index=False)
    order = t.sort_values("log2FC_MS1_vs_MS3")
    mat = order[["log2FC_MS1_vs_Control", "log2FC_MS2_vs_Control", "log2FC_MS3_vs_Control"]].to_numpy()
    fig, ax = plt.subplots(figsize=(5.8, 0.38 * len(order) + 2))
    lim = max(0.5, float(np.nanmax(np.abs(mat))))
    heatmap(ax, mat, ["MS1", "MS2", "MS3"], [f"{s} (chr{c})" for s, c in zip(order.gene_symbol, order.chromosome)], vmin=-lim, vmax=lim,
            cbar_label="log2FC vs euploid controls", annot=True, fmt="{:+.2f}")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Final panel genes: expression change vs controls by subtype\n(PyDESeq2, age/sex adjusted; sorted by MS1-vs-MS3 difference)", fontsize=8)
    fig.tight_layout()
    save_fig("Fig12_panel_gene_subtype_profile.png", fig)
    log.info("panel genes by pattern: %s", t.pattern_vs_control.value_counts().to_dict())


def main() -> None:
    concordance()
    panel_profile()


if __name__ == "__main__":
    run_stage("06 DE concordance with published results, panel subtype profile", main)
