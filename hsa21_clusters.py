"""
HSA21 gene Cluster 1 (97 genes) / Cluster 2 (29 genes) from Donovan et al. 2024 (Nat Commun 15:5473)
and exact polygenic z-scores per participant.

Cluster membership is read, in order of preference, from
  1. data_raw/hsa21_clusters.csv           (user override: columns gene_symbol, cluster)
  2. Supplementary Data 2 (any sheet with a cluster column or 'Cluster 1'/'Cluster 2' columns)
  3. Source_Data.zip (same scan on every .xlsx/.csv inside)
  4. FALLBACK ONLY: data-driven split of the HSA21 genes into 2 co-expression clusters anchored on the
     example genes named in the paper text. This is clearly flagged in the outputs
     (cluster_source != 'published') and must not be presented as the published lists.
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import config as C
from ml_components import CovariateResidualizer
from utils import get_logger, read_excel_skip_banner

log = get_logger()

# Example genes stated explicitly in the paper text (Fig. 1e-g, Fig. 2h, Fig. 3)
ANCHORS = {1: ["PAXBP1", "GABPA", "BRWD1", "N6AMT1", "ATP5PF"],
           2: ["IFNAR1", "IFNGR2", "IL10RB", "KCNJ15", "SLC19A1", "ITGB2"]}
EXPECTED_SIZES = {1: 97, 2: 29}


def _cluster_number(v) -> int | None:
    m = re.search(r"([12])\s*$", str(v).strip()) or re.search(r"cluster\D*([12])", str(v), re.I)
    return int(m.group(1)) if m else None


def _scan_frame(df: pd.DataFrame, universe: set) -> Dict[int, List[str]] | None:
    """Try to extract {1: [...], 2: [...]} from one table."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    # (a) long format: a gene column + a cluster column
    for ccol in [c for c in df.columns if "cluster" in c.lower()]:
        labels = df[ccol].map(_cluster_number)
        if labels.notna().sum() < 10:
            continue
        for gcol in df.columns:
            vals = df[gcol].astype(str).str.strip()
            if vals.isin(universe).mean() >= 0.5 and gcol != ccol:
                out = {k: sorted(set(vals[labels == k]) & universe) for k in (1, 2)}
                if all(len(v) >= 5 for v in out.values()):
                    return out
        if df.index.astype(str).isin(universe).mean() >= 0.5:      # genes in the index
            out = {k: sorted(set(df.index.astype(str)[labels.values == k]) & universe) for k in (1, 2)}
            if all(len(v) >= 5 for v in out.values()):
                return out
    # (b) wide format: columns literally named like 'Cluster 1' / 'HSA21 cluster 2 genes'
    wide = {}
    for c in df.columns:
        m = re.search(r"cluster\D*([12])\b", c, re.I)
        if m:
            vals = set(df[c].dropna().astype(str).str.strip()) & universe
            if len(vals) >= 5:
                wide[int(m.group(1))] = sorted(vals)
    return wide if len(wide) == 2 else None


def _from_supplements(universe: set) -> Tuple[Dict[int, List[str]] | None, str]:
    if C.HSA21_CLUSTER_OVERRIDE.exists():
        d = pd.read_csv(C.HSA21_CLUSTER_OVERRIDE)
        gcol = d.columns[0]
        ccol = [c for c in d.columns if "cluster" in c.lower()][0]
        out = {k: sorted(set(d.loc[d[ccol].map(_cluster_number) == k, gcol].astype(str))) for k in (1, 2)}
        return out, f"published (override file {C.HSA21_CLUSTER_OVERRIDE.name})"
    if C.SUPP2_PATH.exists():
        try:
            for sheet, df in read_excel_skip_banner(C.SUPP2_PATH, sheet_name=None).items():
                out = _scan_frame(df, universe)
                if out:
                    return out, f"published (Supplementary Data 2, sheet '{sheet}')"
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not scan Supplementary Data 2 for clusters: %r", exc)
    if C.SOURCE_DATA_ZIP.exists():
        try:
            with zipfile.ZipFile(C.SOURCE_DATA_ZIP) as zf:
                for name in zf.namelist():
                    frames = {}
                    if name.lower().endswith(".xlsx"):
                        frames = pd.read_excel(io.BytesIO(zf.read(name)), sheet_name=None)
                    elif name.lower().endswith((".csv", ".tsv")):
                        frames = {"": pd.read_csv(io.BytesIO(zf.read(name)), sep=None, engine="python")}
                    for sheet, df in frames.items():
                        out = _scan_frame(df, universe)
                        if out:
                            return out, f"published (Source_Data.zip:{name}:{sheet})"
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not scan Source_Data.zip for clusters: %r", exc)
    return None, ""


def _derive(adjusted_z: pd.DataFrame) -> Dict[int, List[str]]:
    """Fallback: 2-way average-linkage clustering of HSA21 genes on 1 - Spearman rho."""
    rho = adjusted_z.rank(axis=0).corr(method="pearson").to_numpy()
    dist = np.clip(1 - rho, 0, 2)
    np.fill_diagonal(dist, 0)
    lab = fcluster(linkage(squareform(dist, checks=False), "average"), 2, "maxclust")
    genes = np.asarray(adjusted_z.columns)
    a1 = sum(lab[list(genes).index(g)] == 1 for g in ANCHORS[1] if g in genes)
    a2 = sum(lab[list(genes).index(g)] == 2 for g in ANCHORS[1] if g in genes)
    c1_label = 1 if a1 >= a2 else 2
    return {1: sorted(genes[lab == c1_label]), 2: sorted(genes[lab != c1_label])}


def adjust_expression(logcpm_sxg: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """Age/sex/source-adjusted expression (descriptive analyses on the 304 T21 samples)."""
    cols = list(logcpm_sxg.columns)
    frame = pd.concat([logcpm_sxg, cov.loc[logcpm_sxg.index]], axis=1)
    res = CovariateResidualizer(cols, list(cov.columns)).fit(frame)
    return pd.DataFrame(res.transform(frame), index=logcpm_sxg.index, columns=cols)


def load_clusters(hsa21_adjusted_by_symbol: pd.DataFrame) -> Tuple[Dict[int, List[str]], str]:
    universe = set(hsa21_adjusted_by_symbol.columns)
    clusters, source = _from_supplements(universe)
    if clusters is None:
        z = (hsa21_adjusted_by_symbol - hsa21_adjusted_by_symbol.mean()) / hsa21_adjusted_by_symbol.std(ddof=0).replace(0, 1)
        clusters, source = _derive(z), "DERIVED fallback (data-driven, anchored on paper example genes)"
        log.warning("Published HSA21 cluster lists not found -> using %s", source)
    sizes = {k: len(v) for k, v in clusters.items()}
    log.info("HSA21 clusters: %s (expected %s) | source: %s", sizes, EXPECTED_SIZES, source)
    if sizes != EXPECTED_SIZES:
        log.warning("Cluster sizes differ from the published 97/29 (genes not expressed/annotated are dropped).")
    return clusters, source


def polygenic_zscores(adjusted_by_symbol: pd.DataFrame, clusters: Dict[int, List[str]]) -> pd.DataFrame:
    """Per-participant polygenic z-score for each cluster.

    1. every gene is z-scored across participants (adjusted log-CPM),
    2. cluster mean-z = mean of member-gene z-scores  (`cluster{k}_mean_z`),
    3. `cluster{k}_z` re-standardises the mean-z across participants (mean 0, SD 1).
    """
    z = (adjusted_by_symbol - adjusted_by_symbol.mean()) / adjusted_by_symbol.std(ddof=0).replace(0, 1)
    out = pd.DataFrame(index=adjusted_by_symbol.index)
    for k, genes in clusters.items():
        g = [x for x in genes if x in z.columns]
        m = z[g].mean(axis=1)
        out[f"cluster{k}_mean_z"] = m
        out[f"cluster{k}_z"] = (m - m.mean()) / m.std(ddof=0)
        out[f"cluster{k}_n_genes"] = len(g)
    return out
