"""Immune-cell composition scores (MCP-counter / xCell style).

Priority: (1) user-supplied real MCP-counter/xCell output  data_raw/immune_scores.csv (samples x cell types,
produced e.g. with the R packages MCPcounter / xCell), (2) a Python marker-gene proxy: per-sample mean log2-CPM of
canonical marker genes (sample-wise, so no information flows between samples). The proxy is an APPROXIMATION of
MCP-counter and is flagged as such in every output.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

import config as C
from utils import get_logger

log = get_logger()
IMMUNE_CSV = C.DATA_RAW / "immune_scores.csv"

MARKERS: Dict[str, List[str]] = {
    "Neutrophils": ["FCGR3B", "CXCR2", "CSF3R", "CXCR1", "FFAR2", "MNDA", "PROK2", "CA4"],
    "T_cells": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC", "IL7R", "LEF1", "TCF7"],
    "B_lineage": ["CD19", "MS4A1", "CD79A", "CD79B", "BANK1", "CD22", "FCRLA", "VPREB3"],
    "NK_cells": ["NCAM1", "KLRD1", "KLRF1", "NKG7", "GNLY", "PRF1", "KLRB1", "GZMB"],
    "Monocytic_lineage": ["CD14", "FCN1", "VCAN", "CSF1R", "MS4A6A", "CTSS", "LYZ", "CD68"],
    "Myeloid_dendritic": ["CD1C", "CLEC10A", "FCER1A", "CLEC4C", "IL3RA", "LILRA4", "CD1E", "CLEC9A"],
}


def immune_scores(logcpm: pd.DataFrame, ann: pd.DataFrame, samples) -> Tuple[pd.DataFrame, str]:
    """Return (samples x cell types, source description). `logcpm` is samples x gene_id."""
    if IMMUNE_CSV.exists():
        d = pd.read_csv(IMMUNE_CSV, index_col=0)
        if d.index.isin(samples).mean() < 0.5 and d.columns.isin(samples).mean() >= 0.5:
            d = d.T
        d = d.reindex(samples)
        if d.notna().all(axis=None):
            return d.astype(float), f"user-supplied ({IMMUNE_CSV.name})"
        log.warning("immune_scores.csv does not cover all samples -> using marker proxy")
    sym2id = ann.reset_index().drop_duplicates("gene_symbol").set_index("gene_symbol")["gene_id"]
    out, used = {}, {}
    for cell, genes in MARKERS.items():
        ids = [sym2id[g] for g in genes if g in sym2id.index and sym2id[g] in logcpm.columns]
        used[cell] = len(ids)
        if len(ids) >= 3:
            out[cell] = logcpm.loc[samples, ids].mean(axis=1)
    log.info("marker-proxy immune scores; markers found per cell type: %s", used)
    if len(out) < 3:
        raise RuntimeError("Too few immune marker genes found; supply data_raw/immune_scores.csv (MCP-counter/xCell).")
    return pd.DataFrame(out), "MARKER-GENE PROXY for MCP-counter (mean log2-CPM of canonical markers)"
