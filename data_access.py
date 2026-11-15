"""Loaders for the intermediate files written by stage 02 (shared by stages 03-08)."""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

import config as C
from utils import load_json, load_matrix

MANIFEST_PATH = C.PROCESSED / "master_sample_manifest.tsv"
LOGCPM_PATH = C.PROCESSED / "logcpm_all_samples.npz"          # samples x genes (log2(CPM+1), all genes kept)
EXPRESSED_PATH = C.PROCESSED / "expressed_mask_all_samples.npz"  # samples x genes (count >= MIN_COUNT)
ML_GENES_PATH = C.PROCESSED / "ml_gene_list.csv"              # non-HSA21 predictor universe


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST_PATH, sep="\t", index_col=0)


def load_annotation() -> pd.DataFrame:
    """gene_id-indexed annotation of the genes present in the matrices."""
    return pd.read_csv(C.PROCESSED / "gene_annotation_matched.csv", index_col=0)


def load_logcpm() -> pd.DataFrame:
    return load_matrix(LOGCPM_PATH)


def load_expressed() -> pd.DataFrame:
    return load_matrix(EXPRESSED_PATH).astype(bool)


def load_ml_inputs() -> Dict[str, object]:
    """T21-only predictive matrix: X (log-CPM, non-HSA21), expressed mask, covariates, labels, groups."""
    man = load_manifest()
    t21 = man[(man.karyotype == "T21") & man.subtype.isin(C.CLASSES) & (~man.excluded_qc)]
    genes = pd.read_csv(ML_GENES_PATH)["gene_id"].astype(str).tolist()
    logcpm = load_logcpm().loc[t21.index, genes]
    expressed = load_expressed().loc[t21.index, genes]
    cov = t21[C.CONTINUOUS_COVARIATES + C.CATEGORICAL_COVARIATES].copy()
    return dict(X=logcpm, expressed=expressed, cov=cov, y=t21["subtype"].astype(str), groups=t21["participant_id"],
                manifest=t21, genes=genes)


def symbol_lookup(ann: pd.DataFrame) -> pd.Series:
    return ann["gene_symbol"].fillna("").astype(str)
