"""
Stage 02 - metadata harmonisation, QC, published-subtype reproduction, biological baseline (DE).

  A. master sample manifest (GSM <-> LabID <-> karyotype / subtype / age / sex / source) + integrity checks
  B. gene annotation, HSA21 identification
  C. count-matrix QC (library size, detected genes, correlation, PCA) with a >=2-indicator outlier audit
  D. log2-CPM matrices for downstream stages (T21-only ML matrix is built from these)
  E. reproduction of the published MS1-MS3 structure: HSA21 consensus clustering (scipy trapezoid CDF area)
  F. differential expression on RAW counts with ALL 400 samples: the 96 euploid controls are RETAINED for the
     MS1/MS2/MS3-vs-Control contrasts (PyDESeq2; OLS-on-log-CPM fallback if PyDESeq2 is unavailable).
     Only the machine-learning matrix (stage 03) is restricted to the 304 T21 samples.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.integrate import trapezoid                       # NumPy-2.0-safe replacement for np.trapz
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import config as C
import data_access as DA
from hsa21_clusters import _from_supplements, adjust_expression
from utils import (SUBTYPE_COLORS, bh_fdr, find_column, get_logger, heatmap, log_cpm, norm_id, ols_contrasts, open_text,
                   optional_import, parse_soft_samples, read_excel_skip_banner, run_stage, save_fig, save_matrix,
                   save_table, strip_ensembl_version, clean_name)

log = get_logger()


# ============================================================================ A. manifest
def norm_karyotype(v) -> str | float:
    s = str(v).upper()
    if s in {"NAN", "NONE", ""}:
        return np.nan
    if any(t in s for t in ("D21", "EUPLOID", "CONTROL", "NORMAL", "DISOMY")):
        return "Control"
    if any(t in s for t in ("T21", "TRISOMY", "DOWN")) or s.strip() == "DS":
        return "T21"
    return np.nan


def norm_subtype(v) -> str | float:
    """Parse a molecular-subtype cell. Anchored on the WHOLE (stripped) value, not just its ending -- an
    ends-with-digit-1/2/3 pattern also matches huge numbers of ordinary decimal numbers (e.g. a DESeq2
    fold-change value like '0.695703451695381' ends in '1'), which previously caused the value-pattern
    fallback in load_clinical() to lock onto an unrelated numeric column."""
    s = str(v).strip()
    if s.upper() in {"NAN", "NA", ""}:
        return np.nan
    m = re.fullmatch(r"(?:MS\s*)?([123])", s, re.I)
    return f"MS{m.group(1)}" if m else np.nan


def _sniff_separator(path) -> str:
    """Pick the delimiter with the most occurrences on the first data line, rather than assuming tab -- a
    wrong hard-coded separator silently collapses every column into one and is the classic cause of a count
    matrix coming back as (n_genes, 1)."""
    with open_text(path) as fh:
        header = fh.readline()
    counts = {sep: header.count(sep) for sep in ("\t", ",", ";")}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise ValueError(f"No tab/comma/semicolon delimiter found on the first line of {path}. "
                         f"First 200 chars: {header[:200]!r}")
    return best


_PEEK_SIZES = [5_000, 50_000, 500_000, 2_000_000]   # growing peek sizes for classification; a long-format
    # file sorted in per-sample BLOCKS (all of one sample's genes, then the next sample's genes, ...) can
    # show only 1 distinct sample in a small peek, so classification retries with a bigger window rather
    # than assuming a small peek is representative. Even the largest tier here is far smaller than a real
    # tens-of-millions-of-rows file.


def _classify_wide(sample: pd.DataFrame) -> dict | None:
    """gene_id (first column) x many numeric sample columns. Returns {gene_col, sample_cols} or None."""
    gene_col = sample.columns[0]
    data_cols = sample.columns[1:]
    valid_frac = sample[data_cols].apply(pd.to_numeric, errors="coerce").notna().mean()
    keep = valid_frac[valid_frac >= 0.90].index.tolist()
    return {"gene_col": gene_col, "sample_cols": keep} if len(keep) >= 20 else None


def _classify_long(sample: pd.DataFrame) -> dict | None:
    """One row per (gene, sample) pair. Returns {gene_col, sample_col, value_col} or None. See docstring of
    the previous implementation for the detection logic (unchanged); only extracted to work on a small
    sample instead of the whole file, so the expensive full read only happens once we know exactly which
    (and how few) columns are actually needed."""
    parsed_all = sample.apply(pd.to_numeric, errors="coerce")
    numeric_cols = parsed_all.columns[parsed_all.notna().mean() >= 0.90].tolist()
    if len(numeric_cols) != 1:
        return None
    value_col = numeric_cols[0]
    other = [c for c in sample.columns if c != value_col]
    if len(other) < 2:
        return None
    card = {c: sample[c].nunique(dropna=True) for c in other}
    gene_col = max(card, key=card.get)
    id_like = [c for c in other if c != gene_col and re.search(r"sample|gsm|labid|donor|subject|htp", str(c), re.I)]
    remaining = [c for c in other if c != gene_col]
    sample_col = max(id_like, key=lambda c: card[c]) if id_like else \
        (max(remaining, key=lambda c: card[c]) if remaining else None)
    if sample_col is None:
        return None
    # Within a small sample, the TRUE gene count looks artificially close to the sample-row count (each
    # gene/sample pair is likely seen <=once in the first _PEEK_ROWS rows), so this check only requires
    # "clearly more distinct genes than samples", not the stricter full-file ratio.
    if card[gene_col] < 20 or card[sample_col] < 5 or card[gene_col] <= card[sample_col]:
        return None
    return {"gene_col": gene_col, "sample_col": sample_col, "value_col": value_col}


def _read_wide_full(path, sep: str, info: dict) -> pd.DataFrame:
    gene_col, sample_cols = info["gene_col"], info["sample_cols"]
    # Fast path: let pandas parse the sample columns directly as numbers (cheap, compact memory). Falls back
    # to a text read + explicit coercion only if the full file has something the sample didn't show (a
    # stray non-numeric value anywhere outside the first _PEEK_ROWS rows).
    try:
        raw = pd.read_csv(path, sep=sep, compression="infer", usecols=[gene_col] + sample_cols,
                          dtype={gene_col: str, **{c: "float64" for c in sample_cols}},
                          na_values=["NA", "NaN", "null", "None", ""], low_memory=False)
    except (ValueError, TypeError) as exc:
        log.warning("fast numeric read of wide-format counts failed (%r) -> re-reading as text and coercing "
                   "(slower, more memory)", exc)
        raw = pd.read_csv(path, sep=sep, compression="infer", usecols=[gene_col] + sample_cols, dtype=str,
                          low_memory=False)
        raw[sample_cols] = raw[sample_cols].apply(pd.to_numeric, errors="coerce")
    counts = raw[sample_cols].copy()
    counts.index = raw[gene_col].astype(str).map(strip_ensembl_version).values
    counts.columns = [str(c).strip() for c in sample_cols]
    return counts.groupby(level=0).sum(min_count=1).fillna(0)


def _read_long_full(path, sep: str, info: dict, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Two-pass, chunked, memory-bounded pivot of a long/melted (gene, sample, count) table into a wide
    matrix. Never holds more than one `chunksize`-row chunk plus the final (small) output matrix in memory
    at once, regardless of the total row count (tens of millions here) or how the file is sorted (by sample
    block, by gene block, or unsorted) -- unlike reading the whole file at once, which can exceed available
    RAM on a real file of this size.
    Pass 1 reads only the two ID columns to discover the full gene/sample vocabulary (needed to size the
    output matrix and to make pass 2's fill-in correct regardless of row order). Pass 2 reads all three
    needed columns chunk by chunk and scatters each chunk directly into a preallocated dense matrix by
    (gene, sample) coordinate -- an O(1)-per-row operation that needs no sort and no pandas pivot/groupby."""
    gene_col, sample_col, value_col = info["gene_col"], info["sample_col"], info["value_col"]
    log.info("counts file looks like a LONG/melted table, not a wide matrix: pivoting gene column %r x "
             "sample column %r, value column %r via a two-pass chunked read (chunk size %d rows) to bound "
             "memory use regardless of file size", gene_col, sample_col, value_col, chunksize)

    gene_set, sample_set = set(), set()
    n_rows_seen = 0
    for chunk in pd.read_csv(path, sep=sep, compression="infer", usecols=[gene_col, sample_col], dtype=str,
                             chunksize=chunksize):
        gene_set.update(chunk[gene_col].dropna().map(strip_ensembl_version).unique())
        sample_set.update(chunk[sample_col].dropna().str.strip().unique())
        n_rows_seen += len(chunk)
    genes, samples = sorted(gene_set), sorted(sample_set)
    gene_idx = {g: i for i, g in enumerate(genes)}
    sample_idx = {s: i for i, s in enumerate(samples)}
    n_genes, n_samples = len(genes), len(samples)
    log.info("pass 1/2 complete: %d rows scanned -> %d distinct genes x %d distinct samples "
             "(allocating a %.0f MB matrix)", n_rows_seen, n_genes, n_samples, n_genes * n_samples * 8 / 1e6)

    mat = np.zeros((n_genes, n_samples), dtype=np.float64)
    seen = np.zeros((n_genes, n_samples), dtype=bool)
    dupe_count = 0
    for chunk in pd.read_csv(path, sep=sep, compression="infer", usecols=[gene_col, sample_col, value_col],
                             dtype=str, chunksize=chunksize):
        g = chunk[gene_col].map(strip_ensembl_version).map(gene_idx)
        s = chunk[sample_col].str.strip().map(sample_idx)
        v = pd.to_numeric(chunk[value_col], errors="coerce")
        valid = g.notna() & s.notna() & v.notna()
        gi = g[valid].to_numpy(dtype=np.int64)
        si = s[valid].to_numpy(dtype=np.int64)
        vv = v[valid].to_numpy(dtype=np.float64)
        dupe_count += int(seen[gi, si].sum())
        np.add.at(mat, (gi, si), vv)                   # accumulates correctly even for repeated (gi,si) pairs
        seen[gi, si] = True
    if dupe_count:
        log.warning("long-format table had %d duplicate (gene, sample) rows -> summed", dupe_count)
    n_missing = int((~seen).sum())
    if n_missing:
        log.warning("pivoted count matrix has %d missing gene x sample combinations (%.2f%%) -> filled with 0",
                   n_missing, 100 * n_missing / max(seen.size, 1))
    return pd.DataFrame(mat, index=genes, columns=samples)


def load_counts() -> pd.DataFrame:
    sep = _sniff_separator(C.COUNTS_PATH)
    log.info("counts file: sniffed delimiter %r", sep)
    # Classify the file's structure from a small sample first (cheap, bounded memory), THEN do one targeted
    # full read of only the columns that matter, with compact dtypes (categoricals for repeated ID columns
    # rather than raw Python strings). Reading every column of a real tens-of-millions-of-rows table as
    # dtype=str is what previously ran this process out of memory (SIGKILL) on the real GSE190125 file.
    sample, wide_info, long_info = None, None, None
    for n in _PEEK_SIZES:
        sample = pd.read_csv(C.COUNTS_PATH, sep=sep, compression="infer", nrows=n, dtype=str)
        if sample.shape[1] < 2:
            raise ValueError(f"Only {sample.shape[1]} column(s) found in {C.COUNTS_PATH.name} with "
                             f"delimiter {sep!r} -- wrong file or wrong delimiter. "
                             f"Columns: {list(sample.columns)[:10]}")
        wide_info = _classify_wide(sample)
        if wide_info is not None:
            break
        long_info = _classify_long(sample)
        if long_info is not None:
            break
        if len(sample) < n:                          # reached end of file already -- more rows won't help
            break
        log.info("classification inconclusive from the first %d rows (e.g. the file may be sorted in "
                 "per-sample or per-gene blocks, so a small window sees too few distinct samples/genes) -> "
                 "retrying with a larger sample", n)
    if wide_info is None and long_info is None:
        parsed = sample.apply(pd.to_numeric, errors="coerce")
        n_unique = {c: int(sample[c].nunique()) for c in sample.columns}
        raise ValueError(
            f"Could not recognise the structure of {C.COUNTS_PATH.name} as either a wide gene-by-sample "
            f"matrix or a long/melted (gene, sample, count) table, even after trying samples up to "
            f"{len(sample)} rows. Delimiter used: {sep!r}. All columns: {list(sample.columns)}. Per-column "
            f"valid-numeric fraction (in sample): {parsed.notna().mean().round(3).to_dict()}. Per-column "
            f"n-unique (in sample, so cardinality is only a lower bound): {n_unique}. Inspect the file "
            f"directly (zcat '{C.COUNTS_PATH}' | head) and adjust load_counts() in 02_qc_and_baseline.py "
            f"accordingly.")
    if wide_info is not None:
        log.info("counts file classified as WIDE format (gene column %r, %d sample columns) from a %d-row "
                 "sample; reading the full file now", wide_info["gene_col"], len(wide_info["sample_cols"]), len(sample))
        counts = _read_wide_full(C.COUNTS_PATH, sep, wide_info)
        fmt = "wide"
    else:
        counts = _read_long_full(C.COUNTS_PATH, sep, long_info)
        fmt = "long (pivoted)"
    if (counts < 0).any().any():
        raise ValueError("Negative counts found")
    log.info("count matrix (%s format): %d genes x %d samples", fmt, *counts.shape)
    return counts.round().astype(np.int64)


SUBTYPE_NAME_CANDIDATES = ["Molecular subtype", "molecular_subtype", "subtype", "subgroup", "ms group",
                          "ms_group", "ms_subtype", "rna subtype", "rna cluster", "rna_cluster",
                          "transcriptional subtype", "transcriptional cluster", "expression subtype",
                          "expression cluster", "cluster", "classification", "group_rna", "assigned subtype",
                          "ds subtype", "hsa21 subtype"]
LABID_NAME_CANDIDATES = ["LabID", "lab id", "lab_id", "participant id", "participant_id", "sample id",
                         "sample_id", "subject id", "subject_id", "donor id", "donor_id", "htp id", "htp_id",
                         "record id", "record_id", "case id", "case_id", "id"]


def dump_supp2_inventory(sheets: Dict[str, pd.DataFrame]) -> None:
    """Write a fast-to-read inventory of every sheet's columns (name, dtype, n unique, example values) so a
    column-matching failure is a 10-second read instead of another multi-hour pipeline run to find out what
    the real headers are called."""
    lines = []
    for name, df in sheets.items():
        lines.append(f"\n=== sheet: {name!r}  ({df.shape[0]} rows x {df.shape[1]} cols) ===")
        for col in df.columns:
            s = df[col]
            ex = s.dropna().astype(str).str.strip().unique()[:5].tolist()
            lines.append(f"  {col!r:45s} dtype={str(s.dtype):10s} n_unique={s.nunique():5d} examples={ex}")
    path = C.PROCESSED / "supp_data_2_inventory.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Supplementary Data 2 column inventory written -> %s", path)


def _looks_like_subtype_values(s: pd.Series) -> int:
    return int(s.map(norm_subtype).notna().sum())


def _looks_like_id_values(s: pd.Series, n_rows: int) -> float:
    """Heuristic 'this looks like a per-participant identifier' score: high uniqueness, mostly non-numeric
    or alphanumeric-with-prefix, few missing values."""
    s = s.dropna().astype(str).str.strip()
    if len(s) < 0.5 * n_rows:
        return -1.0
    uniqueness = s.nunique() / max(len(s), 1)
    return uniqueness


def detect_subtype_and_id_by_value(sheets: Dict[str, pd.DataFrame]):
    """Fallback when column NAMES don't match anything expected: scan every column of every sheet for one
    whose VALUES look like MS1/MS2/MS3 labels (>=50 parseable, >=2 distinct classes among MS1/MS2/MS3), then
    within that same sheet pick the most identifier-like other column as LabID."""
    best = None  # (score, sheet_name, df, subtype_col, id_col)
    for name, df in sheets.items():
        n = len(df)
        for col in df.columns:
            s = df[col]
            if s.dtype.kind in "fc":                       # pure float/complex columns are never MS1/2/3 labels
                continue
            hits = _looks_like_subtype_values(s)
            if hits < 50:
                continue
            n_classes = s.map(norm_subtype).nunique()
            if n_classes < 2:
                continue
            id_candidates = [c for c in df.columns if c != col]
            if not id_candidates:
                continue
            id_col = max(id_candidates, key=lambda c: _looks_like_id_values(df[c], n))
            if _looks_like_id_values(df[id_col], n) < 0.5:
                continue
            score = hits + 10 * n_classes                  # prefer more labelled rows, then more distinct classes
            if best is None or score > best[0]:
                best = (score, name, df, col, id_col)
    return best


def load_clinical() -> pd.DataFrame:
    sheets = read_excel_skip_banner(C.SUPP2_PATH, sheet_name=None)
    dump_supp2_inventory(sheets)
    O = C.OVERRIDE
    best, best_score = None, -1
    for name, df in sheets.items():
        if O["supp_sheet"] and name != O["supp_sheet"]:
            continue
        sc = find_column(df.columns, SUBTYPE_NAME_CANDIDATES, O["subtype_col"])
        lc = find_column(df.columns, LABID_NAME_CANDIDATES, O["lab_id_col"])
        if sc is None or lc is None:
            continue
        score = df[sc].map(norm_subtype).notna().sum()
        if score > best_score:
            best, best_score = (name, df, sc, lc), score

    detection_method = "column name match"
    if best is None or best_score < 50:
        log.warning("No sheet matched by column NAME (Supplementary Data 2 uses non-standard headers) -> "
                    "falling back to scanning column VALUES for MS1/MS2/MS3-like labels")
        fallback = detect_subtype_and_id_by_value(sheets)
        if fallback is not None:
            _, name, df, sc, lc = fallback
            best, best_score = (name, df, sc, lc), df[sc].map(norm_subtype).notna().sum()
            detection_method = "value-pattern scan"

    if best is None or best_score < 50:
        raise ValueError(
            "Could not find a LabID + molecular-subtype column pair in Supplementary Data 2, by name or by "
            "value pattern. Open processed_data/supp_data_2_inventory.txt (written just before this error) "
            "to see every sheet's real column names and example values, then set "
            "OVERRIDE['supp_sheet'/'lab_id_col'/'subtype_col'] in config.py to the correct ones and re-run "
            "with: python run_pipeline.py --stages 02   (no need to redo the downloads). "
            f"Sheets found: {list(sheets)}")
    name, df, sc, lc = best
    log.info("Supplementary Data 2: using sheet %r, subtype column %r, LabID column %r (%d labelled rows, "
             "detected by %s)", name, sc, lc, best_score, detection_method)
    out = pd.DataFrame({"lab_id": df[lc].astype(str).str.strip(), "subtype": df[sc].map(norm_subtype)})
    for std, cands, key in [("karyotype", ["Karyotype", "trisomy status", "diagnosis", "group", "condition"], "karyotype_col"),
                            ("age", ["Age", "age years", "age_at_visit", "age at visit", "age_yrs"], "age_col"),
                            ("sex", ["Sex", "gender"], "sex_col"),
                            ("source", ["Source", "sample source", "cohort", "site", "collection site"], "source_col"),
                            ("gsm", ["GSM", "geo accession", "gsm_id", "geo_accession"], "gsm_col")]:
        col = find_column(df.columns, cands, O[key])
        if col is not None and col != sc and col != lc:
            out[std] = df[col].values
    return out.drop_duplicates("lab_id").set_index("lab_id", drop=False)


def build_manifest(counts: pd.DataFrame) -> pd.DataFrame:
    soft = parse_soft_samples(C.SOFT_PATH)
    clin = load_clinical()
    lab_norm = {norm_id(k): k for k in clin.index}
    soft_kar = find_column(soft.columns, ["karyotype", "group", "disease_state", "condition"], allow_partial=True)
    soft_sex = find_column(soft.columns, ["sex", "gender"])
    soft_age = find_column(soft.columns, ["age"])
    soft_src = find_column(soft.columns, ["source", "cohort"], allow_partial=False)
    rows = []
    for col in counts.columns:
        rec: Dict[str, object] = {"expression_id": col, "gsm": None, "lab_id": None}
        if col in soft.index:
            rec["gsm"] = col
            for v in soft.loc[col].astype(str).values:
                if norm_id(v) in lab_norm:
                    rec["lab_id"] = lab_norm[norm_id(v)]
                    break
        elif norm_id(col) in lab_norm:
            rec["lab_id"] = lab_norm[norm_id(col)]
            gm = soft.index[[norm_id(col) in norm_id(" ".join(map(str, r))) for r in soft.values]]
            rec["gsm"] = gm[0] if len(gm) else None
        c = clin.loc[rec["lab_id"]] if rec["lab_id"] in clin.index else None
        s = soft.loc[rec["gsm"]] if rec["gsm"] in soft.index else None
        kar = norm_karyotype(c["karyotype"]) if c is not None and "karyotype" in clin else np.nan
        if pd.isna(kar) and s is not None and soft_kar:
            kar = norm_karyotype(s[soft_kar])
        sub = c["subtype"] if c is not None else np.nan
        if pd.isna(kar) and pd.notna(sub):
            kar = "T21"
        pick = lambda k, sk: (c[k] if c is not None and k in clin and pd.notna(c[k]) else
                              (s[sk] if s is not None and sk else np.nan))
        rec.update(karyotype=kar, subtype=sub if kar == "T21" else np.nan, age=pick("age", soft_age),
                   sex=pick("sex", soft_sex), source=pick("source", soft_src))
        rows.append(rec)
    m = pd.DataFrame(rows).set_index("expression_id")
    unmatched = m[m.lab_id.isna() & m.gsm.isna()]
    if len(unmatched):
        log.warning("%d count columns could not be matched to metadata (dropped): %s", len(unmatched),
                    list(unmatched.index[:8]))
    m = m[~(m.lab_id.isna() & m.gsm.isna()) & m.karyotype.notna()].copy()
    m["age"] = pd.to_numeric(m["age"], errors="coerce")
    m["sex"] = m["sex"].astype(str).str.strip().str.upper().str[0].where(m["sex"].notna(), "UNK")
    m["sex"] = m["sex"].replace({"N": "UNK"})
    m["source"] = m["source"].astype(str).str.replace(r"[^A-Za-z0-9]+", "_", regex=True).where(m["source"].notna(), "unknown")
    m["lab_id"] = m["lab_id"].fillna(m.index.to_series())
    m["participant_id"] = m["lab_id"].astype(str).str.extract(r"^(HTP\d+)", expand=False).fillna(m["lab_id"])
    m["group"] = np.where(m.karyotype == "Control", "Control", m.subtype)
    return m


def validate_manifest(m: pd.DataFrame) -> None:
    t21 = m[m.karyotype == "T21"]
    n_ctrl = int((m.karyotype == "Control").sum())
    sub_counts = t21.subtype.value_counts().to_dict()
    log.info("manifest: T21=%d (labelled %d) controls=%d | subtypes=%s", len(t21), t21.subtype.notna().sum(), n_ctrl, sub_counts)
    problems = []
    if len(t21) != C.EXPECTED_T21: problems.append(f"T21 n={len(t21)} (expected {C.EXPECTED_T21})")
    if n_ctrl != C.EXPECTED_CONTROLS: problems.append(f"controls n={n_ctrl} (expected {C.EXPECTED_CONTROLS})")
    for k, v in C.EXPECTED_SUBTYPES.items():
        if sub_counts.get(k, 0) != v: problems.append(f"{k} n={sub_counts.get(k, 0)} (expected {v})")
    if t21.subtype.isna().any(): problems.append(f"{int(t21.subtype.isna().sum())} T21 samples without subtype")
    if m.index.duplicated().any(): problems.append("duplicated expression IDs")
    if m.lab_id.duplicated().any(): problems.append("duplicated LabIDs (grouped CV will be used)")
    if problems:
        msg = "Manifest check: " + "; ".join(problems)
        if C.STRICT_MANIFEST and any("duplicated LabIDs" not in p for p in problems):
            raise RuntimeError(msg + "\nNo model may be run before 304 T21 (107/95/102) + 96 controls are verified. "
                                     "Inspect master_sample_manifest_draft.tsv and the OVERRIDE dict in config.py.")
        log.warning(msg)
    else:
        log.info("manifest verified: 304 T21 (107/95/102) + 96 controls")


def table1_demographics(m: pd.DataFrame) -> None:
    t = m[m.karyotype == "T21"]
    multi_source = m["source"].nunique() > 1                       # a single 'unknown' source carries no information
    rows = []
    for g in ["Control", "MS1", "MS2", "MS3"]:
        d = m[m.group == g]
        row = {"group": g, "n": len(d), "age_median": d.age.median(), "age_IQR": f"{d.age.quantile(.25):.1f}-{d.age.quantile(.75):.1f}",
               "female_%": 100 * (d.sex == "F").mean()}
        if multi_source:
            row.update({f"src_{s}": int((d.source == s).sum()) for s in sorted(m.source.unique())})
        rows.append(row)
    tab = pd.DataFrame(rows)
    tests = [f"age p={stats.kruskal(*[t.loc[t.subtype == s, 'age'].dropna() for s in C.CLASSES]).pvalue:.3g}",
             f"sex p={stats.chi2_contingency(pd.crosstab(t.subtype, t.sex))[1]:.3g}"]
    if multi_source:
        tests.append(f"source p={stats.chi2_contingency(pd.crosstab(t.subtype, t.source))[1]:.3g}")
    tab["test_across_MS1-3"] = ""
    tab.loc[0, "test_across_MS1-3"] = "; ".join(tests) + (" (no source/site information available)" if not multi_source else "")
    save_table(tab, "Table1_sample_characteristics.csv", index=False)


# ============================================================================ B. annotation
def match_annotation(counts: pd.DataFrame) -> pd.DataFrame:
    if not C.GENE_ANNOTATION_CSV.exists():
        raise FileNotFoundError("gene_annotation.csv missing - stage 01 could not fetch GENCODE. Provide "
                                f"{C.GENE_ANNOTATION_CSV} (columns gene_id,gene_symbol,chromosome,gene_type).")
    ann = pd.read_csv(C.GENE_ANNOTATION_CSV, dtype=str)
    by_id = ann.drop_duplicates("gene_id").set_index("gene_id")
    hit = counts.index.isin(by_id.index).mean()
    if hit >= 0.5:
        out = by_id.reindex(counts.index)
        out.index.name = "gene_id"
        out["gene_symbol"] = out["gene_symbol"].fillna("")
    else:                                                     # count matrix keyed by gene symbol?
        by_sym = ann.drop_duplicates("gene_symbol").set_index("gene_symbol")
        hit2 = counts.index.isin(by_sym.index).mean()
        if hit2 < 0.5:
            raise ValueError(f"Annotation matches only {hit:.0%} (Ensembl ID) / {hit2:.0%} (symbol) of count rows")
        out = by_sym.reindex(counts.index)
        out["gene_symbol"] = counts.index
        out["gene_id_ensembl"] = out["gene_id"]
        out.index.name = "gene_id"
    out["chromosome"] = out["chromosome"].fillna("unknown").astype(str).str.replace("chr", "")
    out["gene_type"] = out.get("gene_type", pd.Series("unknown", index=out.index)).fillna("unknown")
    n21 = int((out.chromosome == "21").sum())
    log.info("annotation matched %.1f%% of genes; %d on chr21", 100 * (out.chromosome != "unknown").mean(), n21)
    if n21 < 100 and C.STRICT_MANIFEST:
        raise ValueError(f"Only {n21} chr21 genes detected - fix annotation before modelling.")
    out.to_csv(C.PROCESSED / "gene_annotation_matched.csv")
    return out


# ============================================================================ C. QC
def pca_scores(mat_sxg: np.ndarray, n: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    Xc = mat_sxg - mat_sxg.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :n] * S[:n], (S ** 2 / (S ** 2).sum())[:n]


def qc_and_outliers(counts: pd.DataFrame, lcpm: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    qc = pd.DataFrame({"library_size": counts.sum(axis=0), "detected_genes": (counts > 0).sum(axis=0),
                       "zero_fraction": (counts == 0).mean(axis=0)})
    top = lcpm.var(axis=1).nlargest(min(3000, len(lcpm))).index
    sub = lcpm.loc[top]
    rho = sub.rank(axis=0).corr().to_numpy().copy()
    np.fill_diagonal(rho, np.nan)
    qc["mean_spearman"] = np.nanmean(rho, axis=1)

    def low(v, k=3.0):                                       # Tukey lower fence
        q1, q3 = v.quantile(.25), v.quantile(.75)
        return v < q1 - k * (q3 - q1)
    qc["flag_low_library"] = low(qc.library_size) | (qc.library_size < 0.1 * qc.library_size.median())
    qc["flag_low_detected"] = low(qc.detected_genes)
    qc["flag_low_correlation"] = low(qc.mean_spearman)
    qc["n_flags"] = qc[[c for c in qc if c.startswith("flag_")]].sum(axis=1)
    qc["excluded_qc"] = qc.n_flags >= 2                     # predefined rule: >=2 independent indicators
    qc = qc.join(m[["karyotype", "group", "source", "sex"]])
    save_table(qc, "Table2_QC_and_exclusion_audit.csv")
    log.info("QC: %d samples flagged by >=1 indicator, %d excluded (>=2 indicators)", int((qc.n_flags > 0).sum()),
             int(qc.excluded_qc.sum()))

    pcs, var = pca_scores(sub.T.to_numpy())
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
    ax = ax.ravel()
    ax[0].hist(np.log10(qc.library_size), 40, color="#4c72b0"); ax[0].set_xlabel("log10 library size"); ax[0].set_title("Library size")
    ax[1].scatter(qc.library_size, qc.detected_genes, c=qc.excluded_qc.map({True: "red", False: "#4c72b0"}), s=8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("library size"); ax[1].set_ylabel("detected genes"); ax[1].set_title("Complexity")
    ax[2].hist(qc.mean_spearman, 40, color="#55a868"); ax[2].set_xlabel("mean Spearman to other samples"); ax[2].set_title("Sample correlation")
    for a, col, pal in [(ax[3], "group", SUBTYPE_COLORS), (ax[4], "source", None), (ax[5], "sex", None)]:
        cats = qc[col].astype(str)
        colors = pal or dict(zip(sorted(cats.unique()), plt.cm.tab10.colors))
        for c in sorted(cats.unique()):
            k = (cats == c).to_numpy()
            a.scatter(pcs[k, 0], pcs[k, 1], s=8, label=c, color=colors.get(c))
        a.set_xlabel(f"PC1 ({100 * var[0]:.1f}%)"); a.set_ylabel(f"PC2 ({100 * var[1]:.1f}%)")
        a.set_title(f"PCA by {col}"); a.legend(fontsize=6, markerscale=1.5)
    fig.tight_layout()
    save_fig("Fig2_QC_overview.png", fig)
    return qc


# ============================================================================ E. subtype reproduction
def consensus_clustering(Xz: pd.DataFrame, ks=(2, 3, 4, 5, 6), n_boot: int = 100, frac: float = 0.8, seed: int = 42):
    """ConsensusClusterPlus-style: 1 - Spearman distance, average linkage, item subsampling."""
    rng = np.random.default_rng(seed)
    n = len(Xz)
    ranks = stats.rankdata(Xz.to_numpy(), axis=1)
    dist_full = np.clip(1 - np.corrcoef(ranks), 0, 2)       # Spearman distance of a sample subset = sub-matrix
    np.fill_diagonal(dist_full, 0)
    res = {}
    grid = np.linspace(0, 1, 101)
    for k in ks:
        co, same = np.zeros((n, n)), np.zeros((n, n))
        m = max(k + 2, int(round(frac * n)))
        for _ in range(n_boot):
            idx = np.sort(rng.choice(n, m, replace=False))
            lab = fcluster(linkage(squareform(dist_full[np.ix_(idx, idx)], checks=False), "average"), k, "maxclust")
            co[np.ix_(idx, idx)] += 1
            for l in np.unique(lab):
                mem = idx[lab == l]
                same[np.ix_(mem, mem)] += 1
        cons = np.divide(same, co, out=np.zeros_like(same), where=co > 0)
        np.fill_diagonal(cons, 1)
        Z = linkage(squareform(np.clip(1 - cons, 0, 1), checks=False), "average")
        upper = cons[np.triu_indices(n, 1)]
        cdf = np.array([(upper <= g).mean() for g in grid])
        res[k] = dict(consensus=cons, labels=pd.Series(fcluster(Z, k, "maxclust"), index=Xz.index),
                      order=leaves_list(Z), cdf=cdf, grid=grid, area=float(trapezoid(cdf, grid)))
    return res


def best_mapping(true: pd.Series, clus: pd.Series):
    tab = pd.crosstab(clus, true)
    r, c = linear_sum_assignment(-tab.to_numpy())
    mp = {tab.index[i]: tab.columns[j] for i, j in zip(r, c)}
    mapped = clus.map(mp)
    return mapped, mp, float((mapped == true).mean())


def _published_hsa21_gene_ids(ann: pd.DataFrame, counts_index) -> tuple:
    """Gene IDs of the HSA21 genes the paper actually clustered (Cluster 1 + Cluster 2 lists in Supplementary Data 2,
    97 + 29 = 126). Returns (ids or None, source description)."""
    chr21 = ann[(ann.chromosome == "21") & ann.gene_symbol.notna() & (ann.gene_symbol.astype(str) != "")]
    chr21 = chr21[~chr21.gene_symbol.duplicated()]
    clusters, source = _from_supplements(set(chr21.gene_symbol.astype(str)))
    if clusters is None:
        return None, source
    syms = set(clusters[1]) | set(clusters[2])
    return chr21.index[chr21.gene_symbol.isin(syms)].intersection(counts_index), source


def reproduce_subtypes(lcpm: pd.DataFrame, expr_t21: pd.DataFrame, ann: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    t21 = m[(m.karyotype == "T21") & m.subtype.notna() & ~m.excluded_qc]
    g21 = ann.index[(ann.chromosome == "21") & (ann.gene_type.isin(["protein_coding", "unknown"]))]
    frac = (expr_t21.loc[g21, t21.index] >= C.MIN_COUNT).mean(axis=1)
    g21 = frac.index[frac >= C.MIN_SAMPLE_FRACTION]
    log.info("HSA21 protein-coding genes expressed in >=%d%% of T21 samples: %d (paper: 126)",
             int(100 * C.MIN_SAMPLE_FRACTION), len(g21))
    pub_ids, pub_src = _published_hsa21_gene_ids(ann, expr_t21.index)
    if pub_ids is not None and len(pub_ids) >= 100:
        log.info("consensus clustering uses the %d PUBLISHED HSA21 genes (%s) instead of the %d genes passing my own "
                 "expression filter", len(pub_ids), pub_src, len(g21))
        g21 = pub_ids
    else:
        log.warning("published HSA21 gene list not available -> using the %d genes passing the expression filter", len(g21))
    if len(g21) < 30:
        log.warning("Too few HSA21 genes for consensus clustering - skipping reproduction")
        return pd.DataFrame()
    cov = t21[C.CONTINUOUS_COVARIATES + C.CATEGORICAL_COVARIATES]
    adj = adjust_expression(lcpm.loc[g21, t21.index].T, cov)                      # remove age/sex/source
    Xz = (adj - adj.mean()) / adj.std(ddof=0).replace(0, 1)
    pd.DataFrame({"gene_id": g21, "gene_symbol": ann.loc[g21, "gene_symbol"].values}).to_csv(
        C.PROCESSED / "hsa21_protein_coding_genes.csv", index=False)
    res = consensus_clustering(Xz, n_boot=C.P["consensus_boot"], seed=C.SEED)
    areas = pd.Series({k: v["area"] for k, v in res.items()}, name="cdf_area")
    delta = areas.copy().astype(float)
    delta.iloc[1:] = (areas.values[1:] - areas.values[:-1]) / areas.values[:-1]
    save_table(pd.DataFrame({"cdf_area": areas, "delta_area": delta}), "consensus_cdf_area.csv")

    pub = t21.subtype
    mapped, mp, acc = best_mapping(pub, res[3]["labels"])
    ari = adjusted_rand_score(pub, res[3]["labels"])
    nmi = normalized_mutual_info_score(pub, res[3]["labels"])
    log.info("k=3 reproduction vs published labels: agreement=%.3f ARI=%.3f NMI=%.3f", acc, ari, nmi)
    comp = pd.DataFrame({"published": pub, "reproduced_raw": res[3]["labels"], "reproduced_mapped": mapped,
                         "concordant": mapped == pub})
    save_table(comp, "published_vs_reproduced_subtypes.csv")
    save_table(pd.DataFrame([{"n_hsa21_genes": len(g21), "agreement": acc, "ARI": ari, "NMI": nmi}]),
               "subtype_reproduction_agreement.csv", index=False)

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.3))
    o = res[3]["order"]
    im = ax[0].imshow(res[3]["consensus"][np.ix_(o, o)], cmap="Blues", vmin=0, vmax=1)
    ax[0].set_title(f"Consensus matrix (k=3)\nagreement={acc:.2f}, ARI={ari:.2f}"); plt.colorbar(im, ax=ax[0], fraction=.046)
    for k, v in res.items():
        ax[1].plot(v["grid"], v["cdf"], label=f"k={k}")
    ax[1].set_xlabel("consensus value"); ax[1].set_ylabel("CDF"); ax[1].legend(); ax[1].set_title("Consensus CDF")
    ax[2].bar(delta.index.astype(str), delta.values, color="#4c72b0"); ax[2].set_xlabel("k"); ax[2].set_title("Relative change in CDF area")
    ordr = pub.map({"MS1": 0, "MS2": 1, "MS3": 2}).to_numpy().argsort(kind="stable")
    ax[3].imshow(Xz.iloc[ordr].T, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax[3].set_title("Adjusted HSA21 z-scores\n(samples ordered MS1 | MS2 | MS3)"); ax[3].set_xlabel("participants"); ax[3].set_ylabel("HSA21 genes")
    fig.tight_layout()
    save_fig("Fig3_subtype_reproduction.png", fig)
    return comp


# ============================================================================ F. differential expression
def _run_pydeseq2(counts_sxg: pd.DataFrame, meta: pd.DataFrame, factor: str, contrasts: List[Tuple[str, str]],
                  covars: List[str], ref: str):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    meta = meta.copy()
    levels = [ref] + [l for l in meta[factor].unique() if l != ref]
    meta[factor] = pd.Categorical(meta[factor], categories=levels)
    for c in covars:
        if c != "age":
            meta[c] = meta[c].astype(str)
    n = max(1, C.N_JOBS)
    formula = "~ " + " + ".join(covars + [factor])
    try:
        dds = DeseqDataSet(counts=counts_sxg, metadata=meta, design=formula, refit_cooks=True, n_cpus=n, quiet=True)
    except TypeError:                                           # PyDESeq2 <= 0.4.x API
        dds = DeseqDataSet(counts=counts_sxg, metadata=meta, design_factors=covars + [factor],
                           continuous_factors=["age"], ref_level=[factor, ref], refit_cooks=True, n_cpus=n)
    dds.deseq2()
    out = {}
    for a, b in contrasts:
        st = DeseqStats(dds, contrast=[factor, a, b], n_cpus=n, quiet=True) if _accepts_quiet(DeseqStats) \
            else DeseqStats(dds, contrast=[factor, a, b], n_cpus=n)
        st.summary()
        out[f"{a}_vs_{b}"] = st.results_df.copy()
    return out


def _accepts_quiet(cls) -> bool:
    import inspect
    return "quiet" in inspect.signature(cls.__init__).parameters


def _run_ols_fallback(lcpm_sxg: pd.DataFrame, meta: pd.DataFrame, factor: str, contrasts, covars, ref):
    """Approximation used ONLY if PyDESeq2 is not installed: gene-wise OLS on log2-CPM, same design."""
    D = pd.get_dummies(meta[[factor]].astype(str), dtype=float)
    lv = sorted(meta[factor].unique(), key=lambda x: (x != ref, x))
    D = D.reindex(columns=[f"{factor}_{l}" for l in lv])
    parts = [pd.Series(1.0, index=meta.index, name="intercept"), D.drop(columns=f"{factor}_{ref}")]
    for c in covars:
        parts.append(meta[[c]].astype(float) if c == "age" else pd.get_dummies(meta[c].astype(str), prefix=c, drop_first=True, dtype=float))
    X = pd.concat(parts, axis=1)
    cons = {}
    for a, b in contrasts:
        v = np.zeros(X.shape[1])
        if a != ref: v[list(X.columns).index(f"{factor}_{a}")] += 1
        if b != ref: v[list(X.columns).index(f"{factor}_{b}")] -= 1
        cons[f"{a}_vs_{b}"] = v
    res = ols_contrasts(lcpm_sxg.to_numpy(float), X.to_numpy(float), cons)
    for k in res:
        res[k].index = lcpm_sxg.columns
    return res


def differential_expression(counts: pd.DataFrame, lcpm: pd.DataFrame, ann: pd.DataFrame, m: pd.DataFrame) -> None:
    if os.environ.get("DS_SKIP_DE") == "1":
        log.info("DS_SKIP_DE=1 -> skipping differential expression")
        return
    # ALL 400 samples stay in: 96 controls are the biological baseline (not filtered out).
    keep_s = m.index[~m.excluded_qc]
    t21_ids = m.index[(m.karyotype == "T21") & ~m.excluded_qc]
    frac = (counts.loc[:, t21_ids] >= C.MIN_COUNT).mean(axis=1)
    genes = frac.index[frac >= C.MIN_SAMPLE_FRACTION]
    log.info("DE: %d samples (%d controls retained) x %d genes", len(keep_s), int((m.loc[keep_s, 'karyotype'] == 'Control').sum()), len(genes))
    meta = m.loc[keep_s, ["group", "karyotype", "age", "sex", "source"]].copy()
    meta["age"] = meta["age"].fillna(meta["age"].median())
    meta["sex"] = meta["sex"].astype(str)
    covars = [c for c in ["source", "sex", "age"] if meta[c].nunique() > 1]
    use_pydeseq = optional_import("pydeseq2") is not None
    counts_sxg = counts.loc[genes, keep_s].T
    contrasts = [("MS1", "Control"), ("MS2", "Control"), ("MS3", "Control"),
                 ("MS1", "MS2"), ("MS1", "MS3"), ("MS2", "MS3")]
    method = "PyDESeq2" if use_pydeseq else "OLS_logCPM_fallback (PyDESeq2 not installed)"
    log.info("DE method: %s", method)
    try:
        if use_pydeseq:
            res = _run_pydeseq2(counts_sxg, meta, "group", contrasts, covars, "Control")
            res.update(_run_pydeseq2(counts_sxg, meta.assign(T21=meta.karyotype), "T21", [("T21", "Control")], covars, "Control"))
        else:
            res = _run_ols_fallback(lcpm.loc[genes, keep_s].T, meta, "group", contrasts, covars, "Control")
            res.update(_run_ols_fallback(lcpm.loc[genes, keep_s].T, meta, "karyotype", [("T21", "Control")], covars, "Control"))
    except Exception as exc:  # noqa: BLE001
        if not use_pydeseq:
            raise
        log.error("PyDESeq2 failed (%r) -> falling back to OLS on log-CPM", exc)
        method = "OLS_logCPM_fallback (PyDESeq2 failed)"
        res = _run_ols_fallback(lcpm.loc[genes, keep_s].T, meta, "group", contrasts, covars, "Control")
        res.update(_run_ols_fallback(lcpm.loc[genes, keep_s].T, meta, "karyotype", [("T21", "Control")], covars, "Control"))

    summ = []
    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    ax = ax.ravel()
    for i, (name, r) in enumerate(res.items()):
        r = r.copy()
        r.index.name = "gene_id"
        r["gene_symbol"] = ann.reindex(r.index)["gene_symbol"].values
        r["chromosome"] = ann.reindex(r.index)["chromosome"].values
        r["method"] = method
        r["significant"] = (r.padj < C.FDR_THRESHOLD) & (r.log2FoldChange.abs() >= C.ABS_LFC_THRESHOLD)
        save_table(r.sort_values("padj"), f"DE_{name}.csv")
        summ.append({"contrast": name, "method": method, "n_tested": int(r.padj.notna().sum()),
                     "sig_FDR05_lfc0.5": int(r.significant.sum()), "up": int((r.significant & (r.log2FoldChange > 0)).sum()),
                     "down": int((r.significant & (r.log2FoldChange < 0)).sum()),
                     "sig_FDR10": int((r.padj < C.FDR_SENSITIVITY).sum()),
                     "sig_HSA21_FDR05": int((r.significant & (r.chromosome == "21")).sum())})
        if i < len(ax):
            a = ax[i]; h21 = (r.chromosome == "21").to_numpy()
            y = -np.log10(r.padj.clip(lower=1e-300))
            a.scatter(r.log2FoldChange[~h21], y[~h21], s=3, c=np.where(r.significant[~h21], "#d62728", "#bbbbbb"))
            a.scatter(r.log2FoldChange[h21], y[h21], s=8, c="#ff7f0e", label="HSA21")
            a.axhline(-np.log10(C.FDR_THRESHOLD), ls=":", c="k", lw=.7); a.set_title(name); a.set_xlabel("log2 FC"); a.set_ylabel("-log10 padj")
            if i == 0: a.legend(fontsize=7)
    for unused in ax[len(res):]:
        unused.axis("off")                      # 7 contrasts in a 2 x 4 grid: hide the empty panel
    fig.tight_layout()
    save_fig("Fig4_differential_expression_volcanos.png", fig)
    save_table(pd.DataFrame(summ), "DE_summary.csv", index=False)


# ============================================================================ main
def main() -> None:
    counts = load_counts()
    man = build_manifest(counts)
    man.to_csv(C.PROCESSED / "master_sample_manifest_draft.tsv", sep="\t")
    validate_manifest(man)
    counts = counts[man.index]                                          # drops unmatched columns
    ann = match_annotation(counts)
    table1_demographics(man)

    lcpm_gxs = log_cpm(counts, prior=1.0)                               # library size over ALL genes
    qc = qc_and_outliers(counts, lcpm_gxs, man)
    man["excluded_qc"] = qc["excluded_qc"].reindex(man.index).fillna(False).astype(bool)
    if man.excluded_qc.any():
        log.warning("QC-excluded samples: %s", list(man.index[man.excluded_qc]))
    man.to_csv(DA.MANIFEST_PATH, sep="\t")

    # matrices for downstream stages ---------------------------------------------------------------
    is21 = (ann.chromosome == "21").to_numpy()
    t21_mask = ((man.karyotype == "T21") & (~man.excluded_qc)).to_numpy()
    detected = (counts.loc[:, t21_mask] >= C.MIN_COUNT).mean(axis=1)
    detected_any = (counts > 0).mean(axis=1)
    keep_genes = counts.index[(detected >= 0.10) | (detected_any >= 0.10) | is21]
    save_matrix(lcpm_gxs.loc[keep_genes].T, DA.LOGCPM_PATH)
    save_matrix((counts.loc[keep_genes] >= C.MIN_COUNT).T.astype(np.uint8), DA.EXPRESSED_PATH)
    predictors = ann.index[(~is21) & (ann.chromosome != "unknown")].intersection(keep_genes)
    pd.DataFrame({"gene_id": predictors, "gene_symbol": ann.loc[predictors, "gene_symbol"].values,
                  "chromosome": ann.loc[predictors, "chromosome"].values}).to_csv(DA.ML_GENES_PATH, index=False)
    log.info("ML predictor universe: %d non-HSA21 genes with a mapped chromosome (%d HSA21 genes excluded)",
             len(predictors), int(is21.sum()))
    assert not ann.loc[predictors, "chromosome"].eq("21").any(), "chromosome 21 gene leaked into predictors"

    reproduce_subtypes(lcpm_gxs, counts, ann, man)
    differential_expression(counts, lcpm_gxs, ann, man)


if __name__ == "__main__":
    run_stage("02 QC, subtype reproduction, differential expression", main)
