"""
Central configuration for the DS molecular-subtype pipeline.

Run modes (environment variable DS_MODE, set by run_pipeline.py --mode):
  full : publication settings (5x10 repeated outer CV, 5-fold inner CV, full grids)
  fast : same logic, reduced repeats/grids for a quick real-data test run
  demo : SYNTHETIC data written by demo_data.py -- for smoke-testing the code only.
         Demo output has no biological meaning and lives in ./demo_workspace.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODE = os.environ.get("DS_MODE", "full").lower()
if MODE not in {"full", "fast", "demo"}:
    MODE = "full"
DEMO = MODE == "demo"
WORK = ROOT / "demo_workspace" if DEMO else ROOT

# ----------------------------------------------------------------- directories
DATA_RAW = WORK / "data_raw"
DATA_EXTERNAL = DATA_RAW / "GSE35665"
PROCESSED = WORK / "processed_data"
RESULTS = WORK / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"
LOGS = RESULTS / "logs"
ALL_DIRS = [DATA_RAW, DATA_EXTERNAL, PROCESSED, RESULTS, TABLES, FIGURES, MODELS, LOGS]

# ------------------------------------------------------------------ input files
COUNTS_PATH = DATA_RAW / "GSE190125_Counts_for_GEO.txt.gz"
FPKM_PATH = DATA_RAW / "GSE190125_FPKMs_for_GEO.txt.gz"          # optional QC cross-check
SOFT_PATH = DATA_RAW / "GSE190125_family.soft.gz"
SUPP2_PATH = DATA_RAW / "Supplementary_Data_2.xlsx"
SOURCE_DATA_ZIP = DATA_RAW / "Source_Data.zip"                     # optional
GENCODE_GTF = DATA_RAW / "gencode.v33.annotation.gtf.gz"
GENE_ANNOTATION_CSV = PROCESSED / "gene_annotation.csv"
HSA21_CLUSTER_OVERRIDE = DATA_RAW / "hsa21_clusters.csv"           # optional: gene_symbol,cluster (1|2)

GSE35665_MATRIX = DATA_EXTERNAL / "GSE35665_series_matrix.txt.gz"
GSE35665_PLATFORM = DATA_EXTERNAL / "platform_annotation.txt.gz"

# Each entry: (candidate URLs tried IN ORDER, destination, mandatory). The plain FTP-over-HTTPS mirror is
# listed FIRST: it serves the real bytes directly and supports Range-resume, which is what the resumable
# downloader needs for large files on a flaky connection. NCBI's "/geo/download/?acc=...&format=file" endpoint
# is listed as a fallback only -- for files above a size threshold it can return an HTML "confirm large
# download" interstitial instead of the file, which download_file() now detects (magic-byte / Content-Type
# check) and rejects rather than silently saving as if it were the real file.
DOWNLOADS = {
    "GSE190125 counts": (
        ["https://ftp.ncbi.nlm.nih.gov/geo/series/GSE190nnn/GSE190125/suppl/GSE190125_Counts_for_GEO.txt.gz",
         "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE190125&format=file&file=GSE190125_Counts_for_GEO.txt.gz"],
        COUNTS_PATH, True),
    "GSE190125 SOFT": (
        ["https://ftp.ncbi.nlm.nih.gov/geo/series/GSE190nnn/GSE190125/soft/GSE190125_family.soft.gz"],
        SOFT_PATH, True),
    "Supplementary Data 2": (
        ["https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-024-49781-1/"
         "MediaObjects/41467_2024_49781_MOESM5_ESM.xlsx"],
        SUPP2_PATH, True),
    "GSE190125 FPKM": (
        ["https://ftp.ncbi.nlm.nih.gov/geo/series/GSE190nnn/GSE190125/suppl/GSE190125_FPKMs_for_GEO.txt.gz",
         "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE190125&format=file&file=GSE190125_FPKMs_for_GEO.txt.gz"],
        FPKM_PATH, False),
    "Source Data": (
        ["https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-024-49781-1/"
         "MediaObjects/41467_2024_49781_MOESM13_ESM.zip"],
        SOURCE_DATA_ZIP, False),
    "GENCODE v33 annotation": (
        ["https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_33/gencode.v33.annotation.gtf.gz"],
        GENCODE_GTF, False),
    "GSE35665 series matrix": (
        ["https://ftp.ncbi.nlm.nih.gov/geo/series/GSE35nnn/GSE35665/matrix/GSE35665_series_matrix.txt.gz",
         "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE35665&format=file&file=GSE35665_series_matrix.txt.gz"],
        GSE35665_MATRIX, True),
}

# --------------------------------------------------------------- study design
SEED = 42
N_JOBS = int(os.environ.get("DS_N_JOBS", max(1, (os.cpu_count() or 2) - 1)))
CLASSES = ["MS1", "MS2", "MS3"]
EXPECTED_T21 = 304
EXPECTED_CONTROLS = 96
EXPECTED_SUBTYPES = {"MS1": 107, "MS2": 95, "MS3": 102}
STRICT_MANIFEST = True           # abort if manifest counts differ from the published ones

MIN_COUNT = 10                   # low-expression filter: count >= 10 ...
MIN_SAMPLE_FRACTION = 0.20       # ... in >= 20 % of T21 samples
FDR_THRESHOLD = 0.05
ABS_LFC_THRESHOLD = 0.5
FDR_SENSITIVITY = 0.10

CATEGORICAL_COVARIATES = ["sex", "source"]
CONTINUOUS_COVARIATES = ["age"]

# Column-name overrides for Supplementary Data 2 (None = auto-detect)
OVERRIDE = {
    "supp_sheet": None, "lab_id_col": None, "subtype_col": None, "karyotype_col": None,
    "age_col": None, "sex_col": None, "source_col": None, "gsm_col": None,
}

# ------------------------------------------------------------- CV / modelling
PANEL_SIZES = [5, 10, 20, 50, 100]            # goes INTO the GridSearchCV grid (selector__k)
STABILITY_THRESHOLD = 0.70
STABILITY_SENSITIVITY = [0.60, 0.80]
REDUNDANCY_R = 0.90
MIN_FINAL_PANEL = 5
MAX_FINAL_PANEL = 100

_PRESETS = {
    "full": dict(outer_splits=5, outer_repeats=10, inner_splits=5, prefilter_top=5000,
                 consensus_boot=100, n_boot=5000, n_perm=10000, rf_trees=500, grid="full",
                 models=["elastic_net", "logistic_l2", "svm", "random_forest", "xgboost"]),
    "fast": dict(outer_splits=5, outer_repeats=2, inner_splits=3, prefilter_top=2000,
                 consensus_boot=30, n_boot=500, n_perm=2000, rf_trees=200, grid="small",
                 models=["elastic_net", "logistic_l2", "svm", "random_forest", "xgboost"]),
    "demo": dict(outer_splits=5, outer_repeats=2, inner_splits=3, prefilter_top=800,
                 consensus_boot=20, n_boot=300, n_perm=1000, rf_trees=100, grid="small",
                 models=["elastic_net", "logistic_l2", "random_forest", "xgboost"]),
}
P = _PRESETS[MODE]
OUTER_SPLITS, OUTER_REPEATS, INNER_SPLITS = P["outer_splits"], P["outer_repeats"], P["inner_splits"]
PREFILTER_TOP = P["prefilter_top"]

# Parsimony order used to break ties between models that are within one SE
MODEL_COMPLEXITY = {"elastic_net": 0, "logistic_l2": 1, "svm": 2, "random_forest": 3, "xgboost": 4}
PRIMARY_MODEL = "elastic_net"
LINEAR_MODELS = {"elastic_net", "logistic_l2"}
TREE_MODELS = {"random_forest", "xgboost"}
