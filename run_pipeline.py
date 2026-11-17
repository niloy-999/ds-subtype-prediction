#!/usr/bin/env python3
"""
Master execution script for the Down-syndrome molecular-subtype pipeline.

Usage:
    python run_pipeline.py                       # full publication run (needs internet + real deps)
    python run_pipeline.py --mode fast            # same stages, smaller CV grid, quicker
    python run_pipeline.py --mode demo            # SYNTHETIC data, offline smoke-test of every stage
    python run_pipeline.py --stages 01 02         # run only selected stages (in order)
    python run_pipeline.py --skip-de              # skip PyDESeq2 (DS_SKIP_DE=1)
    python run_pipeline.py --no-zip               # don't produce the deployable .zip at the end

Creates data_raw/, processed_data/, results/{tables,figures,models,logs}/, downloads GSE190125, GSE35665 and
Supplementary Data 2, runs stages 01-05 in order, and archives the whole workspace into
DS_Subtype_Pipeline.zip (unless --no-zip).
"""
from __future__ import annotations

import argparse
import os
import runpy
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("01", "01_setup_and_download.py"),
    ("02", "02_qc_and_baseline.py"),
    ("03", "03_nested_cv_engine.py"),
    ("04", "04_external_validation.py"),
    ("05", "05_explainability.py"),
    ("06", "06_concordance_and_panel_profile.py"),
    ("07", "07_stable_panel_nested_validation.py"),
    ("08", "08_manuscript_figures.py"),
]
ZIP_EXCLUDE_DIRS = {"data_raw", "processed_data", "demo_workspace", "__pycache__", ".ipynb_checkpoints"}
ZIP_EXCLUDE_SUFFIX = {".pyc"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["full", "fast", "demo"], default="full",
                  help="full = publication settings; fast = quicker real-data run; demo = offline synthetic smoke-test")
    p.add_argument("--stages", nargs="+", choices=[s[0] for s in STAGES], default=[s[0] for s in STAGES],
                  help="which stage numbers to run, in order (default: all)")
    p.add_argument("--jobs", type=int, default=None, help="override DS_N_JOBS (parallel workers)")
    p.add_argument("--skip-de", action="store_true", help="skip PyDESeq2 / OLS differential expression in stage 02")
    p.add_argument("--force-recompute", action="store_true", help="ignore stage-03 fold checkpoints and refit everything")
    p.add_argument("--list-stages", action="store_true", help="show the stages this copy of the pipeline knows about and exit")
    p.add_argument("--no-zip", action="store_true", help="do not create the final deployable .zip")
    p.add_argument("--zip-name", default="DS_Subtype_Pipeline.zip", help="name of the output archive")
    return p.parse_args()


def check_dependencies(mode: str) -> None:
    required = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "joblib", "requests", "openpyxl"]
    strongly_recommended = {"statsmodels": "used by some SciPy fallbacks", "xgboost": "XGBoost comparator model",
                            "shap": "SHAP beeswarm/dependence plots", "pydeseq2": "negative-binomial differential expression"}
    missing_req = [m for m in required if _cannot_import(m)]
    if missing_req:
        sys.exit(f"Missing required packages: {missing_req}. Install with:\n"
                 f"  pip install {' '.join(missing_req)}\nSee requirements.txt.")
    for m, why in strongly_recommended.items():
        if _cannot_import(m):
            print(f"[run_pipeline] NOTE: '{m}' not installed ({why}); a documented fallback will be used "
                 f"(see the module docstrings). pip install {m}  to enable it.")


def _cannot_import(name: str) -> bool:
    try:
        __import__(name)
        return False
    except Exception:  # noqa: BLE001
        return True


def run_stage_script(path: Path) -> None:
    print(f"\n{'#' * 78}\n#  RUNNING {path.name}\n{'#' * 78}")
    t0 = time.time()
    runpy.run_path(str(path), run_name="__main__")
    print(f"#  {path.name} finished in {(time.time() - t0) / 60:.1f} min")


def make_zip(mode: str, zip_name: str) -> Path:
    out = ROOT / zip_name
    if out.exists():
        out.unlink()
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in ROOT.rglob("*"):
            if p.is_dir() or p == out:
                continue
            if any(part in ZIP_EXCLUDE_DIRS for part in p.relative_to(ROOT).parts):
                continue
            if p.suffix in ZIP_EXCLUDE_SUFFIX or p.name == zip_name:
                continue
            zf.write(p, p.relative_to(ROOT))
            n += 1
    print(f"\nArchived {n} files -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print("Raw downloads (data_raw/) and intermediate matrices (processed_data/) are excluded to keep the "
         "archive small and reproducible; stage 01 will re-download / stage 02 will rebuild them on extraction.")
    return out


def main() -> None:
    args = parse_args()
    if args.list_stages:
        print("DS Subtype Pipeline - stages available in this copy:")
        for num, fname in STAGES:
            print(f"  {num}  {fname:42s} {'ok' if (ROOT / fname).exists() else 'FILE MISSING'}")
        sys.exit(0)
    os.environ["DS_MODE"] = args.mode
    if args.jobs:
        os.environ["DS_N_JOBS"] = str(args.jobs)
    if args.skip_de:
        os.environ["DS_SKIP_DE"] = "1"
    if args.force_recompute:
        os.environ["DS_FORCE"] = "1"

    check_dependencies(args.mode)
    sys.path.insert(0, str(ROOT))
    import config as C          # noqa: E402  (import after DS_MODE is set)
    from utils import ensure_dirs, get_logger  # noqa: E402

    ensure_dirs()
    log = get_logger()
    from utils import write_environment  # noqa: E402
    write_environment()
    log.info("DS Subtype Pipeline | mode=%s | stages=%s | workspace=%s", args.mode, args.stages, C.WORK)
    if args.mode == "demo":
        log.warning("DEMO MODE: all data are SYNTHETIC. Results have no biological meaning; this run only "
                    "verifies that every stage executes end-to-end.")

    t0 = time.time()
    for num, fname in STAGES:
        if num in args.stages:
            run_stage_script(ROOT / fname)
    log.info("Pipeline complete in %.1f min. Tables: %s | Figures: %s | Model: %s",
             (time.time() - t0) / 60, C.TABLES, C.FIGURES, C.MODELS / "final_subtype_model.joblib")

    if not args.no_zip:
        make_zip(args.mode, args.zip_name)


if __name__ == "__main__":
    main()
