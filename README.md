# DS Subtype Pipeline

[![CI](https://github.com/YOURUSER/ds-subtype-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/YOURUSER/ds-subtype-prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A reproducible, leakage-controlled pipeline for predicting the **molecular subtypes of Down syndrome (MS1, MS2, MS3)** from whole-blood RNA-seq, using genes **outside chromosome 21** only. It accompanies the manuscript *"A whole-blood transcriptional signature outside chromosome 21 predicts the molecular subtypes of Down syndrome"* (link to be added).

> **Research use only.** This code is not a diagnostic tool and has not been validated in an independent labelled cohort.

## What it does

| Stage | Script | Purpose |
|---|---|---|
| 01 | `01_setup_and_download.py` | Directory layout, resumable downloads of the public GEO datasets and supplementary data, integrity manifest |
| 02 | `02_qc_and_baseline.py` | Metadata harmonisation, QC, reproduction of the published subtypes, differential expression (PyDESeq2, controls retained) |
| 03 | `03_nested_cv_engine.py` | Repeated nested cross-validation (5 outer folds x 10 repeats), five classifiers, stability selection, final signature |
| 04 | `04_external_validation.py` | External cohort check (co-expression preservation, composite scores, multimodality tests) |
| 05 | `05_explainability.py` | SHAP (Kernel / Linear / Tree), chromosome-21 cluster polygenic scores, cell-composition benchmark |
| 06 | `06_concordance_and_panel_profile.py` | Concordance with the published DESeq2 results; per-subtype expression profile of the signature genes |
| 07 | `07_stable_panel_nested_validation.py` | Nested validation of the whole stable-panel procedure |
| 08 | `08_manuscript_figures.py` | Assembles the lettered manuscript composite figures from the pipeline outputs |
| - | `run_pipeline.py` | Master script: creates directories, runs the stages, writes an archive |

Design principles: all data-dependent steps (expression filter, covariate residualisation, gene screening, scaling, panel-size tuning) are fitted **inside training folds only**; the covariate residualiser uses `drop_first=True` dummies so the design is never collinear; the panel size is a hyper-parameter inside the grid search; fixed seed (42); package versions recorded on every run.

## Quick start

```bash
git clone https://github.com/YOURUSER/ds-subtype-prediction.git && cd ds-subtype-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt          # exact versions used for the paper (Python 3.13)
# or: pip install -r requirements.txt          # unpinned, minimum versions

python run_pipeline.py --mode demo             # offline smoke test on SYNTHETIC data (~3 min)
python run_pipeline.py --mode fast             # real data, reduced CV grid
python run_pipeline.py                         # full run (hours; downloads ~0.5 GB)
python run_pipeline.py --stages 06 07          # run selected stages only
python run_pipeline.py --help
```

`--mode demo` never touches the network and writes to `demo_workspace/`; its output is synthetic and has no biological meaning.

## Data

No data are redistributed here. Stage 01 downloads the public resources:

* **GSE190125**: whole-blood RNA-seq of the Human Trisome Project cohort (discovery).
* **GSE35665**: whole-blood microarray, Down syndrome versus controls (external check).
* **Supplementary Data 2** of Donovan et al., *Nat Commun* 2024;15:5473 (published subtype labels and DESeq2 results).
* GENCODE v33 gene annotation.

Please respect the terms of use of the original resources. Participant-level outputs are deliberately **not** included in this repository.

## Repository layout

```
run_pipeline.py            master script
01_ ... 08_*.py            numbered analysis stages
config.py                  all tunables (modes, grids, thresholds, paths)
ml_components.py           covariate residualiser, model zoo, one-SE rule
cv_engine.py               per-fold logic, stability selection, final panel
hsa21_clusters.py          chromosome-21 Cluster 1 / 2 polygenic scores
immune.py                  marker-gene proxy for cell composition
utils.py, data_access.py   shared helpers
results_summary/           aggregate result tables (no participant-level data)
requirements*.txt          dependencies (pinned lock file for the paper)
```

Running the pipeline creates `data_raw/`, `processed_data/` and `results/{tables,figures,models,logs}/` (git-ignored).

## Reproducibility notes

* Every run records Python and package versions in `results/logs/environment.json`.
* Stage 03 and stage 07 checkpoint each outer fold, so interrupted runs resume where they stopped (`--force-recompute` to redo).
* The numbers in the manuscript were produced with the versions in `requirements-lock.txt`; other versions may differ in the last decimal.

## Citation

If you use this software, please cite the manuscript (above) and this repository (see `CITATION.cff`). A DOI is provided with each tagged release.

## License

MIT, see [LICENSE](LICENSE).
