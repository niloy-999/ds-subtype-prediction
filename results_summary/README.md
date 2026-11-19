# Results summary

Aggregate result tables produced by the pipeline (gene-level or group-level only).
Participant-level outputs (per-sample predictions, per-sample scores, QC audits), data, trained models and logs are
deliberately **not** included; regenerate them with `python run_pipeline.py`.

Table names follow the pipeline output names (`results/tables/`): model performance (`Table5_*`), signature genes
(`Table6_*`, `Table11_*`), differential-expression summary and concordance with the published analysis (`DE_summary.csv`,
`Table10_*`), SHAP agreement (`Table6c_*`), cell-composition benchmark (`Table8*`), chromosome-21 cluster analyses
(`Table9_*`, `HSA21_*`), stable-panel nested validation (`Table12*`) and the external cohort (`GSE35665_*`).
