"""
Stage 01 - directory structure, data download, integrity manifest, gene annotation.

Downloads (real modes): GSE190125 counts + family SOFT (+FPKM), Supplementary Data 2 and Source Data of
Donovan et al. 2024, GENCODE v33 annotation, GSE35665 series matrix + platform annotation.
Existing non-empty files are never overwritten (raw data are treated as read-only).
Demo mode writes SYNTHETIC files instead (see demo_data.py).
"""
from __future__ import annotations

import gzip
import platform
import re
import sys
from datetime import date

import pandas as pd

import config as C
from utils import (write_environment, download_file, ensure_dirs, get_logger, open_text, parse_series_matrix, run_stage,
                   sha256_file, save_json)

log = get_logger()

MANUAL_HELP = """
Manual fallback: download the file in a browser and place it in {dest_dir} under the name {name}.
  GSE190125 : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE190125  (Supplementary files / SOFT)
  Paper     : https://doi.org/10.1038/s41467-024-49781-1  (Supplementary Data 2 = MOESM5, Source Data = MOESM13)
  GSE35665  : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE35665
"""


def parse_gencode_gtf() -> None:
    """GENCODE GTF -> processed_data/gene_annotation.csv (gene_id without version, symbol, chromosome, type)."""
    if C.GENE_ANNOTATION_CSV.exists() or not C.GENCODE_GTF.exists():
        return
    log.info("Parsing GENCODE GTF (gene records only) ...")
    rx = {k: re.compile(rf'{k} "([^"]+)"') for k in ("gene_id", "gene_type", "gene_name")}
    rows = []
    with gzip.open(C.GENCODE_GTF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 8)
            if len(f) < 9 or f[2] != "gene":
                continue
            attr = f[8]
            g = {k: (m.group(1) if (m := r.search(attr)) else "") for k, r in rx.items()}
            rows.append((re.sub(r"\.\d+$", "", g["gene_id"]), g["gene_name"], f[0].replace("chr", ""), g["gene_type"]))
    ann = pd.DataFrame(rows, columns=["gene_id", "gene_symbol", "chromosome", "gene_type"]).drop_duplicates("gene_id")
    ann.to_csv(C.GENE_ANNOTATION_CSV, index=False)
    log.info("gene annotation: %d genes, %d on chr21", len(ann), (ann.chromosome == "21").sum())


def download_platform_annotation() -> None:
    """Fetch just the probe -> gene-symbol annotation table for the GSE35665 platform.

    Uses the GEO "acc.cgi?...&targ=self&view=full" endpoint, which returns ONLY that platform's own SOFT
    record (definition + data table) as plain text. This is deliberately NOT the platforms/.../soft/
    <GPL>_family.soft.gz file: for a widely used commercial array, the platform "family" file bundles every
    series and sample ever run on that platform and can run into the GIGABYTES (that file is what a previous
    version of this script tried to download here, and it is almost certainly why that attempt was
    reading multiple GB and still timing out). The curated .annot.gz static file is tried first since it is
    smaller and cleaner when it exists; acc.cgi is the reliable fallback that exists for essentially every
    platform.
    """
    if C.GSE35665_PLATFORM.exists() or not C.GSE35665_MATRIX.exists():
        return
    try:
        _, _, gpl = parse_series_matrix(C.GSE35665_MATRIX)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read platform id from GSE35665 series matrix: %r", exc)
        return
    gpl = gpl or "GPL5175"
    prefix = "GPL" + re.sub(r"\D", "", gpl)[:-3] + "nnn"
    annot_url = f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{prefix}/{gpl}/annot/{gpl}.annot.gz"
    if download_file(annot_url, C.GSE35665_PLATFORM, retries=2, min_expected_bytes=2048):
        return
    log.info("no curated .annot.gz for %s -> fetching the platform's own SOFT record via acc.cgi", gpl)
    acc_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gpl}&targ=self&form=text&view=full"
    try:
        import requests
        r = requests.get(acc_url, timeout=(30, 120),
                         headers={"User-Agent": "Mozilla/5.0 (DS-Subtype-Pipeline; academic use)"})
        r.raise_for_status()
        text = r.text
        if "!platform_table_begin" not in text or len(text) < 2048:
            raise ValueError(f"acc.cgi response for {gpl} does not look like a platform record "
                             f"({len(text)} chars, has table: {'!platform_table_begin' in text})")
        C.GSE35665_PLATFORM.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(C.GSE35665_PLATFORM, "wt", encoding="utf-8") as fh:
            fh.write(text)
        log.info("downloaded %s platform annotation via acc.cgi (%.1f KB)", gpl, len(text) / 1e3)
        return
    except Exception as exc:  # noqa: BLE001
        log.warning("acc.cgi platform-record fetch failed for %s: %r", gpl, exc)
    log.warning("Platform annotation for %s could not be downloaded. %s", gpl,
                MANUAL_HELP.format(dest_dir=C.DATA_EXTERNAL, name=C.GSE35665_PLATFORM.name))


def write_manifest() -> None:
    rows = []
    for name, (url, dest, mandatory) in C.DOWNLOADS.items():
        if dest.exists():
            rows.append({"resource": name, "file": dest.name, "source_url": url, "mandatory": mandatory,
                         "size_bytes": dest.stat().st_size, "sha256": sha256_file(dest),
                         "retrieved": date.today().isoformat()})
    if C.GSE35665_PLATFORM.exists():
        rows.append({"resource": "GSE35665 platform", "file": C.GSE35665_PLATFORM.name, "source_url": "GEO FTP",
                     "mandatory": True, "size_bytes": C.GSE35665_PLATFORM.stat().st_size,
                     "sha256": sha256_file(C.GSE35665_PLATFORM), "retrieved": date.today().isoformat()})
    m = pd.DataFrame(rows)
    m.to_csv(C.DATA_RAW / "data_manifest.tsv", sep="\t", index=False)
    with open(C.DATA_RAW / "checksums.txt", "w") as fh:
        for r in rows:
            fh.write(f"{r['sha256']}  {r['file']}\n")
    log.info("data manifest + SHA-256 checksums written (%d files)", len(rows))


def main() -> None:
    ensure_dirs()
    write_environment()
    if C.DEMO:
        import demo_data
        demo_data.generate()
        write_manifest()
        return

    missing = []
    for name, (url, dest, mandatory) in C.DOWNLOADS.items():
        if dest.exists() and dest.stat().st_size > 0:
            log.info("present: %s", dest.name)
            continue
        ok = download_file(url, dest)
        if not ok:
            (log.error if mandatory else log.warning)("could not download %s", name)
            if mandatory:
                missing.append((name, dest))
    download_platform_annotation()
    parse_gencode_gtf()
    write_manifest()
    if missing:
        msg = "; ".join(f"{n} -> {d.name}" for n, d in missing)
        raise RuntimeError("Mandatory files missing: " + msg + "\n" +
                           MANUAL_HELP.format(dest_dir=C.DATA_RAW, name="<file name above>"))
    if not C.GENE_ANNOTATION_CSV.exists():
        log.warning("No GENCODE annotation available: stage 02 will fall back to annotation columns inside the "
                    "count matrix (or fail with instructions).")


if __name__ == "__main__":
    run_stage("01 setup & download", main)
