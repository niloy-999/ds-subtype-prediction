"""
SYNTHETIC data generator (DS_MODE=demo) -- for smoke-testing the pipeline offline.

Writes files in the same formats the real parsers expect (GEO counts, family SOFT, Supplementary Data 2 xlsx,
GENCODE-style annotation table, GSE35665 series matrix + platform table). The numbers are simulated and carry
NO biological meaning. Never report demo output.
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

import config as C
from utils import get_logger, ensure_dirs

log = get_logger()
CELL_MARKERS = {
    "neutrophil": ["FCGR3B", "CXCR2", "CSF3R", "CXCR1", "FFAR2", "MNDA", "PROK2", "CA4"],
    "t_cell": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC", "IL7R", "LEF1", "TCF7"],
    "b_cell": ["CD19", "MS4A1", "CD79A", "CD79B", "BANK1", "CD22", "FCRLA", "VPREB3"],
    "nk_cell": ["NCAM1", "KLRD1", "KLRF1", "NKG7", "GNLY", "PRF1", "KLRB1", "GZMB"],
    "monocyte": ["CD14", "FCN1", "VCAN", "CSF1R", "MS4A6A", "CTSS", "LYZ", "CD68"],
    "dendritic": ["CD1C", "CLEC10A", "FCER1A", "CLEC4C", "IL3RA", "LILRA4", "CD1E", "CLEC9A"],
}
ANCH1 = ["PAXBP1", "GABPA", "BRWD1", "N6AMT1", "ATP5PF"]
ANCH2 = ["IFNAR1", "IFNGR2", "IL10RB", "KCNJ15", "SLC19A1", "ITGB2"]


def generate(seed: int = 7, n_other: int = 2500) -> None:
    ensure_dirs()
    rng = np.random.default_rng(seed)
    subtypes = np.array(["MS1"] * 107 + ["MS2"] * 95 + ["MS3"] * 102)
    n_t, n_c = 304, 96
    n = n_t + n_c
    lab = np.array([f"HTP{i:04d}A" for i in rng.permutation(np.arange(1, 2000))[:n]])
    gsm = np.array([f"GSM57{i:05d}" for i in range(n)])
    karyo = np.array(["T21"] * n_t + ["D21"] * n_c)
    sub = np.concatenate([subtypes, np.array([None] * n_c)])
    sex = rng.choice(["Female", "Male"], n)
    age = np.round(rng.uniform(1, 55, n), 1)
    source = rng.choice(["Trisome_Project", "Clinic_B", "Registry_C"], n, p=[.6, .25, .15])

    # ---------------- latent state
    mean_c1 = {"MS1": 1.0, "MS2": 0.2, "MS3": -1.0}
    mean_c2 = {"MS1": -1.0, "MS2": 0.3, "MS3": 1.0}
    c1 = np.zeros(n); c2 = np.zeros(n)
    for i in range(n_t):
        c1[i] = mean_c1[sub[i]] + rng.normal(0, .55)
        c2[i] = mean_c2[sub[i]] + rng.normal(0, .55)
    t21 = (karyo == "T21").astype(float)
    neut = 0.55 + 0.05 * c2 + rng.normal(0, .05, n)
    tcell = 0.25 - 0.03 * c2 + rng.normal(0, .04, n)
    bcell = 0.08 - 0.01 * c2 + rng.normal(0, .015, n)
    nk = 0.06 + rng.normal(0, .015, n)
    mono = 0.06 + rng.normal(0, .015, n)
    dc = 0.01 + rng.normal(0, .004, n)
    fr = {"neutrophil": neut, "t_cell": tcell, "b_cell": bcell, "nk_cell": nk, "monocyte": mono, "dendritic": dc}

    # ---------------- genes
    hsa21_sym = ANCH1 + [f"C1G{i:03d}" for i in range(97 - len(ANCH1))] + ANCH2 + [f"C2G{i:02d}" for i in range(29 - len(ANCH2))]
    hsa21_cl = np.array([1] * 97 + [2] * 29)
    marker_syms = [g for v in CELL_MARKERS.values() for g in v]
    ms1_up = [f"DOWN1_{i:02d}" for i in range(45)]
    ms3_up = [f"DOWN3_{i:02d}" for i in range(45)]
    n_noise = n_other - len(marker_syms) - len(ms1_up) - len(ms3_up)
    other_sym = marker_syms + ms1_up + ms3_up + [f"NOISE{i:04d}" for i in range(n_noise)]
    symbols = hsa21_sym + other_sym
    chrom = ["21"] * 126 + list(rng.choice([str(x) for x in range(1, 21)] + ["22", "X"], len(other_sym)))
    ens = [f"ENSG{100000 + i:011d}" for i in range(len(symbols))]
    G = len(symbols)

    base = np.exp(rng.normal(4.6, 1.1, G))
    lfc = np.zeros((n, G))
    for j in range(126):                                                 # HSA21: dosage + cluster activity
        act = c1 if hsa21_cl[j] == 1 else c2
        lfc[:, j] = 0.58 * t21 + 0.22 * t21 * act
    off = 126
    for cell, genes in CELL_MARKERS.items():                              # cell-composition-driven genes
        z = (fr[cell] - fr[cell].mean()) / fr[cell].std()
        for g in genes:
            lfc[:, off + marker_syms.index(g)] = 0.35 * z
    m0 = off + len(marker_syms)
    for j in range(len(ms1_up)):                                          # downstream, cell-independent
        lfc[:, m0 + j] = (0.26 + 0.005 * j) * c1 * t21
    for j in range(len(ms3_up)):
        lfc[:, m0 + len(ms1_up) + j] = (0.26 + 0.005 * j) * c2 * t21
    sex_genes = rng.choice(G, 40, replace=False); age_genes = rng.choice(G, 40, replace=False)
    src_genes = rng.choice(G, 60, replace=False)
    for j in sex_genes: lfc[:, j] += 0.3 * (sex == "Male")
    for j in age_genes: lfc[:, j] += 0.006 * (age - 25)
    for j in src_genes: lfc[:, j] += 0.25 * (source == "Clinic_B") - 0.2 * (source == "Registry_C")
    lib = np.exp(rng.normal(0, .2, n))
    mu = base[None, :] * lib[:, None] * 2.0 ** lfc
    disp = 0.08
    counts = rng.negative_binomial(1 / disp, 1 / (1 + mu * disp)).astype(int)

    # ---------------- write GEO-like files
    df = pd.DataFrame(counts.T, index=ens, columns=gsm)
    df.index.name = "gene_id"
    with gzip.open(C.COUNTS_PATH, "wt") as fh:
        df.reset_index().to_csv(fh, sep="\t", index=False)
    with gzip.open(C.SOFT_PATH, "wt") as fh:
        fh.write("^SERIES = GSE190125\n!Series_title = DEMO SYNTHETIC\n")
        for i in range(n):
            fh.write(f"^SAMPLE = {gsm[i]}\n!Sample_title = {lab[i]}\n!Sample_geo_accession = {gsm[i]}\n"
                     f"!Sample_source_name_ch1 = Whole blood\n!Sample_characteristics_ch1 = karyotype: {karyo[i]}\n"
                     f"!Sample_characteristics_ch1 = Sex: {sex[i]}\n")
    clin = pd.DataFrame({"LabID": lab, "Karyotype": karyo, "Molecular subtype": sub, "Age": age,
                         "Sex": sex, "Source": source})
    clusters = pd.DataFrame({"Gene": hsa21_sym, "HSA21 cluster": hsa21_cl})
    with pd.ExcelWriter(C.SUPP2_PATH) as xw:
        pd.DataFrame({"note": ["SYNTHETIC DEMO DATA"]}).to_excel(xw, sheet_name="README", index=False)
        clin.to_excel(xw, sheet_name="Clinical variables", index=False)
        clusters.to_excel(xw, sheet_name="HSA21 clusters", index=False)
    ann = pd.DataFrame({"gene_id": ens, "gene_symbol": symbols, "chromosome": chrom, "gene_type": "protein_coding"})
    ann.to_csv(C.GENE_ANNOTATION_CSV, index=False)

    # ---------------- GSE35665-like microarray (15 DS / 22 control), different platform & scale
    n_ds, n_ct = 15, 22
    ext_n = n_ds + n_ct
    is_ds = np.array([1] * n_ds + [0] * n_ct)
    grp = rng.choice([0, 1], n_ds, p=[.55, .45])                          # bimodal DS population
    e1 = np.zeros(ext_n); e2 = np.zeros(ext_n)
    for i in range(n_ds):
        e1[i], e2[i] = (1.0, -1.0) if grp[i] == 0 else (-1.0, 1.0)
    e1 += rng.normal(0, .5, ext_n); e2 += rng.normal(0, .5, ext_n)
    e1[n_ds:] = rng.normal(0, .4, n_ct); e2[n_ds:] = rng.normal(0, .4, n_ct)
    keep_syms = marker_syms + ms1_up + ms3_up + [f"NOISE{i:04d}" for i in range(400)] + hsa21_sym[:60]
    keep_syms = [s for s in keep_syms if rng.random() < 0.85]
    probes, rows = [], []
    for k, s in enumerate(keep_syms):
        for rep in range(2 if k % 5 == 0 else 1):
            pid = f"{7900000 + k * 3 + rep}"
            base_e = rng.uniform(5, 10)
            eff = np.zeros(ext_n)
            if s in ms1_up: eff = (0.28 + 0.005 * ms1_up.index(s)) * e1
            if s in ms3_up: eff = (0.28 + 0.005 * ms3_up.index(s)) * e2
            rows.append(base_e + eff + rng.normal(0, .3, ext_n)); probes.append((pid, s))
    E = pd.DataFrame(np.vstack(rows), index=[p[0] for p in probes],
                     columns=[f"GSM87{i:04d}" for i in range(ext_n)])
    with gzip.open(C.GSE35665_MATRIX, "wt") as fh:
        fh.write('!Series_platform_id\t"GPL5175"\n')
        fh.write("!Sample_geo_accession\t" + "\t".join(f'"{c}"' for c in E.columns) + "\n")
        fh.write("!Sample_title\t" + "\t".join(f'"{"DS" if d else "CTL"}_{i}"' for i, d in enumerate(is_ds)) + "\n")
        fh.write("!Sample_characteristics_ch1\t" + "\t".join(
            f'"disease state: {"Down syndrome" if d else "normal control"}"' for d in is_ds) + "\n")
        fh.write("!Sample_characteristics_ch1\t" + "\t".join(
            f'"age group: {"neonate" if i % 2 else "child"}"' for i in range(ext_n)) + "\n")
        fh.write("!series_matrix_table_begin\n")
        E.reset_index().rename(columns={"index": "ID_REF"}).to_csv(fh, sep="\t", index=False, quoting=1)
        fh.write("!series_matrix_table_end\n")
    plat = pd.DataFrame({"ID": [p[0] for p in probes],
                         "gene_assignment": [f"NM_{k:06d} // {p[1]} // demo transcript // 1q21 // {k}"
                                             for k, p in enumerate(probes)]})
    with gzip.open(C.GSE35665_PLATFORM, "wt") as fh:
        fh.write("^PLATFORM = GPL5175\n!platform_table_begin\n")
        plat.to_csv(fh, sep="\t", index=False)
        fh.write("!platform_table_end\n")
    log.info("DEMO: wrote synthetic GSE190125-like (%d genes x %d samples) and GSE35665-like data", G, n)
