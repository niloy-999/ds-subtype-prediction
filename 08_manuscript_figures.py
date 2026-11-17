"""
Stage 08 - assemble lettered, manuscript-ready composite figures from the pipeline's own figure files.

Main figures (results/figures/manuscript/):
  Fig2_subtype_reproduction, Fig3_model_comparison_panel_size, Fig4_performance_and_stable_panel_validation,
  Fig5_panel_gene_behaviour, Fig6_hsa21_axis_and_cell_composition, Fig7_external_cohort_GSE35665
Supplementary figures (results/figures/manuscript/supplementary/): FigS1 ... FigS8.
(Fig. 1, the study-design schematic, is a static drawing supplied with the manuscript package.)
Panels A/B of Figure 4 (confusion matrix, calibration) are re-plotted from the pooled out-of-fold predictions.
Composites whose source files are missing (e.g. a stage was not run) are skipped with a warning.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import config as C
from utils import get_logger, run_stage

log = get_logger()
SRC = C.FIGURES
OUT = C.FIGURES / "manuscript"
SUP = OUT / "supplementary"
W = 3200
FONT = os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans-Bold.ttf")


def load(name: str) -> Optional[Image.Image]:
    p = SRC / name
    return Image.open(p).convert("RGB") if p.exists() else None


def pick(*names: str) -> Optional[Image.Image]:
    """First existing file among alternatives (e.g. the 60% sensitivity version, else the 70% one)."""
    for n in names:
        im = load(n)
        if im is not None:
            return im
    return None


def fw(im: Image.Image, w: int) -> Image.Image:
    return im.resize((w, max(1, round(im.height * w / im.width))), Image.LANCZOS)


def letter(cv: Image.Image, xy, s: str, size: int = 100) -> None:
    ImageDraw.Draw(cv).text(xy, s, font=ImageFont.truetype(FONT, size), fill="black")


def save(im: Image.Image, path, maxw: int = W) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if im.width > maxw:
        im = fw(im, maxw)
    im.save(path, optimize=True)
    log.info("composite -> %s %s", path.relative_to(C.WORK), im.size)


def stack(imgs: Sequence[Image.Image], letters: Sequence[Optional[str]], width: int = W, gap: int = 140, top: int = 80) -> Image.Image:
    imgs = [fw(i, width) for i in imgs]
    cv = Image.new("RGB", (width, top + sum(i.height for i in imgs) + gap * (len(imgs) - 1)), "white")
    y = top
    for im, s in zip(imgs, letters):
        cv.paste(im, (0, y))
        if s:
            letter(cv, (10, y - top), s)
        y += im.height + gap
    return cv


def columns(im: Image.Image, n: int, letters: str, width: int = W, top: int = 90) -> Image.Image:
    im = fw(im, width)
    cv = Image.new("RGB", (width, im.height + top), "white")
    cv.paste(im, (0, top))
    for i, s in enumerate(letters):
        letter(cv, (int(i * width / n) + 10, 0), s)
    return cv


def row(imgs: Sequence[Image.Image], height: int, letters: Sequence[str], size: int = 90, gap: int = 30) -> Image.Image:
    imgs = [i.resize((max(1, round(i.width * height / i.height)), height), Image.LANCZOS) for i in imgs]
    cv = Image.new("RGB", (sum(i.width for i in imgs) + gap * len(imgs), height + size + 20), "white")
    x = 0
    for im, s in zip(imgs, letters):
        cv.paste(im, (x, size + 20))
        letter(cv, (x + 10, 0), s, size)
        x += im.width + gap
    return cv


def need(label: str, *ims) -> bool:
    if any(i is None for i in ims):
        log.warning("skipping %s: a source figure is missing (run the earlier stages)", label)
        return False
    return True


def fig4_panels_ab() -> Optional[Image.Image]:
    p = C.TABLES / "chosen_model_oof_predictions_aggregated.csv"
    if not p.exists():
        return None
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import label_binarize
    a = pd.read_csv(p)
    classes = C.CLASSES
    colors = {"MS1": "#d62728", "MS2": "#2ca02c", "MS3": "#1f77b4"}
    cm = confusion_matrix(a.true, a.pred, labels=classes)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax[0].text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(classes); ax[0].set_yticks(range(3)); ax[0].set_yticklabels(classes)
    ax[0].set_xlabel("Predicted subtype"); ax[0].set_ylabel("Published subtype")
    ax[0].set_title(f"Pooled out-of-fold confusion matrix\n({int(np.trace(cm))}/{cm.sum()} correct, {100 * np.trace(cm) / cm.sum():.1f}%)", fontsize=10)
    yb = label_binarize(a.true, classes=classes)
    for j, c in enumerate(classes):
        frac, mp = calibration_curve(yb[:, j], a[f"prob_{c}"], n_bins=6, strategy="quantile")
        ax[1].plot(mp, frac, "o-", color=colors[c], label=c)
    ax[1].plot([0, 1], [0, 1], "k:")
    ax[1].set_xlabel("Mean predicted probability"); ax[1].set_ylabel("Observed fraction")
    ax[1].set_title("One-vs-rest calibration (6 quantile bins)", fontsize=10)
    ax[1].legend(frameon=False)
    for s in ("top", "right"):
        ax[1].spines[s].set_visible(False)
    fig.tight_layout()
    tmp = C.FIGURES / "_fig4AB_tmp.png"
    fig.savefig(tmp, dpi=300)
    plt.close(fig)
    im = Image.open(tmp).convert("RGB")
    tmp.unlink()
    return im


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SUP.mkdir(parents=True, exist_ok=True)

    im = load("Fig3_subtype_reproduction.png")
    if need("Figure 2", im):
        save(columns(im, 4, "ABCD"), OUT / "Fig2_subtype_reproduction.png")

    a, b = load("Fig5_model_comparison.png"), load("Fig6_panel_size_and_stability.png")
    if need("Figure 3", a, b):
        a, b = fw(a, W), fw(b, W)
        cv = Image.new("RGB", (W, a.height + b.height + 200), "white")
        cv.paste(a, (0, 90)); cv.paste(b, (0, a.height + 200))
        for i, s in enumerate("ABC"):
            letter(cv, (int(i * W / 3) + 10, 0), s)
        for i, s in enumerate("DEF"):
            letter(cv, (int(i * W / 3) + 10, a.height + 110), s)
        save(cv, OUT / "Fig3_model_comparison_panel_size.png")

    ab, c = fig4_panels_ab(), load("Fig13_stable_panel_nested_validation.png")
    if need("Figure 4", ab, c):
        ab, c = fw(ab, W), fw(c, W)
        cv = Image.new("RGB", (W, ab.height + c.height + 150), "white")
        cv.paste(ab, (0, 80)); cv.paste(c, (0, ab.height + 150))
        letter(cv, (10, 0), "A"); letter(cv, (W // 2 + 10, 0), "B"); letter(cv, (10, ab.height + 80), "C")
        save(cv, OUT / "Fig4_performance_and_stable_panel_validation.png")

    a, b = load("Fig12_panel_gene_subtype_profile.png"), load("Fig8_shap_beeswarm_MS1.png")
    if need("Figure 5", a, b):
        save(row([a, b], 2100, "AB", 100), OUT / "Fig5_panel_gene_behaviour.png")

    a = pick("Fig9_HSA21_cluster_validation_freq60.png", "Fig9_HSA21_cluster_validation.png")
    b = load("Fig10c_cell_composition_benchmark.png")
    if need("Figure 6", a, b):
        save(stack([a, b], "AB"), OUT / "Fig6_hsa21_axis_and_cell_composition.png")

    a = pick("Fig10a_coexpression_preservation_freq60.png", "Fig10a_coexpression_preservation.png")
    b = pick("Fig10b_composite_signature_scores_freq60.png", "Fig10b_composite_signature_scores.png")
    if need("Figure 7", a, b):
        save(stack([a, b], "AB"), OUT / "Fig7_external_cohort_GSE35665.png")

    # ---------------- supplementary
    for name, out in [("Fig2_QC_overview.png", "FigS1_QC_overview.png"),
                      ("Fig4_differential_expression_volcanos.png", "FigS2_differential_expression_volcanos.png"),
                      ("Fig11_DE_concordance_with_published.png", "FigS3_DE_concordance_with_published.png"),
                      ("Fig8_coefficients_and_permutation_importance.png", "FigS5_permutation_importance_coefficients.png"),
                      ("Fig9_HSA21_cluster_validation.png", "FigS6_HSA21_16gene_signature.png")]:
        im = load(name)
        if need(out, im):
            save(im, SUP / out)

    def beeswarms(prefix: str, classes: Sequence[str], letters: str):
        ims = [load(f"{prefix}_shap_beeswarm_{c}.png") for c in classes]
        return row(ims, 1400, letters, 80) if all(i is not None for i in ims) else None

    for prefix, classes, letters, out, lts in [("Fig8S_elastic_net", ("MS1", "MS2", "MS3"), "ABC", "FigS4_elastic_net_SHAP.png", [None, "D", "E"]),
                                              ("Fig8", ("MS2", "MS3"), "AB", "FigS8_SVM_Kernel_SHAP_additional.png", [None, "C", "D"])]:
        r, dep, loc = beeswarms(prefix, classes, letters), load(f"{prefix}_shap_dependence.png"), load(f"{prefix}_shap_local_examples.png")
        if need(out, r, dep, loc):
            save(stack([r, dep, loc], lts, gap=120), SUP / out)

    a, b = load("Fig10a_coexpression_preservation.png"), load("Fig10b_composite_signature_scores.png")
    if need("FigS7", a, b):
        save(stack([a, b], "AB"), SUP / "FigS7_GSE35665_16gene_signature.png")


if __name__ == "__main__":
    run_stage("08 manuscript figure composites", main)
