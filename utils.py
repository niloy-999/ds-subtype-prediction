"""Shared helpers: logging, IO, GEO parsers, statistics."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import re
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
SUBTYPE_COLORS = {"MS1": "#d62728", "MS2": "#2ca02c", "MS3": "#1f77b4", "Control": "#7f7f7f", "T21": "#ff7f0e"}


def read_excel_skip_banner(path, sheet_name=None, max_scan_rows: int = 15, min_header_cells: int = 2):
    """Read an Excel workbook whose sheets may start with one or more banner/title rows (e.g. a single
    descriptive sentence in column 0, possibly followed by blank rows) before the real header row -- a
    pattern used throughout the real Donovan et al. Supplementary Data 2 workbook (2 title rows + 1 blank
    row before the header on every sheet). The first row (within the first `max_scan_rows`) with at least
    `min_header_cells` non-null values is treated as the header; rows above it are discarded. Returns a dict
    {sheet_name: DataFrame} (or a single DataFrame if `sheet_name` names one sheet)."""
    raw_sheets = pd.read_excel(path, sheet_name=sheet_name, header=None)
    single = isinstance(raw_sheets, pd.DataFrame)
    items = {sheet_name: raw_sheets} if single else raw_sheets
    out: Dict[str, pd.DataFrame] = {}
    for name, df in items.items():
        header_row = 0
        for i in range(min(max_scan_rows, len(df))):
            if df.iloc[i].notna().sum() >= min_header_cells:
                header_row = i
                break
        cols, seen, dedup = df.iloc[header_row].tolist(), {}, []
        for j, c in enumerate(cols):
            name_c = str(c).strip() if pd.notna(c) else f"Unnamed_{j}"
            if name_c in seen:
                seen[name_c] += 1
                name_c = f"{name_c}.{seen[name_c]}"
            else:
                seen[name_c] = 0
            dedup.append(name_c)
        data = df.iloc[header_row + 1:].copy()
        data.columns = dedup
        out[name] = data.reset_index(drop=True)
    return out[sheet_name] if single else out


# ---------------------------------------------------------------- environment
def ensure_dirs() -> None:
    for d in C.ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def write_environment() -> Path:
    """Record Python/OS/package versions + mode/seed in results/logs/environment.json (for the methods section)."""
    import importlib.metadata as md
    import platform
    ensure_dirs()
    pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "pydeseq2", "shap", "xgboost",
            "openpyxl", "requests", "diptest"]
    versions = {}
    for pk in pkgs:
        try:
            versions[pk] = md.version(pk)
        except Exception:  # noqa: BLE001
            versions[pk] = "not installed"
    path = C.LOGS / "environment.json"
    save_json({"python": sys.version, "platform": platform.platform(), "mode": C.MODE, "seed": C.SEED,
               "packages": versions}, path)
    return path


def get_logger(name: str = "pipeline") -> logging.Logger:
    ensure_dirs()
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(C.LOGS / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)
    return log


def run_stage(title: str, fn: Callable[[], None]) -> None:
    """Run a stage's main(); log the traceback and exit non-zero on failure."""
    log = get_logger()
    log.info("=" * 78)
    log.info("STAGE: %s   [mode=%s]", title, C.MODE)
    log.info("=" * 78)
    t0 = time.time()
    try:
        fn()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        log.error("STAGE FAILED: %s -> %r", title, exc)
        log.error(traceback.format_exc())
        sys.exit(1)
    log.info("STAGE DONE: %s (%.1f min)", title, (time.time() - t0) / 60)


def optional_import(name: str):
    try:
        return __import__(name)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------- IO
def open_text(path: Path):
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8", errors="ignore") if str(path).endswith(".gz") \
        else open(path, "rt", encoding="utf-8", errors="ignore")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


_MAGIC = {".gz": [b"\x1f\x8b"], ".zip": [b"PK\x03\x04", b"PK\x05\x06"], ".xlsx": [b"PK\x03\x04", b"PK\x05\x06"]}


def _content_looks_valid(dest: Path, part: Path) -> tuple[bool, str]:
    """Sniff the first bytes of a completed download against the expected magic number for dest's extension.
    Catches the (surprisingly common) case where a server returns an HTML interstitial / error / redirect
    page with a 200 status and a plausible size instead of the real binary file."""
    magics = _MAGIC.get(dest.suffix.lower())
    if not magics:
        return True, ""
    with open(part, "rb") as fh:
        head = fh.read(64)
    if any(head.startswith(m) for m in magics):
        return True, ""
    snippet = head[:40].decode("latin-1", errors="replace").replace("\n", " ")
    return False, snippet


class _BadContent(Exception):
    """Raised when a response completed at the HTTP level but the content is wrong (HTML interstitial,
    truncated/tiny body, bad magic bytes) -- as opposed to a transport-level failure (timeout, reset), where
    the partial bytes already on disk are still good and should be kept for the next Range-resume attempt."""


def _download_one_url(session, url: str, dest: Path, part: Path, retries: int, connect_timeout: int,
                      read_timeout: int, min_expected_bytes: int) -> bool:
    """Resumable (HTTP Range) download of a single URL into <part>. Each retry resumes from the bytes
    already on disk instead of restarting -- this is what makes large (>100MB) GEO files survive a flaky
    connection that resets every few minutes. After a completed transfer, the content is sniffed against the
    expected magic bytes for dest's extension (.gz / .zip / .xlsx): a server that silently returns an HTML
    interstitial instead of the real file (e.g. a GEO "large file, confirm download" page) is treated as a
    failed attempt, not a successful one, even though the HTTP status was 200 and the body was non-trivial.
    Content problems always discard <part> and restart clean; transport problems (timeouts, resets) keep it
    so the next attempt resumes instead of re-downloading from byte 0."""
    log = get_logger()
    headers = {"User-Agent": "Mozilla/5.0 (DS-Subtype-Pipeline; academic use)"}
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req_headers = dict(headers)
        if have:
            req_headers["Range"] = f"bytes={have}-"
        try:
            with session.get(url, stream=True, timeout=(connect_timeout, read_timeout), headers=req_headers) as r:
                if have and r.status_code == 416:                      # server says we already have it all
                    break
                if have and r.status_code != 206:                      # server can't resume -> start over
                    log.warning("%s does not support resume (status %s) -> restarting from 0", dest.name, r.status_code)
                    part.unlink(missing_ok=True)
                    have = 0
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "")
                if "html" in ctype.lower() and dest.suffix.lower() != ".html":
                    raise _BadContent(f"server returned Content-Type: {ctype!r} instead of a binary file "
                                      "(likely a redirect / confirmation / error page)")
                mode = "ab" if have else "wb"
                with open(part, mode) as fh:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            fh.write(chunk)
            size = part.stat().st_size
            if size < min_expected_bytes:
                raise _BadContent(f"downloaded file suspiciously small ({size} bytes)")
            ok, snippet = _content_looks_valid(dest, part)
            if not ok:
                raise _BadContent(f"content does not match expected file type for {dest.suffix} "
                                  f"(server likely returned an HTML/error page, starts with: {snippet!r})")
            return True
        except _BadContent as exc:
            log.warning("download attempt %d/%d failed for %s: %r -- discarding and restarting clean",
                       attempt, retries, dest.name, exc)
            part.unlink(missing_ok=True)
            time.sleep(min(30, 2 * attempt))
        except Exception as exc:  # noqa: BLE001  -- transport-level: keep partial bytes for Range-resume
            got = part.stat().st_size if part.exists() else 0
            log.warning("download attempt %d/%d failed for %s (have %.1f MB so far, will resume): %r",
                       attempt, retries, dest.name, got / 1e6, exc)
            time.sleep(min(30, 2 * attempt))
    if not (part.exists() and part.stat().st_size >= min_expected_bytes):
        return False
    ok, _ = _content_looks_valid(dest, part)
    return ok


def download_file(url, dest: Path, retries: int = 6, connect_timeout: int = 30, read_timeout: int = 120,
                  min_expected_bytes: int = 1024) -> bool:
    """Download to <dest>.part with HTTP-Range resume, then rename to <dest>. `url` may be a single URL string
    or a list of candidate URLs tried in order (each gets its own `retries` resumable attempts, continuing
    the same .part file only while trying the SAME url; switching url restarts that url's own attempts fresh).
    Returns True on success."""
    import requests
    log = get_logger()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    urls = [url] if isinstance(url, str) else list(url)
    for i, u in enumerate(urls):
        if i > 0:
            part.unlink(missing_ok=True)     # different host/endpoint -> byte ranges aren't comparable
        with requests.Session() as session:
            ok = _download_one_url(session, u, dest, part, retries, connect_timeout, read_timeout, min_expected_bytes)
        if ok:
            part.replace(dest)
            log.info("downloaded %s (%.1f MB) <- %s", dest.name, dest.stat().st_size / 1e6, u)
            return True
        log.warning("giving up on URL %d/%d for %s: %s", i + 1, len(urls), dest.name, u)
    part.unlink(missing_ok=True)
    return False


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> Path:
    path = C.TABLES / name
    df.to_csv(path, index=index)
    get_logger().info("table  -> %s  %s", path.relative_to(C.WORK), tuple(df.shape))
    return path


def save_fig(name: str, fig=None) -> Path:
    path = C.FIGURES / name
    (fig or plt.gcf()).savefig(path)
    plt.close(fig or plt.gcf())
    get_logger().info("figure -> %s", path.relative_to(C.WORK))
    return path


def save_matrix(df: pd.DataFrame, path: Path) -> None:
    """Portable compressed matrix store (values + labels), independent of pandas version."""
    np.savez_compressed(path, values=df.to_numpy(dtype=np.float32),
                        index=np.asarray(df.index.astype(str).tolist(), dtype="U"),
                        columns=np.asarray(df.columns.astype(str).tolist(), dtype="U"))


def load_matrix(path: Path) -> pd.DataFrame:
    z = np.load(path, allow_pickle=False)
    return pd.DataFrame(z["values"], index=z["index"], columns=z["columns"])


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------ id / name tools
def clean_name(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(v).strip().lower()).strip("_")


def norm_id(v: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(v).upper())


def strip_ensembl_version(v: str) -> str:
    return re.sub(r"\.\d+$", "", str(v))


def find_column(columns: Sequence[str], candidates: Sequence[str], override: Optional[str] = None,
                allow_partial: bool = True) -> Optional[str]:
    if override:
        if override not in columns:
            raise KeyError(f"Override column {override!r} not found in {list(columns)[:20]}")
        return override
    norm = {clean_name(c): c for c in columns}
    for cand in candidates:
        if clean_name(cand) in norm:
            return norm[clean_name(cand)]
    if allow_partial:
        for cand in candidates:
            tok = clean_name(cand)
            for n, orig in norm.items():
                if tok and tok in n:
                    return orig
    return None


# ---------------------------------------------------------------- statistics
def bh_fdr(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    q = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1].clip(0, 1)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[ok] = tmp
    return out


def log_cpm(counts_gxs: pd.DataFrame, prior: float = 1.0) -> pd.DataFrame:
    """log2(CPM + prior); library size = column sum over ALL genes supplied."""
    lib = counts_gxs.sum(axis=0).replace(0, np.nan)
    return np.log2(counts_gxs.div(lib, axis=1) * 1e6 + prior)


def ols_contrasts(Y: np.ndarray, X: np.ndarray, contrasts: Dict[str, np.ndarray]) -> Dict[str, pd.DataFrame]:
    """Vectorised per-gene OLS t-tests. Y: n x g, X: n x p (with intercept). Returns per-contrast tables."""
    from scipy import stats
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    resid = Y - X @ beta
    dof = max(n - np.linalg.matrix_rank(X), 1)
    s2 = (resid ** 2).sum(axis=0) / dof
    out = {}
    for name, c in contrasts.items():
        c = np.asarray(c, dtype=float)
        est = c @ beta
        se = np.sqrt(s2 * (c @ XtX_inv @ c))
        t = est / np.where(se == 0, np.nan, se)
        pv = 2 * stats.t.sf(np.abs(t), dof)
        out[name] = pd.DataFrame({"log2FoldChange": est, "stat": t, "pvalue": pv, "padj": bh_fdr(pv)})
    return out


# ------------------------------------------------------------- GEO parsers
def parse_soft_samples(path: Path) -> pd.DataFrame:
    """Lightweight GEO family-SOFT parser -> DataFrame(index=GSM) with title/source/characteristics."""
    rows: List[Dict[str, str]] = []
    cur: Optional[Dict[str, str]] = None
    skip_table = False
    with open_text(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE"):
                cur = {"gsm": line.split("=", 1)[1].strip()}
                rows.append(cur)
                skip_table = False
            elif line.startswith("^"):
                cur = None
            elif cur is None:
                continue
            elif line.startswith("!sample_table_begin"):
                skip_table = True
            elif line.startswith("!sample_table_end"):
                skip_table = False
            elif skip_table:
                continue
            elif line.startswith("!Sample_") and "=" in line:
                key, val = line[1:].split("=", 1)
                key, val = clean_name(key.replace("Sample_", "")), val.strip()
                if key.startswith("characteristics"):
                    if ":" in val:
                        k2, v2 = val.split(":", 1)
                        cur[clean_name(k2)] = v2.strip()
                    else:
                        cur.setdefault("characteristics_misc", val)
                elif key in cur:
                    cur[key] = cur[key] + " | " + val
                else:
                    cur[key] = val
    if not rows:
        raise ValueError(f"No ^SAMPLE records found in {path}")
    return pd.DataFrame(rows).set_index("gsm")


def parse_series_matrix(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """GEO series matrix -> (expression probes x samples, sample metadata, platform id)."""
    meta_lines: Dict[str, List[List[str]]] = {}
    table: List[str] = []
    platform = None
    in_table = False
    with open_text(path) as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                table.append(line)
            elif line.startswith("!Sample_"):
                parts = line.rstrip("\n").split("\t")
                meta_lines.setdefault(parts[0], []).append([p.strip().strip('"') for p in parts[1:]])
            elif line.startswith("!Series_platform_id"):
                platform = line.rstrip("\n").split("\t")[1].strip().strip('"')
    expr = pd.read_csv(io.StringIO("".join(table)), sep="\t", index_col=0, quotechar='"')
    expr.index = expr.index.astype(str)
    samples = expr.columns.tolist()
    meta = pd.DataFrame(index=samples)
    for key, blocks in meta_lines.items():
        base = clean_name(key.replace("!Sample_", ""))
        for j, vals in enumerate(blocks):
            if len(vals) != len(samples):
                continue
            if base.startswith("characteristics"):
                kv = [v.split(":", 1) if ":" in v else ["", v] for v in vals]
                name = clean_name(kv[0][0]) or f"{base}_{j}"
                meta[name] = [x[1].strip() for x in kv]
            else:
                meta[base if j == 0 else f"{base}_{j}"] = vals
    return expr.apply(pd.to_numeric, errors="coerce"), meta, platform


def parse_platform_table(path: Path) -> pd.DataFrame:
    """Read the tabular part of a GEO platform SOFT/annot file."""
    lines, on = [], False
    with open_text(path) as fh:
        for line in fh:
            if line.startswith("!platform_table_begin"):
                on = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if on:
                lines.append(line)
    if not lines:
        raise ValueError(f"No platform table in {path}")
    return pd.read_csv(io.StringIO("".join(lines)), sep="\t", dtype=str, quoting=3, on_bad_lines="skip")


def probes_to_symbols(platform: pd.DataFrame) -> pd.Series:
    """Return Series probe_id -> gene symbol from a GEO platform table (handles Affymetrix gene_assignment)."""
    id_col = find_column(platform.columns, ["ID", "ID_REF", "probe_id"], allow_partial=False) or platform.columns[0]
    sym_col = find_column(platform.columns, ["Gene Symbol", "gene_symbol", "GENE_SYMBOL", "Symbol"], allow_partial=False)
    if sym_col is not None:
        sym = platform[sym_col].fillna("").astype(str).str.split(r"\s*///\s*").str[0].str.strip()
    else:
        ga = find_column(platform.columns, ["gene_assignment"], allow_partial=False)
        if ga is None:
            raise ValueError("Platform table has no gene-symbol or gene_assignment column")
        def _sym(s: str) -> str:
            first = str(s).split("///")[0].split("//")
            return first[1].strip() if len(first) > 1 else ""
        sym = platform[ga].fillna("").map(_sym)
    out = pd.Series(sym.values, index=platform[id_col].astype(str).values)
    out = out[(out != "") & (out != "---") & (out.str.lower() != "nan")]
    return out[~out.index.duplicated()]


# ------------------------------------------------------------ plotting helpers
def heatmap(ax, mat: np.ndarray, xlabels: Sequence[str], ylabels: Sequence[str], vmin=-1, vmax=1,
            cmap="RdBu_r", cbar_label: str = "", annot: bool = False, fmt: str = "{:.2f}"):
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    if annot:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                dark = abs(mat[i, j]) > 0.6 * max(abs(vmin), abs(vmax))
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=6, color="white" if dark else "black")
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.03)
    return im
