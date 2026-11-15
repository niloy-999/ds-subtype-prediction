"""
Reusable modelling components.

* CovariateResidualizer      - age/sex/source residualisation, drop_first=True + fit_intercept=True
                               (no manual intercept column -> no exact collinearity)
* ElasticNetClassifier       - multinomial elastic net (sklearn-version aware wrapper)
* XGBStringClassifier        - XGBoost wrapper for string labels (only if xgboost is installed)
* get_model_specs()          - model zoo; panel size (selector__k) lives INSIDE every param grid
* make_one_se_refit()        - GridSearchCV `refit` callable implementing the one-SE parsimony rule
* make_splits()              - repeated stratified (group-aware if needed) outer splits
* prepare_fold()             - per-outer-fold eligibility filter + variance screen + residualiser,
                               fitted ONCE on the outer-training samples
* PanelModel                 - deployable object (residualiser + scaler + estimator on fixed panel)
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             log_loss, matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

import config as C

_SK = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
_SK_GE_18 = _SK >= (1, 8)

try:  # optional dependency
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:  # noqa: BLE001
    XGBOOST_AVAILABLE = False


# ============================================================================
# 1. Covariate residualiser (leakage- and collinearity-safe)
# ============================================================================
class CovariateResidualizer(BaseEstimator, TransformerMixin):
    """Residualise gene columns on covariates. Fit on TRAINING rows only.

    X must be a DataFrame holding `gene_columns` and `covariate_columns`.
    Design matrix = numeric covariates + get_dummies(categoricals, drop_first=True).
    The intercept is supplied by LinearRegression(fit_intercept=True); NO intercept column is
    inserted manually, so the design is never exactly collinear with a fully one-hot encoding.
    """

    def __init__(self, gene_columns: Sequence[str], covariate_columns: Sequence[str]):
        self.gene_columns = gene_columns
        self.covariate_columns = covariate_columns

    def _design(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        cov = X[list(self.covariate_columns)]
        num_cols = [c for c in cov.columns if pd.api.types.is_numeric_dtype(cov[c])]
        cat_cols = [c for c in cov.columns if c not in num_cols]
        parts: List[pd.DataFrame] = []
        if num_cols:
            num = cov[num_cols].apply(pd.to_numeric, errors="coerce")
            if fit:
                self.numeric_medians_ = num.median().fillna(0.0)
            parts.append(num.fillna(self.numeric_medians_))
        if cat_cols:
            cat = cov[cat_cols].astype(str)
            if fit:
                dummies = pd.get_dummies(cat, drop_first=True, dtype=float)      # <- drop_first=True
                self.dummy_columns_ = dummies.columns.tolist()
            else:
                # Encode with ALL levels, then keep only the training columns. (Using drop_first=True on a
                # test fold would drop a *different* baseline level whenever the fold lacks a level.)
                dummies = pd.get_dummies(cat, drop_first=False, dtype=float).reindex(
                    columns=self.dummy_columns_, fill_value=0.0)
            parts.append(dummies)
        if not parts:
            return np.zeros((len(X), 0))
        design = pd.concat(parts, axis=1)
        if fit:
            keep = design.columns[design.std(ddof=0) > 1e-12]                    # drop constant columns
            self.design_columns_ = keep.tolist()
        return design.reindex(columns=self.design_columns_, fill_value=0.0).to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CovariateResidualizer requires a pandas DataFrame")
        genes = X[list(self.gene_columns)].to_numpy(dtype=float)
        D = self._design(X, fit=True)
        self.n_features_in_ = genes.shape[1]
        if D.shape[1] == 0:
            self.model_ = None
            self.mean_ = genes.mean(axis=0)
        else:
            self.model_ = LinearRegression(fit_intercept=True).fit(D, genes)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["n_features_in_"])
        genes = X[list(self.gene_columns)].to_numpy(dtype=float)
        if self.model_ is None:
            return genes - self.mean_
        return genes - self.model_.predict(self._design(X, fit=False))


# ============================================================================
# 2. Estimators
# ============================================================================
class ElasticNetClassifier(ClassifierMixin, BaseEstimator):
    """Multinomial elastic-net logistic regression (saga). Works across sklearn 1.x APIs."""

    def __init__(self, C: float = 1.0, l1_ratio: float = 0.5, class_weight: Any = "balanced",
                 max_iter: int = 5000, random_state: int = 42):
        self.C = C
        self.l1_ratio = l1_ratio
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        kw = dict(solver="saga", C=self.C, l1_ratio=self.l1_ratio, class_weight=self.class_weight,
                  max_iter=self.max_iter, random_state=self.random_state)
        if not _SK_GE_18:                       # <1.8 needs penalty="elasticnet"; >=1.8 deprecates `penalty`
            kw["penalty"] = "elasticnet"
        self.model_ = LogisticRegression(**kw).fit(X, y)
        self.classes_ = self.model_.classes_
        self.coef_, self.intercept_ = self.model_.coef_, self.model_.intercept_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


class XGBStringClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost wrapper accepting string class labels."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 3, learning_rate: float = 0.05,
                 subsample: float = 0.8, colsample_bytree: float = 0.8, reg_lambda: float = 1.0,
                 random_state: int = 42, n_jobs: int = 1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed")
        self.classes_ = np.asarray(sorted(pd.Series(y).astype(str).unique()))
        enc = {c: i for i, c in enumerate(self.classes_)}
        self.model_ = XGBClassifier(
            objective="multi:softprob", num_class=len(self.classes_), eval_metric="mlogloss",
            n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate,
            subsample=self.subsample, colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda,
            random_state=self.random_state, n_jobs=self.n_jobs, tree_method="hist", verbosity=0)
        self.model_.fit(X, pd.Series(y).astype(str).map(enc).to_numpy())
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.model_.predict(X).astype(int)]


# ============================================================================
# 3. Model zoo -- panel size is a hyper-parameter (selector__k) of EVERY grid
# ============================================================================
def _with_k(grid: Any) -> List[Dict[str, list]]:
    grids = grid if isinstance(grid, list) else [grid]
    return [{**g, "selector__k": list(C.PANEL_SIZES)} for g in grids]


def get_model_specs(names: Sequence[str] | None = None) -> Dict[str, Dict[str, Any]]:
    small = C.P["grid"] == "small"
    seed = C.SEED
    specs: Dict[str, Dict[str, Any]] = {
        "elastic_net": dict(
            estimator=ElasticNetClassifier(random_state=seed),
            grid=_with_k({"model__C": [0.01, 0.1, 1.0] if small else [0.01, 0.1, 1.0, 10.0],
                          "model__l1_ratio": [0.25, 0.75] if small else [0.1, 0.5, 0.9]})),
        "logistic_l2": dict(
            estimator=LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
            grid=_with_k({"model__C": [0.01, 0.1, 1.0] if small else [0.001, 0.01, 0.1, 1.0, 10.0]})),
        "svm": dict(
            estimator=SVC(probability=True, class_weight="balanced", random_state=seed),
            grid=_with_k([{"model__kernel": ["linear"], "model__C": [0.01, 0.1, 1.0]},
                          {"model__kernel": ["rbf"], "model__C": [1.0, 10.0] if small else [0.1, 1.0, 10.0, 100.0],
                           "model__gamma": ["scale"] if small else ["scale", 1e-3, 1e-2]}])),
        "random_forest": dict(
            estimator=RandomForestClassifier(n_estimators=C.P["rf_trees"], class_weight="balanced_subsample",
                                             n_jobs=1, random_state=seed),
            grid=_with_k({"model__max_depth": [None, 6], "model__max_features": ["sqrt", 0.3],
                          "model__min_samples_leaf": [2]})),
    }
    if XGBOOST_AVAILABLE:
        specs["xgboost"] = dict(
            estimator=XGBStringClassifier(random_state=seed, n_jobs=1),
            grid=_with_k({"model__n_estimators": [150] if small else [150, 300],
                          "model__max_depth": [2, 4], "model__learning_rate": [0.05, 0.1]}))
    names = names or list(specs)
    return {n: specs[n] for n in names if n in specs}


def build_pipeline(estimator) -> Pipeline:
    """selector -> scaler -> model; every step is fitted inside the CV training split only."""
    return Pipeline([("selector", SelectKBest(f_classif, k=10)),
                     ("scaler", StandardScaler()),
                     ("model", clone(estimator))])


def make_one_se_refit(k_param: str = "selector__k"):
    """GridSearchCV `refit` callable -> one-standard-error parsimony rule.

    Best configuration = highest mean inner-CV macro-F1. Every configuration whose mean is within one
    standard error of it is eligible; the eligible configuration with the SMALLEST panel (selector__k)
    wins (ties -> higher mean score). Because panel size is part of the grid, this is done in a single
    GridSearchCV per outer fold instead of an outer Python loop over panel sizes.
    """
    def _refit(cv_results: Dict[str, Any]) -> int:
        mean = np.asarray(cv_results["mean_test_score"], dtype=float)
        std = np.asarray(cv_results["std_test_score"], dtype=float)
        n_splits = sum(1 for k in cv_results if k.startswith("split") and k.endswith("_test_score"))
        best = int(np.nanargmax(mean))
        se = std[best] / np.sqrt(max(n_splits, 1))
        eligible = np.where(mean >= mean[best] - se)[0]
        ks = np.array([cv_results["params"][i][k_param] for i in eligible])
        order = np.lexsort((-mean[eligible], ks))
        return int(eligible[order[0]])
    return _refit


# ============================================================================
# 4. Outer splits, metrics, per-fold preparation
# ============================================================================
def make_splits(y: pd.Series, groups: pd.Series | None, n_splits: int, n_repeats: int,
                seed: int) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield (repeat, fold, train_idx, test_idx). Group-aware when participants repeat."""
    grouped = groups is not None and groups.duplicated().any()
    for r in range(n_repeats):
        if grouped:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
            it = cv.split(np.zeros(len(y)), y, groups)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
            it = cv.split(np.zeros(len(y)), y)
        for f, (tr, te) in enumerate(it):
            yield r, f, tr, te


def multiclass_brier(y_true, proba, classes) -> float:
    onehot = label_binarize(y_true, classes=list(classes))
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def evaluate_predictions(y_true, y_pred, proba, classes) -> Dict[str, float]:
    m = {"accuracy": accuracy_score(y_true, y_pred),
         "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
         "macro_f1": f1_score(y_true, y_pred, average="macro"),
         "mcc": matthews_corrcoef(y_true, y_pred),
         "log_loss": log_loss(y_true, proba, labels=list(classes)),
         "brier": multiclass_brier(y_true, proba, classes)}
    yb = label_binarize(y_true, classes=list(classes))
    try:
        m["roc_auc_ovr_macro"] = roc_auc_score(yb, proba, average="macro")
        m["pr_auc_macro"] = average_precision_score(yb, proba, average="macro")
    except ValueError:
        m["roc_auc_ovr_macro"] = m["pr_auc_macro"] = np.nan
    return m


def align_proba(proba: np.ndarray, est_classes: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    out = np.zeros((proba.shape[0], len(classes)))
    for j, c in enumerate(est_classes):
        out[:, list(classes).index(c)] = proba[:, j]
    return out


def prepare_fold(logcpm: pd.DataFrame, expressed: pd.DataFrame, cov: pd.DataFrame,
                 tr: np.ndarray, te: np.ndarray, prefilter_top: int, min_frac: float):
    """Everything that must be learned from the OUTER-TRAINING samples, done once per outer fold:
       1. expression-eligibility filter (count>=MIN_COUNT in >=min_frac of training samples)
       2. unsupervised variance screen (top `prefilter_top` genes)
       3. covariate residualiser (age / sex / source)  -- fitted once, applied to train and test.
    Returns residualised train/test matrices, gene names and the fitted residualiser."""
    eligible = expressed.iloc[tr].mean(axis=0) >= min_frac
    genes = eligible.index[eligible.to_numpy()]
    var = logcpm.iloc[tr][genes].var(axis=0)
    keep = var.nlargest(min(prefilter_top, len(var))).index.tolist()
    cov_cols = list(cov.columns)
    frame_tr = pd.concat([logcpm.iloc[tr][keep], cov.iloc[tr]], axis=1)
    frame_te = pd.concat([logcpm.iloc[te][keep], cov.iloc[te]], axis=1)
    res = CovariateResidualizer(keep, cov_cols).fit(frame_tr)
    return res.transform(frame_tr), res.transform(frame_te), keep, res


def selected_features(pipe: Pipeline, gene_names: Sequence[str]) -> pd.DataFrame:
    sel = pipe.named_steps["selector"]
    mask = sel.get_support()
    names = np.asarray(gene_names, dtype=object)[mask]
    scores = sel.scores_[mask]
    order = np.argsort(-scores)
    return pd.DataFrame({"gene_id": names[order], "rank": np.arange(1, len(order) + 1), "f_score": scores[order]})


# ============================================================================
# 5. Deployable panel model
# ============================================================================
class PanelModel:
    """Residualiser + scaler + estimator on a FIXED gene panel (fitted on all 304 T21 samples)."""

    def __init__(self, genes: Sequence[str], residualizer: CovariateResidualizer, scaler: StandardScaler,
                 estimator, model_name: str, params: Dict[str, Any]):
        self.genes, self.residualizer, self.scaler = list(genes), residualizer, scaler
        self.estimator, self.model_name, self.params = estimator, model_name, params
        self.classes_ = np.asarray(estimator.classes_)

    def transform(self, logcpm: pd.DataFrame, cov: pd.DataFrame) -> np.ndarray:
        frame = pd.concat([logcpm[self.genes], cov], axis=1)
        return self.scaler.transform(self.residualizer.transform(frame))

    def predict_proba(self, logcpm: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
        p = self.estimator.predict_proba(self.transform(logcpm, cov))
        return pd.DataFrame(p, index=logcpm.index, columns=self.classes_)
