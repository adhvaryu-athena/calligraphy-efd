"""Stage 4 — Classification, with grid search + permutation importance.

Three classifiers (nearest centroid, random forest, PCA + logistic regression) under
two cross-validation protocols (per-glyph stratified k-fold, per-font GroupKFold).
Bootstrap 95% CI on macro-F1.

Two additions over the calibration version:
  1. Random forest gets GridSearchCV over max_depth and min_samples_leaf
     (PRD: {5, 10, None} x {1, 2, 5}). Best config picked per CV fold.
  2. Feature importance reports BOTH:
       - MDI (mean decrease in impurity), the default, biased toward high-cardinality
       - Permutation importance (Strobl 2007 — more honest)
     Both written to results JSON; visualize.py plots them side-by-side.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# PRD's RF grid
RF_PARAM_GRID = {
    "max_depth": [5, 10, None],
    "min_samples_leaf": [1, 2, 5],
}


def make_classifiers(seed: int = 42, grid_search_rf: bool = True):
    """Return classifier instances. RF wraps a GridSearchCV when grid_search_rf=True."""
    rf_base = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    if grid_search_rf:
        rf = GridSearchCV(rf_base, RF_PARAM_GRID, cv=3, scoring="f1_macro", n_jobs=-1)
    else:
        rf = rf_base
    return {
        "nearest_centroid": NearestCentroid(),
        "random_forest": rf,
        "pca_logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=10, random_state=seed)),
            ("lr", LogisticRegression(max_iter=2000, random_state=seed))]),
    }


def _split_iter(X, y, groups, cv_type, n_splits, seed):
    if cv_type == "stratified":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from cv.split(X, y)
    elif cv_type == "group":
        if groups is None:
            raise ValueError("groups required for group CV")
        n_groups = len(np.unique(groups))
        k = min(n_splits, n_groups)
        cv = GroupKFold(n_splits=k)
        yield from cv.split(X, y, groups)
    else:
        raise ValueError(f"Unknown cv_type {cv_type!r}")


def cross_val_predict_safe(clf, X, y, groups, cv_type, n_splits, seed):
    y_pred = np.empty_like(y)
    for tr, te in _split_iter(X, y, groups, cv_type, n_splits, seed):
        c = clone(clf)
        c.fit(X[tr], y[tr])
        y_pred[te] = c.predict(X[te])
    return y_pred


def bootstrap_macro_f1(y_true, y_pred, n_bootstrap: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        f1s[i] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    return float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5)), f1s


def evaluate(
    clf_name: str, clf, X, y, groups=None, cv_type: str = "stratified",
    n_splits: int = 5, seed: int = 42, n_bootstrap: int = 1000,
    compute_permutation: bool = False, perm_n_repeats: int = 10,
):
    y_pred = cross_val_predict_safe(clf, X, y, groups, cv_type, n_splits, seed)
    macro_f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
    ci_low, ci_high, _ = bootstrap_macro_f1(y, y_pred, n_bootstrap=n_bootstrap, seed=seed)
    classes = sorted(np.unique(y).tolist())
    cm = confusion_matrix(y, y_pred, labels=classes)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)

    # Refit on full data for importances
    fitted = clone(clf).fit(X, y)
    # Drill through GridSearchCV wrapper to reach the inner RF
    estimator = getattr(fitted, "best_estimator_", fitted)
    importances_mdi = getattr(estimator, "feature_importances_", None)
    if importances_mdi is not None:
        importances_mdi = importances_mdi.tolist()

    importances_perm = None
    importances_perm_std = None
    if compute_permutation and importances_mdi is not None:
        pi = permutation_importance(
            fitted, X, y, n_repeats=perm_n_repeats,
            random_state=seed, scoring="f1_macro", n_jobs=-1,
        )
        importances_perm = pi.importances_mean.tolist()
        importances_perm_std = pi.importances_std.tolist()

    best_params = getattr(fitted, "best_params_", None)

    return {
        "classifier": clf_name, "cv_type": cv_type, "n_splits": n_splits,
        "n_samples": int(len(y)), "n_classes": len(classes), "classes": classes,
        "macro_f1": macro_f1, "ci_low_95": ci_low, "ci_high_95": ci_high,
        "confusion_matrix": cm.tolist(), "classification_report": report,
        "feature_importances": importances_mdi,
        "feature_importances_perm": importances_perm,
        "feature_importances_perm_std": importances_perm_std,
        "best_params": best_params,
        "y_pred": y_pred.tolist(), "y_true": y.tolist(),
    }


def run_sensitivity(
    feats: Dict, clf_name: str = "random_forest", cv_type: str = "group",
    n_splits: int = 5, seed: int = 42, n_bootstrap: int = 1000,
    verbose: bool = True, grid_search_rf: bool = False,
):
    """Sweep across harmonic orders.

    grid_search_rf defaults to False here — doing grid search at every order is slow.
    The primary-order classification block uses grid search; sensitivity uses defaults.
    """
    out = {}
    for order, bundle in feats.items():
        clf = make_classifiers(seed=seed, grid_search_rf=grid_search_rf)[clf_name]
        res = evaluate(
            clf_name, clf, bundle["X"], bundle["y"], bundle["groups"],
            cv_type=cv_type, n_splits=n_splits, seed=seed,
            n_bootstrap=n_bootstrap, compute_permutation=False,
        )
        out[order] = res
        if verbose:
            print(f"  N={order:3d}  macro-F1={res['macro_f1']:.3f} "
                  f"[{res['ci_low_95']:.3f}, {res['ci_high_95']:.3f}]")
    return out


def run(
    feats: Dict, primary_order: int = 20, n_splits: int = 5, seed: int = 42,
    n_bootstrap: int = 1000, output_path: Optional[Path] = None, verbose: bool = True,
    grid_search_rf: bool = True, compute_permutation: bool = True,
):
    """Full Stage 4 with grid search + permutation importance."""
    bundle = feats[primary_order]
    X, y, groups = bundle["X"], bundle["y"], bundle["groups"]
    results = {
        "primary_order": primary_order, "n_samples": int(len(y)),
        "classes": sorted(set(y.tolist())), "main": {}, "sensitivity": {},
    }

    for clf_name in ("nearest_centroid", "random_forest", "pca_logreg"):
        results["main"][clf_name] = {}
        for cv_type in ("stratified", "group"):
            if verbose:
                print(f"[{clf_name} | {cv_type}]")
            do_grid = grid_search_rf and clf_name == "random_forest"
            do_perm = (compute_permutation and clf_name == "random_forest"
                       and cv_type == "group")
            clf = make_classifiers(seed=seed, grid_search_rf=do_grid)[clf_name]
            res = evaluate(
                clf_name, clf, X, y, groups, cv_type=cv_type,
                n_splits=n_splits, seed=seed, n_bootstrap=n_bootstrap,
                compute_permutation=do_perm,
            )
            results["main"][clf_name][cv_type] = res
            if verbose:
                msg = (f"  macro-F1 = {res['macro_f1']:.3f} "
                       f"[{res['ci_low_95']:.3f}, {res['ci_high_95']:.3f}]")
                if res.get("best_params"):
                    msg += f"   best={res['best_params']}"
                print(msg)

    if verbose:
        print("[sensitivity: random_forest | group CV across orders]")
    sens = run_sensitivity(
        feats, clf_name="random_forest", cv_type="group",
        n_splits=n_splits, seed=seed, n_bootstrap=n_bootstrap,
        verbose=verbose, grid_search_rf=False,
    )
    results["sensitivity"] = {str(k): v for k, v in sens.items()}

    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"Saved classification results to {output_path}")
    return results
