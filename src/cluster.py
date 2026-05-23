"""Phase 1a — Unsupervised clustering across all feature variants.

Loads all font_features_{config}_{mode}.npz files and runs:
  - K-means (k=2..10): silhouette, Davies-Bouldin, gap statistic
  - Hierarchical (Ward): full linkage matrix, labels at k=2..6
  - HDBSCAN (min_cluster_size=5,10,15)
  - GMM (k=2..10): BIC, labels at BIC-optimal k

Saves:
  outputs/clustering_results.json
  outputs/dendrograms/{config}_{mode}.npz
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan as _hdbscan_mod
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gap statistic (Tibshirani 2001)
# ---------------------------------------------------------------------------

def _within_cluster_dispersion(X: np.ndarray, labels: np.ndarray) -> float:
    """Sum of pairwise squared distances within each cluster / (2 * n_k)."""
    W = 0.0
    for k in np.unique(labels):
        mask = labels == k
        Xk = X[mask]
        n_k = Xk.shape[0]
        if n_k < 2:
            continue
        # ||x_i - x_j||^2 = 2 * n_k * var  (cheaper than full pairwise)
        W += np.sum(np.var(Xk, axis=0)) * n_k
    return W


def _gap_statistic(
    X: np.ndarray,
    k_range: range,
    B: int = 10,
    random_state: int = 42,
) -> Dict[int, float]:
    """Compute gap(k) for each k in k_range.

    gap(k) = E*[log W_k] - log W_k
    Reference distribution: uniform over [min, max] of each feature.
    """
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    col_mins = X.min(axis=0)
    col_maxs = X.max(axis=0)

    log_W_ref = {k: [] for k in k_range}

    for _ in range(B):
        X_ref = rng.uniform(size=(n, p)) * (col_maxs - col_mins) + col_mins
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=3)
            labels_ref = km.fit_predict(X_ref)
            W_ref = _within_cluster_dispersion(X_ref, labels_ref)
            log_W_ref[k].append(np.log(max(W_ref, 1e-10)))

    gaps: Dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=5)
        labels = km.fit_predict(X)
        W = _within_cluster_dispersion(X, labels)
        gap = float(np.mean(log_W_ref[k]) - np.log(max(W, 1e-10)))
        gaps[k] = gap

    return gaps


# ---------------------------------------------------------------------------
# Clustering runners
# ---------------------------------------------------------------------------

def _run_kmeans(
    X: np.ndarray,
    k_range: range,
    gap_values: Dict[int, float],
    random_state: int = 42,
) -> Dict:
    result = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        try:
            sil = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else float("nan")
            db = float(davies_bouldin_score(X, labels)) if len(np.unique(labels)) > 1 else float("nan")
        except Exception as exc:
            _logger.warning("Metrics failed for k=%d: %s", k, exc)
            sil, db = float("nan"), float("nan")
        result[f"k_{k}"] = {
            "silhouette": sil,
            "davies_bouldin": db,
            "gap": gap_values.get(k, float("nan")),
            "labels": labels.tolist(),
        }
    return result


def _run_hierarchical(
    X: np.ndarray,
    Z: np.ndarray,
    k_range: range = range(2, 7),
) -> Dict:
    result = {}
    for k in k_range:
        labels = fcluster(Z, k, criterion="maxclust")
        result[f"k_{k}"] = {"labels": labels.tolist()}
    return result


def _run_hdbscan(X: np.ndarray, min_sizes: Tuple[int, ...] = (5, 10, 15)) -> Dict:
    if not _HDBSCAN_AVAILABLE:
        _logger.warning("hdbscan not installed — skipping HDBSCAN clustering")
        return {}
    result = {}
    for ms in min_sizes:
        clusterer = _hdbscan_mod.HDBSCAN(min_cluster_size=ms)
        labels = clusterer.fit_predict(X)
        unique_labels = set(labels)
        n_noise = int(np.sum(labels == -1))
        n_clusters = len(unique_labels - {-1})
        noise_frac = n_noise / max(len(labels), 1)
        result[f"min_{ms}"] = {
            "n_clusters": n_clusters,
            "noise_fraction": float(noise_frac),
            "labels": labels.tolist(),
        }
    return result


def _run_gmm(
    X: np.ndarray,
    k_range: range,
    random_state: int = 42,
) -> Dict:
    bic_scores: Dict[int, float] = {}
    all_labels: Dict[str, List[int]] = {}
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=3)
        gmm.fit(X)
        bic = float(gmm.bic(X))
        bic_scores[k] = bic
        all_labels[f"k_{k}"] = gmm.predict(X).tolist()

    optimal_k = min(bic_scores, key=bic_scores.__getitem__)
    result = {"optimal_k": int(optimal_k)}
    for k in k_range:
        result[f"k_{k}"] = {
            "bic": bic_scores[k],
            "labels": all_labels[f"k_{k}"],
        }
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_all_clustering(
    font_feature_dir: str | Path = "outputs",
    k_range: range = range(2, 11),
    random_state: int = 42,
    gap_B: int = 10,
) -> Dict:
    """Run all clustering methods for every font_features_*.npz file found.

    Parameters
    ----------
    font_feature_dir : directory containing font_features_{config}_{mode}.npz files
    k_range          : range of k values for k-means and GMM
    random_state     : RNG seed
    gap_B            : number of reference datasets for gap statistic

    Returns
    -------
    Nested dict matching the results JSON structure.
    """
    font_feature_dir = Path(font_feature_dir)
    out_dir = font_feature_dir
    dendrogram_dir = out_dir / "dendrograms"
    dendrogram_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(font_feature_dir.glob("font_features_*.npz"))
    if not npz_files:
        _logger.warning("No font_features_*.npz files found in %s", font_feature_dir)
        return {}

    all_results: Dict = {}

    for npz_path in npz_files:
        # Parse variant key from filename: font_features_{config}_{mode}.npz
        stem = npz_path.stem  # e.g. "font_features_A_mean"
        parts = stem.split("_", 2)  # ["font", "features", "A_mean"]
        if len(parts) < 3:
            _logger.warning("Unexpected filename format: %s — skipping", npz_path.name)
            continue
        variant_key = parts[2]  # e.g. "A_mean"

        _logger.info("Processing variant: %s", variant_key)

        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as exc:
            _logger.warning("Failed to load %s: %s", npz_path, exc)
            continue

        if "X" not in data:
            _logger.warning("No 'X' array in %s — skipping", npz_path.name)
            continue

        X_raw = data["X"].astype(float)
        font_names = data["font_names"].tolist() if "font_names" in data else []
        labels_arr = data["labels"].tolist() if "labels" in data else []

        # StandardScaler normalisation
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        n_samples, n_features = X.shape
        _logger.info("  shape=%s", X.shape)

        # Clamp k_range to valid values
        valid_k = range(max(k_range.start, 2), min(k_range.stop, n_samples))
        if len(valid_k) == 0:
            _logger.warning("Not enough samples (%d) for clustering — skipping", n_samples)
            continue

        # --- Gap statistic (shared across k-means and returned per k) ---
        _logger.info("  Computing gap statistic (B=%d)...", gap_B)
        try:
            gap_values = _gap_statistic(X, valid_k, B=gap_B, random_state=random_state)
        except Exception as exc:
            _logger.warning("Gap statistic failed: %s", exc)
            gap_values = {}

        # --- K-means ---
        _logger.info("  Running K-means...")
        try:
            kmeans_res = _run_kmeans(X, valid_k, gap_values, random_state=random_state)
        except Exception as exc:
            _logger.warning("K-means failed: %s", exc)
            kmeans_res = {}

        # --- Hierarchical (Ward) ---
        _logger.info("  Running hierarchical clustering...")
        try:
            Z = linkage(X, method="ward")
            hier_k_range = range(2, min(7, n_samples))
            hier_res = _run_hierarchical(X, Z, k_range=hier_k_range)
            # Save dendrogram linkage matrix
            dend_path = dendrogram_dir / f"{variant_key}.npz"
            np.savez(dend_path, Z=Z)
            _logger.info("  Saved dendrogram → %s", dend_path)
        except Exception as exc:
            _logger.warning("Hierarchical clustering failed: %s", exc)
            hier_res = {}

        # --- HDBSCAN ---
        _logger.info("  Running HDBSCAN...")
        try:
            hdbscan_res = _run_hdbscan(X, min_sizes=(5, 10, 15))
        except Exception as exc:
            _logger.warning("HDBSCAN failed: %s", exc)
            hdbscan_res = {}

        # --- GMM ---
        _logger.info("  Running GMM...")
        try:
            gmm_res = _run_gmm(X, valid_k, random_state=random_state)
        except Exception as exc:
            _logger.warning("GMM failed: %s", exc)
            gmm_res = {}

        all_results[variant_key] = {
            "font_names": font_names,
            "true_labels": labels_arr,
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "kmeans": kmeans_res,
            "hierarchical": hier_res,
            "hdbscan": hdbscan_res,
            "gmm": gmm_res,
        }

    # Save JSON results
    results_path = out_dir / "clustering_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    _logger.info("Saved clustering results → %s", results_path)

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Phase 1a clustering (Config A / mean only for speed)")
    parser.add_argument("--font-feature-dir", default="outputs", help="Directory with font_features_*.npz files")
    parser.add_argument("--config", default="A", help="Config to run (default: A)")
    parser.add_argument("--mode", default="mean", help="Aggregation mode (default: mean)")
    args = parser.parse_args()

    font_feature_dir = Path(args.font_feature_dir)
    # For __main__, restrict to Config A / mean only
    target_file = font_feature_dir / f"font_features_{args.config}_{args.mode}.npz"
    if not target_file.exists():
        _logger.error("File not found: %s", target_file)
        raise SystemExit(1)

    # Temporarily scope to just this one file by creating a temp view
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        shutil.copy(target_file, tmpdir_path / target_file.name)
        results = run_all_clustering(font_feature_dir=tmpdir_path, k_range=range(2, 11), random_state=42)
        # Also save dendrograms to real output dir
        real_dend_dir = font_feature_dir / "dendrograms"
        real_dend_dir.mkdir(parents=True, exist_ok=True)
        tmp_dend_dir = tmpdir_path / "dendrograms"
        if tmp_dend_dir.exists():
            for f in tmp_dend_dir.iterdir():
                shutil.copy(f, real_dend_dir / f.name)

    # Save results to real output dir
    results_path = font_feature_dir / "clustering_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    _logger.info("Done. Results → %s", results_path)
