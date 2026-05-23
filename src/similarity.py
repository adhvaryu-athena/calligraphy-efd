"""Phase 1c — Pairwise distance matrices and k-nearest neighbors."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler

_logger = logging.getLogger(__name__)

_METRICS = ("cosine", "euclidean", "mahalanobis")


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def _cosine_dist(X: np.ndarray) -> np.ndarray:
    return cosine_distances(X)


def _euclidean_dist(X: np.ndarray) -> np.ndarray:
    return cdist(X, X, metric="euclidean")


def _mahalanobis_dist(X: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    """Pairwise Mahalanobis distances with regularised precision matrix."""
    p = X.shape[1]
    cov = np.cov(X.T)
    # Regularise to ensure positive-definite
    reg_cov = cov + alpha * np.eye(p)
    VI = np.linalg.pinv(reg_cov)
    return cdist(X, X, metric="mahalanobis", VI=VI)


def _compute_distance_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        return _cosine_dist(X)
    elif metric == "euclidean":
        return _euclidean_dist(X)
    elif metric == "mahalanobis":
        return _mahalanobis_dist(X)
    else:
        raise ValueError(f"Unknown metric: {metric!r}")


# ---------------------------------------------------------------------------
# k-NN extraction
# ---------------------------------------------------------------------------

def _get_neighbors(
    D: np.ndarray,
    font_names: List[str],
    labels: List[str],
    k_neighbors: Tuple[int, ...],
) -> List[Dict]:
    """Extract k nearest neighbors for each font from distance matrix D."""
    n = D.shape[0]
    max_k = max(k_neighbors)
    rows = []
    for i in range(n):
        row_d = D[i].copy()
        row_d[i] = np.inf  # exclude self
        sorted_idx = np.argsort(row_d)
        neighbors_ordered = [font_names[j] for j in sorted_idx[:max_k]]
        record: Dict = {
            "font_name": font_names[i],
            "label": labels[i] if labels else "",
        }
        for k in k_neighbors:
            for rank in range(1, k + 1):
                col = f"rank_{rank}"
                if col not in record:
                    record[col] = neighbors_ordered[rank - 1] if rank - 1 < len(neighbors_ordered) else ""
        rows.append(record)
    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_similarity(
    font_feature_dir: str | Path = "outputs",
    k_neighbors: Tuple[int, ...] = (5, 10, 20),
) -> Dict:
    """Compute pairwise distance matrices and k-NN for all font_features_*.npz files.

    Parameters
    ----------
    font_feature_dir : directory containing font_features_{config}_{mode}.npz
    k_neighbors      : tuple of k values for nearest neighbors

    Returns
    -------
    Nested dict: {variant_key: {metric: {"pairwise_path": str, "neighbors_path": str}}}
    """
    font_feature_dir = Path(font_feature_dir)
    sim_dir = font_feature_dir / "similarity"
    sim_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(font_feature_dir.glob("font_features_*.npz"))
    if not npz_files:
        _logger.warning("No font_features_*.npz files found in %s", font_feature_dir)
        return {}

    all_results: Dict = {}

    for npz_path in npz_files:
        stem = npz_path.stem  # e.g. "font_features_A_mean"
        parts = stem.split("_", 2)
        if len(parts) < 3:
            _logger.warning("Unexpected filename format: %s — skipping", npz_path.name)
            continue
        variant_key = parts[2]  # e.g. "A_mean"

        _logger.info("Processing similarity for variant: %s", variant_key)

        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as exc:
            _logger.warning("Failed to load %s: %s", npz_path, exc)
            continue

        if "X" not in data:
            _logger.warning("No 'X' array in %s — skipping", npz_path.name)
            continue

        X_raw = data["X"].astype(float)
        font_names = data["font_names"].tolist() if "font_names" in data else [str(i) for i in range(X_raw.shape[0])]
        labels = data["labels"].tolist() if "labels" in data else [""] * X_raw.shape[0]

        # StandardScaler for euclidean and Mahalanobis; cosine uses raw scaled
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        variant_result: Dict = {}

        for metric in _METRICS:
            _logger.info("  Computing %s distances...", metric)
            try:
                D = _compute_distance_matrix(X, metric)
            except Exception as exc:
                _logger.warning("Distance computation failed for metric=%s variant=%s: %s", metric, variant_key, exc)
                continue

            # Save pairwise matrix
            pairwise_fname = f"pairwise_{variant_key}_{metric}.npz"
            pairwise_path = sim_dir / pairwise_fname
            np.savez(
                pairwise_path,
                D=D,
                font_names=np.array(font_names),
                labels=np.array(labels),
            )
            _logger.info("    Saved pairwise → %s", pairwise_path)

            # Compute and save k-NN CSV
            neighbors_fname = f"neighbors_{variant_key}_{metric}.csv"
            neighbors_path = sim_dir / neighbors_fname
            try:
                neighbor_rows = _get_neighbors(D, font_names, labels, k_neighbors)
                max_k = max(k_neighbors)
                fieldnames = ["font_name", "label"] + [f"rank_{r}" for r in range(1, max_k + 1)]
                with open(neighbors_path, "w", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(neighbor_rows)
                _logger.info("    Saved neighbors CSV → %s", neighbors_path)
            except Exception as exc:
                _logger.warning("Failed to write neighbors CSV for %s %s: %s", variant_key, metric, exc)

            variant_result[metric] = {
                "pairwise_path": str(pairwise_path),
                "neighbors_path": str(neighbors_path),
            }

        all_results[variant_key] = variant_result

    import json
    meta_path = sim_dir / "similarity_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_results, f, indent=2)
    _logger.info("Saved similarity metadata → %s", meta_path)

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Phase 1c pairwise similarity")
    parser.add_argument("--font-feature-dir", default="outputs", help="Directory with font_features_*.npz")
    parser.add_argument("--k-neighbors", type=int, nargs="+", default=[5, 10, 20])
    args = parser.parse_args()

    results = compute_similarity(
        font_feature_dir=args.font_feature_dir,
        k_neighbors=tuple(args.k_neighbors),
    )
    _logger.info("Done. Processed %d variants.", len(results))
