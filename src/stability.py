"""Phase 2a — Cluster stability analysis.

Three analyses:
1. bootstrap_cluster_stability: 80% resampling, 100 bootstraps, ARI between pairs
2. test_aggregation_mode_stability: cross-mode ARI comparison
3. permutation_test_clustering: permutation test against null
"""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

_logger = logging.getLogger(__name__)

MODES = ["mean", "median", "concat", "weighted"]


def bootstrap_cluster_stability(
    feature_matrix: np.ndarray,
    n_bootstrap: int = 100,
    k_range=range(2, 7),
    random_state: int = 42,
) -> Dict[int, float]:
    """Bootstrap cluster stability via 80% resampling.

    For each k in k_range:
    1. Resample 80% of fonts (without replacement) 100 times
    2. Run k-means on each resample
    3. Compute ARI between each pair of resamples that share fonts
    4. Report mean ARI per k

    Returns dict {k: mean_ari}
    """
    rng = np.random.default_rng(random_state)
    n_samples = feature_matrix.shape[0]
    resample_size = int(0.8 * n_samples)

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)

    result: Dict[int, float] = {}

    for k in k_range:
        _logger.info("bootstrap_cluster_stability: k=%d", k)
        # Generate all resamples: list of (indices, labels)
        resample_indices: List[np.ndarray] = []
        resample_labels: List[np.ndarray] = []

        for b in range(n_bootstrap):
            idx = rng.choice(n_samples, size=resample_size, replace=False)
            idx_sorted = np.sort(idx)
            X_sub = X[idx_sorted]
            km = KMeans(n_clusters=k, random_state=random_state + b, n_init=10)
            labels = km.fit_predict(X_sub)
            resample_indices.append(idx_sorted)
            resample_labels.append(labels)

        # Compute ARI for all pairs with overlapping fonts
        ari_scores: List[float] = []
        for i, j in itertools.combinations(range(n_bootstrap), 2):
            idx_i = resample_indices[i]
            idx_j = resample_indices[j]
            # Find intersection of font indices
            shared_idx, pos_i, pos_j = np.intersect1d(
                idx_i, idx_j, return_indices=True
            )
            if len(shared_idx) < 2:
                continue
            labels_i = resample_labels[i][pos_i]
            labels_j = resample_labels[j][pos_j]
            ari = adjusted_rand_score(labels_i, labels_j)
            ari_scores.append(ari)

        mean_ari = float(np.mean(ari_scores)) if ari_scores else float("nan")
        result[k] = mean_ari
        _logger.info("  k=%d: mean_ARI=%.4f (over %d pairs)", k, mean_ari, len(ari_scores))

    return result


def test_aggregation_mode_stability(
    feature_dir: str = "outputs",
    algorithm: str = "kmeans",
    k_range=range(2, 7),
) -> np.ndarray:
    """Cross-mode ARI comparison.

    For each pair of aggregation modes (mean, median, concat, weighted),
    using Config A features:
    1. Run algorithm at each k
    2. Compute ARI between assignments from the two modes

    Returns shape (4, 4, len(k_range)) array. MODES = ["mean","median","concat","weighted"]
    Also saves outputs/stability/mode_ari_matrix.npz
    """
    feature_dir = Path(feature_dir)
    k_list = list(k_range)

    # Load feature matrices for each mode
    mode_data: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    for mode in MODES:
        npz_path = feature_dir / f"font_features_A_{mode}.npz"
        if not npz_path.exists():
            _logger.warning("Missing feature file: %s — skipping mode %s", npz_path, mode)
            continue
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"].astype(float)
        font_names = data["font_names"].tolist()
        mode_data[mode] = (X, font_names)

    available_modes = [m for m in MODES if m in mode_data]
    _logger.info("Loaded modes: %s", available_modes)

    n_modes = len(MODES)
    n_k = len(k_list)
    ari_matrix = np.full((n_modes, n_modes, n_k), np.nan)

    for ki, k in enumerate(k_list):
        # Compute labels for each available mode
        mode_labels: Dict[str, np.ndarray] = {}
        mode_font_names: Dict[str, List[str]] = {}
        for mode in available_modes:
            X_raw, font_names = mode_data[mode]
            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            mode_labels[mode] = labels
            mode_font_names[mode] = font_names

        # Compute pairwise ARI across all mode pairs
        for i, mode_i in enumerate(MODES):
            for j, mode_j in enumerate(MODES):
                if mode_i not in mode_labels or mode_j not in mode_labels:
                    continue
                if i == j:
                    ari_matrix[i, j, ki] = 1.0
                    continue
                # Align to intersection of font names
                fn_i = mode_font_names[mode_i]
                fn_j = mode_font_names[mode_j]
                set_i = {name: idx for idx, name in enumerate(fn_i)}
                set_j = {name: idx for idx, name in enumerate(fn_j)}
                shared = sorted(set(fn_i) & set(fn_j))
                if len(shared) < 2:
                    continue
                idx_i = np.array([set_i[n] for n in shared])
                idx_j = np.array([set_j[n] for n in shared])
                labs_i = mode_labels[mode_i][idx_i]
                labs_j = mode_labels[mode_j][idx_j]
                ari = adjusted_rand_score(labs_i, labs_j)
                ari_matrix[i, j, ki] = ari
                _logger.info(
                    "  k=%d %s vs %s: ARI=%.4f (n_shared=%d)",
                    k, mode_i, mode_j, ari, len(shared),
                )

    # Save results
    out_dir = feature_dir / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "mode_ari_matrix.npz"
    np.savez(
        save_path,
        ari_matrix=ari_matrix,
        modes=np.array(MODES),
        k_values=np.array(k_list),
    )
    _logger.info("Saved mode ARI matrix → %s", save_path)

    return ari_matrix


def permutation_test_clustering(
    feature_matrix: np.ndarray,
    n_permutations: int = 100,
    k: int = 4,
    random_state: int = 42,
) -> Tuple[float, np.ndarray, float]:
    """Permutation test for clustering significance.

    1. Shuffle each feature dimension independently
    2. Run k-means at specified k
    3. Compute silhouette score

    Returns (observed_silhouette, null_distribution, p_value)
    """
    rng = np.random.default_rng(random_state)

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)

    # Observed silhouette
    km_obs = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels_obs = km_obs.fit_predict(X)
    if len(np.unique(labels_obs)) < 2:
        observed_sil = float("nan")
    else:
        observed_sil = float(silhouette_score(X, labels_obs))
    _logger.info("permutation_test_clustering: observed silhouette=%.4f", observed_sil)

    # Null distribution
    null_scores = np.empty(n_permutations)
    for p_idx in range(n_permutations):
        X_perm = X.copy()
        for col in range(X_perm.shape[1]):
            X_perm[:, col] = rng.permutation(X_perm[:, col])
        km_perm = KMeans(n_clusters=k, random_state=random_state + p_idx, n_init=10)
        labels_perm = km_perm.fit_predict(X_perm)
        if len(np.unique(labels_perm)) < 2:
            null_scores[p_idx] = float("nan")
        else:
            null_scores[p_idx] = float(silhouette_score(X_perm, labels_perm))

    valid_null = null_scores[~np.isnan(null_scores)]
    if len(valid_null) == 0 or np.isnan(observed_sil):
        p_value = float("nan")
    else:
        p_value = float(np.mean(valid_null >= observed_sil))

    _logger.info(
        "  null mean=%.4f, std=%.4f, p_value=%.4f",
        float(np.nanmean(null_scores)),
        float(np.nanstd(null_scores)),
        p_value,
    )

    return observed_sil, null_scores, p_value


def run_all_stability(
    feature_dir: str = "outputs",
    output_dir: str = "outputs/stability",
) -> Dict:
    """Orchestrates all three stability analyses and saves results."""
    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict = {}

    # Load Config A / mean features for bootstrap and permutation tests
    npz_path = feature_dir / "font_features_A_mean.npz"
    if not npz_path.exists():
        _logger.error("font_features_A_mean.npz not found in %s — cannot run stability", feature_dir)
        return results

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(float)
    _logger.info("Loaded feature matrix: shape=%s", X.shape)

    # 1. Bootstrap cluster stability
    _logger.info("Running bootstrap cluster stability...")
    bootstrap_result = bootstrap_cluster_stability(X, n_bootstrap=100, k_range=range(2, 7))
    bootstrap_path = output_dir / "bootstrap_ari.json"
    with open(bootstrap_path, "w") as f:
        json.dump({str(k): v for k, v in bootstrap_result.items()}, f, indent=2)
    _logger.info("Saved bootstrap ARI → %s", bootstrap_path)
    results["bootstrap"] = bootstrap_result

    # 2. Mode stability
    _logger.info("Running aggregation mode stability...")
    ari_matrix = test_aggregation_mode_stability(feature_dir=str(feature_dir), k_range=range(2, 7))
    results["mode_stability"] = {
        "shape": list(ari_matrix.shape),
        "modes": MODES,
        "mean_ari_across_k": {
            f"{MODES[i]}_vs_{MODES[j]}": float(np.nanmean(ari_matrix[i, j]))
            for i in range(len(MODES))
            for j in range(len(MODES))
            if i != j
        },
    }

    # 3. Permutation test
    _logger.info("Running permutation test...")
    obs_sil, null_dist, p_value = permutation_test_clustering(X, n_permutations=100, k=4)
    perm_path = output_dir / "permutation_test.json"
    perm_result = {
        "observed_silhouette": float(obs_sil) if not np.isnan(obs_sil) else None,
        "null_mean": float(np.nanmean(null_dist)),
        "null_std": float(np.nanstd(null_dist)),
        "p_value": float(p_value) if not np.isnan(p_value) else None,
        "n_permutations": 100,
        "k": 4,
    }
    with open(perm_path, "w") as f:
        json.dump(perm_result, f, indent=2)
    _logger.info("Saved permutation test → %s", perm_path)
    results["permutation"] = perm_result

    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Phase 2a cluster stability analysis")
    parser.add_argument("--feature-dir", default="outputs")
    parser.add_argument("--output-dir", default="outputs/stability")
    args = parser.parse_args()

    results = run_all_stability(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
    )
    _logger.info("Stability analysis complete. Keys: %s", list(results.keys()))
