"""Phase 1b — Dimensionality reduction embeddings.

For each feature variant:
  1. PCA: top-10 components + explained variance ratios
  2. t-SNE: perplexity in {15, 30, 50}, n_iter=1000
  3. UMAP: n_neighbors in {5,15,30}, min_dist in {0.0, 0.1, 0.5}

Canonical embeddings saved to outputs/embeddings/canonical/
All embeddings saved to outputs/embeddings/
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap as _umap_mod
    _UMAP_AVAILABLE = True
except ImportError:
    try:
        import umap.umap_ as _umap_mod  # type: ignore
        _UMAP_AVAILABLE = True
    except ImportError:
        _UMAP_AVAILABLE = False

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual embedding functions
# ---------------------------------------------------------------------------

def _run_pca(
    X: np.ndarray,
    font_names: List[str],
    labels: List[str],
    n_components: int,
    random_state: int,
    out_dir: Path,
    variant_key: str,
    is_canonical: bool = False,
    canonical_dir: Optional[Path] = None,
) -> Dict:
    """Fit PCA and save embedding."""
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    embedding = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_.tolist()

    params_str = f"n{n_components}"
    fname = f"{variant_key}_pca_{params_str}.npz"
    save_path = out_dir / fname
    np.savez(
        save_path,
        embedding=embedding,
        font_names=np.array(font_names),
        labels=np.array(labels),
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
    _logger.info("Saved PCA embedding → %s", save_path)

    if is_canonical and canonical_dir is not None:
        canonical_path = canonical_dir / fname
        np.savez(
            canonical_path,
            embedding=embedding,
            font_names=np.array(font_names),
            labels=np.array(labels),
            explained_variance_ratio=pca.explained_variance_ratio_,
        )
        _logger.info("Saved canonical PCA → %s", canonical_path)

    return {
        "method": "pca",
        "params": {"n_components": n_components},
        "explained_variance_ratio": explained_variance,
        "cumulative_variance": float(np.sum(pca.explained_variance_ratio_)),
        "path": str(save_path),
    }


def _run_tsne(
    X: np.ndarray,
    font_names: List[str],
    labels: List[str],
    perplexity: float,
    n_iter: int,
    random_state: int,
    out_dir: Path,
    variant_key: str,
    is_canonical: bool = False,
    canonical_dir: Optional[Path] = None,
) -> Dict:
    """Fit t-SNE and save embedding."""
    import sklearn
    from packaging.version import Version

    # t-SNE requires perplexity < n_samples; clamp if needed
    n_samples = X.shape[0]
    actual_perplexity = min(perplexity, n_samples - 1)
    if actual_perplexity != perplexity:
        _logger.warning(
            "t-SNE perplexity clamped from %.0f to %.0f (n_samples=%d)",
            perplexity, actual_perplexity, n_samples,
        )

    # sklearn >= 1.2 renamed n_iter → max_iter
    try:
        _sklearn_ver = Version(sklearn.__version__)
        _iter_kwarg = "max_iter" if _sklearn_ver >= Version("1.2") else "n_iter"
    except Exception:
        _iter_kwarg = "max_iter"

    tsne = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        **{_iter_kwarg: n_iter},
        random_state=random_state,
    )
    embedding = tsne.fit_transform(X)

    params_str = f"perp{int(perplexity)}_iter{n_iter}"
    fname = f"{variant_key}_tsne_{params_str}.npz"
    save_path = out_dir / fname
    np.savez(
        save_path,
        embedding=embedding,
        font_names=np.array(font_names),
        labels=np.array(labels),
    )
    _logger.info("Saved t-SNE embedding → %s", save_path)

    if is_canonical and canonical_dir is not None:
        canonical_path = canonical_dir / fname
        np.savez(
            canonical_path,
            embedding=embedding,
            font_names=np.array(font_names),
            labels=np.array(labels),
        )
        _logger.info("Saved canonical t-SNE → %s", canonical_path)

    return {
        "method": "tsne",
        "params": {"perplexity": actual_perplexity, "n_iter": n_iter},
        "path": str(save_path),
    }


def _run_umap(
    X: np.ndarray,
    font_names: List[str],
    labels: List[str],
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    out_dir: Path,
    variant_key: str,
    is_canonical: bool = False,
    canonical_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """Fit UMAP and save embedding. Returns None if UMAP is unavailable."""
    if not _UMAP_AVAILABLE:
        _logger.warning("umap-learn not installed — skipping UMAP embeddings")
        return None

    reducer = _umap_mod.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X)

    min_dist_str = str(min_dist).replace(".", "p")
    params_str = f"nn{n_neighbors}_md{min_dist_str}"
    fname = f"{variant_key}_umap_{params_str}.npz"
    save_path = out_dir / fname
    np.savez(
        save_path,
        embedding=embedding,
        font_names=np.array(font_names),
        labels=np.array(labels),
    )
    _logger.info("Saved UMAP embedding → %s", save_path)

    if is_canonical and canonical_dir is not None:
        canonical_path = canonical_dir / fname
        np.savez(
            canonical_path,
            embedding=embedding,
            font_names=np.array(font_names),
            labels=np.array(labels),
        )
        _logger.info("Saved canonical UMAP → %s", canonical_path)

    return {
        "method": "umap",
        "params": {"n_neighbors": n_neighbors, "min_dist": min_dist},
        "path": str(save_path),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_embeddings(
    font_feature_dir: str | Path = "outputs",
    random_state: int = 42,
    tsne_n_iter: int = 1000,
    pca_n_components: int = 10,
) -> Dict:
    """Compute all embeddings for every font_features_*.npz file.

    Parameters
    ----------
    font_feature_dir  : directory containing font_features_{config}_{mode}.npz
    random_state      : RNG seed
    tsne_n_iter       : t-SNE iterations
    pca_n_components  : number of PCA components (max)

    Returns
    -------
    Dict mapping variant_key → list of embedding metadata dicts.
    """
    font_feature_dir = Path(font_feature_dir)
    embed_dir = font_feature_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir = embed_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    # Canonical parameter settings
    CANONICAL_PCA_N = pca_n_components
    CANONICAL_TSNE_PERPLEXITY = 30
    CANONICAL_UMAP_N_NEIGHBORS = 15
    CANONICAL_UMAP_MIN_DIST = 0.1

    tsne_perplexities = (15, 30, 50)
    umap_n_neighbors_list = (5, 15, 30)
    umap_min_dists = (0.0, 0.1, 0.5)

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
        variant_key = parts[2]

        _logger.info("Computing embeddings for variant: %s", variant_key)

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
        labels = data["labels"].tolist() if "labels" in data else []

        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        variant_results: List[Dict] = []

        # --- PCA ---
        _logger.info("  PCA (n_components=%d)", CANONICAL_PCA_N)
        try:
            pca_res = _run_pca(
                X, font_names, labels,
                n_components=CANONICAL_PCA_N,
                random_state=random_state,
                out_dir=embed_dir,
                variant_key=variant_key,
                is_canonical=True,
                canonical_dir=canonical_dir,
            )
            variant_results.append(pca_res)
        except Exception as exc:
            _logger.warning("PCA failed for %s: %s", variant_key, exc)

        # --- t-SNE ---
        for perp in tsne_perplexities:
            is_canon = (perp == CANONICAL_TSNE_PERPLEXITY)
            _logger.info("  t-SNE perplexity=%d (canonical=%s)", perp, is_canon)
            try:
                tsne_res = _run_tsne(
                    X, font_names, labels,
                    perplexity=float(perp),
                    n_iter=tsne_n_iter,
                    random_state=random_state,
                    out_dir=embed_dir,
                    variant_key=variant_key,
                    is_canonical=is_canon,
                    canonical_dir=canonical_dir,
                )
                variant_results.append(tsne_res)
            except Exception as exc:
                _logger.warning("t-SNE (perp=%d) failed for %s: %s", perp, variant_key, exc)

        # --- UMAP ---
        for n_neighbors in umap_n_neighbors_list:
            for min_dist in umap_min_dists:
                is_canon = (
                    n_neighbors == CANONICAL_UMAP_N_NEIGHBORS
                    and min_dist == CANONICAL_UMAP_MIN_DIST
                )
                _logger.info(
                    "  UMAP n_neighbors=%d min_dist=%.1f (canonical=%s)",
                    n_neighbors, min_dist, is_canon,
                )
                try:
                    umap_res = _run_umap(
                        X, font_names, labels,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=random_state,
                        out_dir=embed_dir,
                        variant_key=variant_key,
                        is_canonical=is_canon,
                        canonical_dir=canonical_dir,
                    )
                    if umap_res is not None:
                        variant_results.append(umap_res)
                except Exception as exc:
                    _logger.warning(
                        "UMAP (nn=%d, md=%.1f) failed for %s: %s",
                        n_neighbors, min_dist, variant_key, exc,
                    )

        all_results[variant_key] = variant_results
        _logger.info("  Completed %d embeddings for %s", len(variant_results), variant_key)

    # Save metadata JSON
    import json
    meta_path = font_feature_dir / "embeddings" / "embedding_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_results, f, indent=2)
    _logger.info("Saved embedding metadata → %s", meta_path)

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Phase 1b dimensionality reduction embeddings")
    parser.add_argument("--font-feature-dir", default="outputs", help="Directory with font_features_*.npz")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tsne-n-iter", type=int, default=1000)
    parser.add_argument("--pca-n-components", type=int, default=10)
    args = parser.parse_args()

    results = compute_embeddings(
        font_feature_dir=args.font_feature_dir,
        random_state=args.random_state,
        tsne_n_iter=args.tsne_n_iter,
        pca_n_components=args.pca_n_components,
    )
    _logger.info("Done. Processed %d variants.", len(results))
