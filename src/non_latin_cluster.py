"""Phase 4b — Clustering within script and cross-script UMAP embedding."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

_logger = logging.getLogger(__name__)


def cluster_within_script(
    script: str,
    feature_dir: str = "outputs/non_latin",
    k_range=range(2, 8),
) -> Dict:
    """K-means + hierarchical + silhouette for one script's font features.

    Returns dict with 'kmeans', 'hierarchical', 'best_k', 'best_silhouette'.
    """
    feature_dir = Path(feature_dir)
    npz_path = feature_dir / f"{script}_features.npz"

    if not npz_path.exists():
        _logger.error("Features not found for script %r at %s", script, npz_path)
        return {}

    data = np.load(npz_path, allow_pickle=True)
    X_raw = data["X"].astype(float)
    font_names = data["font_names"].tolist() if "font_names" in data else []
    labels = data["labels"].tolist() if "labels" in data else []

    n_samples = X_raw.shape[0]
    _logger.info("cluster_within_script: script=%s n_fonts=%d", script, n_samples)

    if n_samples < 2:
        _logger.warning("Too few fonts (%d) for script %r — skipping", n_samples, script)
        return {}

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    k_list = [k for k in k_range if 2 <= k < n_samples]
    if not k_list:
        _logger.warning("No valid k values for n_samples=%d", n_samples)
        return {}

    # K-means
    kmeans_results: Dict[int, Dict] = {}
    best_k_kmeans = None
    best_sil_kmeans = float("-inf")

    for k in k_list:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(X)
        n_unique = len(np.unique(cluster_labels))
        if n_unique < 2:
            sil = float("nan")
            db = float("nan")
        else:
            try:
                sil = float(silhouette_score(X, cluster_labels))
                db = float(davies_bouldin_score(X, cluster_labels))
            except Exception as exc:
                _logger.warning("Metrics failed k=%d: %s", k, exc)
                sil, db = float("nan"), float("nan")

        kmeans_results[k] = {
            "silhouette": sil,
            "davies_bouldin": db,
            "labels": cluster_labels.tolist(),
        }
        _logger.info("  kmeans k=%d: silhouette=%.4f db=%.4f", k, sil, db)

        if not (sil != sil) and sil > best_sil_kmeans:  # nan-safe comparison
            best_sil_kmeans = sil
            best_k_kmeans = k

    # Hierarchical (Ward)
    hierarchical_results: Dict[int, Dict] = {}
    try:
        Z = linkage(X, method="ward")
        for k in k_list:
            hier_labels = fcluster(Z, k, criterion="maxclust")
            n_unique = len(np.unique(hier_labels))
            if n_unique < 2:
                sil = float("nan")
            else:
                try:
                    sil = float(silhouette_score(X, hier_labels))
                except Exception:
                    sil = float("nan")
            hierarchical_results[k] = {
                "silhouette": sil,
                "labels": hier_labels.tolist(),
            }
            _logger.info("  hierarchical k=%d: silhouette=%.4f", k, sil)
    except Exception as exc:
        _logger.warning("Hierarchical clustering failed: %s", exc)

    result = {
        "script": script,
        "n_fonts": n_samples,
        "font_names": font_names,
        "kmeans": kmeans_results,
        "hierarchical": hierarchical_results,
        "best_k": best_k_kmeans,
        "best_silhouette": best_sil_kmeans if best_k_kmeans is not None else float("nan"),
    }

    # Save per-script results
    out_dir = Path(feature_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    results_path = out_dir / f"{script}_cluster_results.json"
    with open(results_path, "w") as f:
        json.dump({k: (v if not isinstance(v, float) else v) for k, v in result.items()}, f, indent=2)
    _logger.info("Saved %s cluster results → %s", script, results_path)

    return result


def embed_cross_script(
    scripts: Tuple[str, ...] = ("latin", "devanagari", "arabic"),
    feature_dir: str = "outputs",
    output_dir: str = "outputs/non_latin",
) -> Optional[pd.DataFrame]:
    """UMAP fit on Latin, transform non-Latin. Save CSV + PNG.

    Saves:
    - outputs/non_latin/cross_script_umap.csv (font_name, script, style_label, umap_x, umap_y)
    - outputs/figures/fig_cross_script_umap.png
    """
    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    figures_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        import umap as umap_mod
    except ImportError:
        _logger.warning("umap-learn not available — skipping cross-script UMAP embedding")
        return None

    # Load Latin features
    latin_npz = feature_dir / "font_features_A_mean.npz"
    if not latin_npz.exists():
        _logger.error("Latin features not found at %s", latin_npz)
        return None

    latin_data = np.load(latin_npz, allow_pickle=True)
    X_latin = latin_data["X"].astype(float)
    latin_font_names = latin_data["font_names"].tolist() if "font_names" in latin_data else [
        f"latin_{i}" for i in range(X_latin.shape[0])
    ]
    latin_labels = latin_data["labels"].tolist() if "labels" in latin_data else [""] * X_latin.shape[0]

    # Fit scaler on Latin
    scaler = StandardScaler()
    X_latin_scaled = scaler.fit_transform(X_latin)

    # Collect all data points
    all_X: List[np.ndarray] = [X_latin_scaled]
    all_font_names: List[str] = latin_font_names
    all_scripts: List[str] = ["latin"] * len(latin_font_names)
    all_style_labels: List[str] = latin_labels

    for script in scripts:
        if script == "latin":
            continue
        nl_npz = feature_dir / "non_latin" / f"{script}_features.npz"
        if not nl_npz.exists():
            _logger.warning("Non-Latin features not found for script %r at %s", script, nl_npz)
            continue

        nl_data = np.load(nl_npz, allow_pickle=True)
        X_nl = nl_data["X"].astype(float)
        nl_font_names = nl_data["font_names"].tolist() if "font_names" in nl_data else [
            f"{script}_{i}" for i in range(X_nl.shape[0])
        ]
        nl_labels = nl_data["labels"].tolist() if "labels" in nl_data else [""] * X_nl.shape[0]

        # Align feature dimension to Latin's
        latin_dim = X_latin.shape[1]
        nl_dim = X_nl.shape[1]
        if nl_dim > latin_dim:
            X_nl = X_nl[:, :latin_dim]
        elif nl_dim < latin_dim:
            padded = np.zeros((X_nl.shape[0], latin_dim), dtype=float)
            padded[:, :nl_dim] = X_nl
            X_nl = padded

        X_nl_scaled = scaler.transform(X_nl)

        all_X.append(X_nl_scaled)
        all_font_names.extend(nl_font_names)
        all_scripts.extend([script] * len(nl_font_names))
        all_style_labels.extend(nl_labels)
        _logger.info("Loaded %s: %d fonts", script, X_nl.shape[0])

    X_all = np.vstack(all_X)
    n_total = X_all.shape[0]
    n_latin = X_latin_scaled.shape[0]

    _logger.info("UMAP: fitting on %d Latin fonts, transforming %d total", n_latin, n_total)

    try:
        reducer = umap_mod.UMAP(n_components=2, random_state=42)
        reducer.fit(X_latin_scaled)
        embedding = reducer.transform(X_all)
    except Exception as exc:
        _logger.error("UMAP failed: %s", exc)
        return None

    df = pd.DataFrame({
        "font_name": all_font_names,
        "script": all_scripts,
        "style_label": all_style_labels,
        "umap_x": embedding[:, 0],
        "umap_y": embedding[:, 1],
    })

    csv_path = output_dir / "cross_script_umap.csv"
    df.to_csv(csv_path, index=False)
    _logger.info("Saved cross-script UMAP CSV → %s", csv_path)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        palette = {
            "latin": "#4477AA",
            "devanagari": "#EE6677",
            "arabic": "#228833",
            "hiragana": "#CCBB44",
        }
        for script_name in df["script"].unique():
            mask = df["script"] == script_name
            color = palette.get(script_name, "#BBBBBB")
            ax.scatter(
                df.loc[mask, "umap_x"],
                df.loc[mask, "umap_y"],
                label=script_name,
                color=color,
                alpha=0.6,
                s=20,
            )
        ax.legend(title="Script")
        ax.set_title("Cross-script UMAP embedding (Latin fit)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        plt.tight_layout()
        png_path = figures_dir / "fig_cross_script_umap.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        _logger.info("Saved cross-script UMAP plot → %s", png_path)
    except Exception as exc:
        _logger.warning("UMAP plot failed: %s", exc)

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Phase 4b non-Latin clustering + UMAP")
    parser.add_argument("--script", default="devanagari",
                        help="Script to cluster within (devanagari/arabic/hiragana)")
    parser.add_argument("--feature-dir", default="outputs")
    parser.add_argument("--non-latin-dir", default="outputs/non_latin")
    parser.add_argument("--cross-script", action="store_true",
                        help="Run cross-script UMAP instead of within-script clustering")
    args = parser.parse_args()

    if args.cross_script:
        df = embed_cross_script(
            scripts=("latin", "devanagari", "arabic"),
            feature_dir=args.feature_dir,
            output_dir=args.non_latin_dir,
        )
        if df is not None:
            print(df.head())
    else:
        result = cluster_within_script(
            script=args.script,
            feature_dir=args.non_latin_dir,
        )
        if result:
            print(f"Best k={result.get('best_k')} silhouette={result.get('best_silhouette'):.4f}")
