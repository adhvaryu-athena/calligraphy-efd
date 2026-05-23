"""Phase 3b — External validation: cluster labels vs Google Fonts categories."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

_logger = logging.getLogger(__name__)


def compute_cluster_vs_google(
    cluster_labels: List,
    google_labels: List,
    font_names: List[str],
) -> Dict:
    """Compute contingency table, ARI, NMI between cluster labels and Google labels.

    Saves:
    - outputs/validation/contingency_table.csv
    - outputs/figures/fig_contingency_heatmap.png
    - outputs/validation/cluster_validation_results.json

    Returns dict with:
    - contingency_table: pd.DataFrame
    - ari: float
    - nmi: float
    - contingency_csv_path: str
    - heatmap_png_path: str
    """
    validation_dir = Path("outputs/validation")
    figures_dir = Path("outputs/figures")
    validation_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cluster_series = pd.Series(cluster_labels, name="cluster")
    google_series = pd.Series(google_labels, name="google_category")

    contingency_table = pd.crosstab(cluster_series, google_series)

    ari = float(adjusted_rand_score(google_labels, cluster_labels))
    nmi = float(normalized_mutual_info_score(google_labels, cluster_labels))

    _logger.info("ARI=%.4f  NMI=%.4f", ari, nmi)

    # Save contingency table CSV
    contingency_csv_path = str(validation_dir / "contingency_table.csv")
    contingency_table.to_csv(contingency_csv_path)
    _logger.info("Saved contingency table → %s", contingency_csv_path)

    # Save heatmap
    heatmap_png_path = str(figures_dir / "fig_contingency_heatmap.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            contingency_table,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
        )
        ax.set_title("Cluster assignments vs Google Fonts categories")
        ax.set_xlabel("Google category")
        ax.set_ylabel("Cluster")
        plt.tight_layout()
        fig.savefig(heatmap_png_path, dpi=150)
        plt.close(fig)
        _logger.info("Saved heatmap → %s", heatmap_png_path)
    except Exception as exc:
        _logger.warning("Heatmap generation failed: %s", exc)
        heatmap_png_path = ""

    # Save JSON results
    results_json = {
        "ari": ari,
        "nmi": nmi,
        "n_fonts": len(font_names),
        "n_clusters": int(len(np.unique(cluster_labels))),
        "n_google_categories": int(len(np.unique(google_labels))),
        "contingency_csv": contingency_csv_path,
        "heatmap_png": heatmap_png_path,
        "caveat": (
            "ARI is penalised when n_clusters != n_true_categories; "
            "contingency table is the primary result."
        ),
    }
    json_path = validation_dir / "cluster_validation_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    _logger.info("Saved validation results → %s", json_path)

    return {
        "contingency_table": contingency_table,
        "ari": ari,
        "nmi": nmi,
        "contingency_csv_path": contingency_csv_path,
        "heatmap_png_path": heatmap_png_path,
    }


def run_validation(
    clustering_results_path: str = "outputs/clustering_results.json",
    fonts_csv_path: str = "outputs/fonts.csv",
) -> Optional[Dict]:
    """Orchestrate validation for best-performing clustering configuration.

    Best = highest mean silhouette score across k values for kmeans.
    """
    clustering_results_path = Path(clustering_results_path)
    fonts_csv_path = Path(fonts_csv_path)

    if not clustering_results_path.exists():
        _logger.error("clustering_results.json not found at %s", clustering_results_path)
        return None

    if not fonts_csv_path.exists():
        _logger.error("fonts.csv not found at %s", fonts_csv_path)
        return None

    with open(clustering_results_path) as f:
        clustering_results = json.load(f)

    fonts_df = pd.read_csv(fonts_csv_path)
    google_col = "google_category" if "google_category" in fonts_df.columns else "style_class"
    google_label_map = dict(zip(fonts_df["font_name"], fonts_df[google_col]))

    # Find best variant: highest mean silhouette across k values for kmeans
    best_variant = None
    best_mean_sil = float("-inf")

    for variant_key, variant_data in clustering_results.items():
        kmeans_data = variant_data.get("kmeans", {})
        sil_scores = []
        for k_key, k_data in kmeans_data.items():
            sil = k_data.get("silhouette", float("nan"))
            if not (sil != sil):  # nan check
                sil_scores.append(sil)
        if not sil_scores:
            continue
        mean_sil = float(np.mean(sil_scores))
        if mean_sil > best_mean_sil:
            best_mean_sil = mean_sil
            best_variant = variant_key

    if best_variant is None:
        _logger.error("Could not determine best variant from clustering results")
        return None

    _logger.info("Best variant: %s (mean_silhouette=%.4f)", best_variant, best_mean_sil)

    # Find best k within this variant
    kmeans_data = clustering_results[best_variant]["kmeans"]
    best_k_key = max(
        kmeans_data.keys(),
        key=lambda kk: kmeans_data[kk].get("silhouette", float("-inf"))
        if not (kmeans_data[kk].get("silhouette", 0) != kmeans_data[kk].get("silhouette", 0))
        else float("-inf"),
    )
    best_k_data = kmeans_data[best_k_key]
    cluster_labels = best_k_data["labels"]
    font_names = clustering_results[best_variant].get("font_names", [])

    if not font_names:
        _logger.error("No font_names in clustering results for variant %s", best_variant)
        return None

    google_labels = [google_label_map.get(fn, "UNKNOWN") for fn in font_names]

    _logger.info(
        "Running validation: variant=%s k=%s n_fonts=%d",
        best_variant, best_k_key, len(font_names),
    )

    result = compute_cluster_vs_google(cluster_labels, google_labels, font_names)
    result["best_variant"] = best_variant
    result["best_k"] = best_k_key
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Phase 3b cluster validation")
    parser.add_argument("--clustering-results", default="outputs/clustering_results.json")
    parser.add_argument("--fonts-csv", default="outputs/fonts.csv")
    args = parser.parse_args()

    result = run_validation(
        clustering_results_path=args.clustering_results,
        fonts_csv_path=args.fonts_csv,
    )
    if result:
        print(f"ARI={result['ari']:.4f}  NMI={result['nmi']:.4f}")
        print(f"Best variant: {result.get('best_variant')}  k: {result.get('best_k')}")
        print(result["contingency_table"])
