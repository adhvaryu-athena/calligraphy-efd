"""Phase 2b — Sensitivity analysis: harmonic order sweep + ablation table."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

_logger = logging.getLogger(__name__)


def cluster_quality_vs_harmonic_order(
    fonts_df: pd.DataFrame,
    harmonic_orders: Tuple[int, ...] = (5, 10, 15, 20, 30, 40),
    algorithm: str = "kmeans",
    k: int = 4,
    mode: str = "mean",
    random_state: int = 42,
) -> Dict[int, float]:
    """Recompute EFD features at each harmonic order and measure cluster quality.

    Loads outputs/outlines.pkl, recomputes EFD features, aggregates to font
    level using `mode`, runs algorithm at k, returns {n: silhouette_score}.
    """
    from src.efd import compute_efd_features
    from src.aggregate import aggregate_to_font_level

    outlines_path = Path("outputs/outlines.pkl")
    if not outlines_path.exists():
        _logger.error("outlines.pkl not found at %s", outlines_path)
        return {}

    with open(outlines_path, "rb") as f:
        outlines: Dict[str, Dict[str, np.ndarray]] = pickle.load(f)

    # Build label map
    label_col = "style_class" if "style_class" in fonts_df.columns else fonts_df.columns[1]
    label_map = dict(zip(fonts_df["font_name"], fonts_df[label_col]))

    result: Dict[int, float] = {}

    for order in harmonic_orders:
        _logger.info("cluster_quality_vs_harmonic_order: order=%d", order)
        font_names_out = []
        font_vectors = []
        labels_out = []

        for font_name, glyph_dict in outlines.items():
            label = label_map.get(font_name)
            if label is None:
                continue
            glyph_feats = []
            for glyph_char, contour in glyph_dict.items():
                if contour is None:
                    continue
                try:
                    feat = compute_efd_features(contour, order=order, normalize=True)
                    glyph_feats.append(feat)
                except Exception as exc:
                    _logger.debug("EFD failed font=%s glyph=%s order=%d: %s",
                                  font_name, glyph_char, order, exc)
                    continue
            if not glyph_feats:
                continue
            try:
                font_vec = aggregate_to_font_level(glyph_feats, mode=mode)
            except Exception as exc:
                _logger.warning("aggregate failed font=%s order=%d: %s", font_name, order, exc)
                continue
            font_names_out.append(font_name)
            font_vectors.append(font_vec)
            labels_out.append(str(label))

        if len(font_vectors) < 2:
            _logger.warning("Too few font vectors at order=%d — skipping", order)
            result[order] = float("nan")
            continue

        # Pad to uniform width (important for concat mode)
        max_len = max(v.shape[0] for v in font_vectors)
        X_rows = []
        for v in font_vectors:
            if v.shape[0] < max_len:
                padded = np.zeros(max_len, dtype=float)
                padded[: v.shape[0]] = v
                X_rows.append(padded)
            else:
                X_rows.append(v.astype(float))
        X_raw = np.vstack(X_rows)

        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = km.fit_predict(X)

        n_unique = len(np.unique(cluster_labels))
        if n_unique < 2:
            _logger.warning("Only 1 cluster formed at order=%d", order)
            result[order] = float("nan")
            continue

        sil = float(silhouette_score(X, cluster_labels))
        result[order] = sil
        _logger.info("  order=%d: silhouette=%.4f (n_fonts=%d)", order, sil, len(font_vectors))

    return result


def run_ablation_table(
    feature_dir: str = "outputs",
    output_path: str = "outputs/ablation_table.csv",
) -> pd.DataFrame:
    """Produce ablation table CSV.

    Columns: config | mode | algorithm | k | silhouette | davies_bouldin | ari_vs_google | notes

    Loads pre-computed clustering_results.json for most rows.
    Also adds harmonic order rows (Config A / mean / kmeans, k=4).
    Also adds distance metric rows (cosine, euclidean, mahalanobis).
    """
    feature_dir = Path(feature_dir)
    rows = []

    clustering_results_path = feature_dir / "clustering_results.json"
    if not clustering_results_path.exists():
        _logger.warning("clustering_results.json not found — ablation table will be sparse")
        clustering_results = {}
    else:
        import json
        with open(clustering_results_path) as f:
            clustering_results = json.load(f)

    # Load fonts.csv for google labels
    fonts_csv_path = feature_dir / "fonts.csv"
    if fonts_csv_path.exists():
        fonts_df = pd.read_csv(fonts_csv_path)
        google_col = "google_category" if "google_category" in fonts_df.columns else "style_class"
        google_label_map = dict(zip(fonts_df["font_name"], fonts_df[google_col]))
    else:
        fonts_df = None
        google_label_map = {}

    # Process clustering_results.json rows
    for variant_key, variant_data in clustering_results.items():
        # Parse variant key like "A_mean" or "B_concat"
        parts = variant_key.split("_", 1)
        config = parts[0] if len(parts) >= 1 else "?"
        mode = parts[1] if len(parts) >= 2 else "?"

        font_names = variant_data.get("font_names", [])
        true_labels = variant_data.get("true_labels", [])

        # Load actual X for DB score recomputation
        npz_path = feature_dir / f"font_features_{variant_key}.npz"
        X_scaled = None
        if npz_path.exists():
            try:
                data = np.load(npz_path, allow_pickle=True)
                X_raw = data["X"].astype(float)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_raw)
                if not font_names:
                    font_names = data["font_names"].tolist() if "font_names" in data else []
            except Exception as exc:
                _logger.warning("Could not load X for variant %s: %s", variant_key, exc)

        def _ari(cluster_labels):
            if not cluster_labels or not font_names or not google_label_map:
                return float("nan")
            google_labels = [google_label_map.get(fn, "") for fn in font_names]
            try:
                return float(adjusted_rand_score(google_labels, cluster_labels))
            except Exception:
                return float("nan")

        def _metrics(cluster_labels):
            sil = float("nan")
            db = float("nan")
            if X_scaled is not None and cluster_labels and len(np.unique(cluster_labels)) > 1:
                try:
                    sil = float(silhouette_score(X_scaled, cluster_labels))
                    db = float(davies_bouldin_score(X_scaled, cluster_labels))
                except Exception:
                    pass
            return sil, db

        # KMeans rows
        kmeans_data = variant_data.get("kmeans", {})
        for k_key, k_data in kmeans_data.items():
            k_val = int(k_key.split("_")[1])
            sil = k_data.get("silhouette", float("nan"))
            db = k_data.get("davies_bouldin", float("nan"))
            cluster_labels = k_data.get("labels", [])
            if X_scaled is not None and cluster_labels and len(np.unique(cluster_labels)) > 1:
                try:
                    db = float(davies_bouldin_score(X_scaled, cluster_labels))
                except Exception:
                    pass
            rows.append({
                "config": config, "mode": mode, "algorithm": "kmeans", "k": k_val,
                "silhouette": sil, "davies_bouldin": db,
                "ari_vs_google": _ari(cluster_labels), "notes": "",
            })

        # Hierarchical rows
        for k_key, k_data in variant_data.get("hierarchical", {}).items():
            k_val = int(k_key.split("_")[1])
            cluster_labels = k_data.get("labels", [])
            sil, db = _metrics(cluster_labels)
            rows.append({
                "config": config, "mode": mode, "algorithm": "hierarchical", "k": k_val,
                "silhouette": sil, "davies_bouldin": db,
                "ari_vs_google": _ari(cluster_labels), "notes": "",
            })

        # GMM rows
        gmm_data = variant_data.get("gmm", {})
        optimal_k = gmm_data.get("optimal_k")
        for k_key, k_data in gmm_data.items():
            if not k_key.startswith("k_"):
                continue
            k_val = int(k_key.split("_")[1])
            cluster_labels = k_data.get("labels", [])
            sil, db = _metrics(cluster_labels)
            note = "BIC-optimal k" if k_val == optimal_k else ""
            rows.append({
                "config": config, "mode": mode, "algorithm": "gmm", "k": k_val,
                "silhouette": sil, "davies_bouldin": db,
                "ari_vs_google": _ari(cluster_labels), "notes": note,
            })

    # Harmonic order rows (Config A / mean / kmeans, k=4)
    if fonts_csv_path.exists():
        try:
            fonts_df_tmp = pd.read_csv(fonts_csv_path)
            harmonic_scores = cluster_quality_vs_harmonic_order(
                fonts_df_tmp,
                harmonic_orders=(5, 10, 15, 20, 30, 40),
                algorithm="kmeans",
                k=4,
                mode="mean",
                random_state=42,
            )
            for order, sil in harmonic_scores.items():
                rows.append({
                    "config": "A", "mode": "mean", "algorithm": "kmeans", "k": 4,
                    "silhouette": sil, "davies_bouldin": float("nan"),
                    "ari_vs_google": float("nan"),
                    "notes": f"harmonic_order={order}",
                })
        except Exception as exc:
            _logger.warning("Harmonic order sweep failed: %s", exc)

    # Distance metric rows (cosine, euclidean) via agglomerative on precomputed D
    best_npz = feature_dir / "font_features_A_mean.npz"
    if best_npz.exists():
        try:
            from scipy.spatial.distance import cdist
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics.pairwise import cosine_distances

            data = np.load(best_npz, allow_pickle=True)
            X_raw = data["X"].astype(float)
            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)
            fn_list = data["font_names"].tolist() if "font_names" in data else []

            for metric in ("cosine", "euclidean"):
                D = cosine_distances(X) if metric == "cosine" else cdist(X, X, metric="euclidean")
                for k_val in (3, 4, 5):
                    try:
                        agg = AgglomerativeClustering(
                            n_clusters=k_val, metric="precomputed", linkage="average"
                        )
                        cl = agg.fit_predict(D)
                        sil = float("nan")
                        if len(np.unique(cl)) > 1:
                            sil = float(silhouette_score(D, cl, metric="precomputed"))
                        ari_val = float("nan")
                        if fn_list and google_label_map:
                            gl = [google_label_map.get(fn, "") for fn in fn_list]
                            try:
                                ari_val = float(adjusted_rand_score(gl, cl))
                            except Exception:
                                pass
                        rows.append({
                            "config": "A", "mode": "mean",
                            "algorithm": f"hierarchical_{metric}",
                            "k": k_val,
                            "silhouette": sil, "davies_bouldin": float("nan"),
                            "ari_vs_google": ari_val,
                            "notes": f"distance_metric={metric}",
                        })
                    except Exception as exc:
                        _logger.warning("Distance metric k=%d metric=%s failed: %s", k_val, metric, exc)
        except Exception as exc:
            _logger.warning("Distance metric ablation section failed: %s", exc)

    if not rows:
        _logger.warning("No rows produced for ablation table")
        df = pd.DataFrame(columns=["config", "mode", "algorithm", "k",
                                   "silhouette", "davies_bouldin", "ari_vs_google", "notes"])
    else:
        df = pd.DataFrame(rows)
        df = df.sort_values("silhouette", ascending=False, na_position="last").reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    _logger.info("Saved ablation table → %s (rows=%d)", output_path, len(df))

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Phase 2b sensitivity analysis")
    parser.add_argument("--feature-dir", default="outputs")
    parser.add_argument("--output-path", default="outputs/ablation_table.csv")
    parser.add_argument("--harmonic-only", action="store_true")
    args = parser.parse_args()

    if args.harmonic_only:
        _fonts_df = pd.read_csv(Path(args.feature_dir) / "fonts.csv")
        scores = cluster_quality_vs_harmonic_order(_fonts_df)
        for order, sil in scores.items():
            print(f"  order={order}: silhouette={sil:.4f}")
    else:
        df = run_ablation_table(
            feature_dir=args.feature_dir,
            output_path=args.output_path,
        )
        print(df.head(20).to_string())
