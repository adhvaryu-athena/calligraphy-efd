"""Phase 2 driver: cluster stability analysis and ablation table."""
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    outputs_dir = Path("outputs")
    canonical_npz = outputs_dir / "font_features_A_mean.npz"

    if not canonical_npz.exists():
        logger.error(
            "font_features_A_mean.npz not found in outputs/. "
            "Run Phase 0 first."
        )
        sys.exit(1)

    stability_dir = outputs_dir / "stability"
    stability_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2a: Stability analysis
    logger.info("=== Phase 2a: Cluster stability ===")
    try:
        from src.stability import run_all_stability
        results = run_all_stability(
            feature_dir=str(outputs_dir),
            output_dir=str(stability_dir),
        )
        logger.info("Stability analysis done. Keys: %s", list(results.keys()))

        if "bootstrap" in results:
            bootstrap = results["bootstrap"]
            logger.info(
                "Bootstrap ARI by k: %s",
                {k: f"{v:.4f}" for k, v in bootstrap.items()},
            )
        if "permutation" in results:
            perm = results["permutation"]
            logger.info(
                "Permutation test: observed_sil=%.4f, p_value=%s",
                perm.get("observed_silhouette") or float("nan"),
                perm.get("p_value"),
            )
    except Exception as exc:
        logger.error("Stability analysis failed: %s", exc)
        sys.exit(1)

    # Phase 2b: Ablation table (Config A / B / C comparison)
    logger.info("=== Phase 2b: Ablation table ===")
    try:
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        ablation_rows = []
        for config in ("A", "B", "C"):
            for mode in ("mean", "median"):
                npz_path = outputs_dir / f"font_features_{config}_{mode}.npz"
                if not npz_path.exists():
                    logger.info("Skipping %s_%s (file not found)", config, mode)
                    continue
                data = np.load(npz_path, allow_pickle=True)
                X_raw = data["X"].astype(float)
                scaler = StandardScaler()
                X = scaler.fit_transform(X_raw)

                row = {"config": config, "mode": mode, "n_fonts": X.shape[0],
                       "n_features": X.shape[1]}

                for k in (4, 5, 6):
                    if k >= X.shape[0]:
                        continue
                    try:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = km.fit_predict(X)
                        sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else float("nan")
                    except Exception:
                        sil = float("nan")
                    row[f"sil_k{k}"] = round(sil, 4) if not (sil != sil) else None

                ablation_rows.append(row)
                logger.info("Ablation %s_%s: %s", config, mode, row)

        if ablation_rows:
            import pandas as pd
            ablation_df = pd.DataFrame(ablation_rows)
            ablation_path = stability_dir / "ablation_table.csv"
            ablation_df.to_csv(ablation_path, index=False)
            logger.info("Saved ablation table → %s", ablation_path)
        else:
            logger.warning("No feature matrices found for ablation table.")
    except Exception as exc:
        logger.error("Ablation table failed: %s", exc)
        sys.exit(1)

    logger.info("Phase 2 complete.")


if __name__ == "__main__":
    main()
