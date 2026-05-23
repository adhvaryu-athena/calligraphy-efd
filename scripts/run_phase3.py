"""Phase 3 driver: cluster validation (cluster labels vs Google categories)."""
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

    clustering_results_path = Path("outputs/clustering_results.json")
    fonts_csv_path = Path("outputs/fonts.csv")

    if not clustering_results_path.exists():
        logger.error(
            "clustering_results.json not found at %s. Run Phase 1 first.",
            clustering_results_path,
        )
        sys.exit(1)

    if not fonts_csv_path.exists():
        logger.error(
            "fonts.csv not found at %s. Run the corpus build step first.",
            fonts_csv_path,
        )
        sys.exit(1)

    logger.info("=== Phase 3: Cluster validation ===")
    try:
        from src.cluster_validation import run_validation
        result = run_validation(
            clustering_results_path=str(clustering_results_path),
            fonts_csv_path=str(fonts_csv_path),
        )
        if result is None:
            logger.error("Validation returned no results.")
            sys.exit(1)

        logger.info(
            "Validation complete: ARI=%.4f, NMI=%.4f",
            result.get("ari", float("nan")),
            result.get("nmi", float("nan")),
        )
        logger.info("Best variant: %s  Best k: %s",
                    result.get("best_variant"), result.get("best_k"))

        contingency = result.get("contingency_table")
        if contingency is not None:
            logger.info("Contingency table shape: %s", contingency.shape)
    except Exception as exc:
        logger.error("Cluster validation failed: %s", exc)
        sys.exit(1)

    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    main()
