"""Phase 1 driver: clustering, embeddings, similarity, and discovery."""
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
    if not outputs_dir.exists():
        logger.error("outputs/ directory not found. Run Phase 0 first.")
        sys.exit(1)

    # Check for at least one font feature matrix
    npz_files = sorted(outputs_dir.glob("font_features_*.npz"))
    if not npz_files:
        logger.error(
            "No font_features_*.npz files found in outputs/. "
            "Run Phase 0 first."
        )
        sys.exit(1)

    logger.info("Found %d feature matrix file(s): %s ...",
                len(npz_files), [f.name for f in npz_files[:3]])

    # Phase 1a: Clustering
    logger.info("=== Phase 1a: Clustering ===")
    try:
        from src.cluster import run_all_clustering
        clustering_results = run_all_clustering(
            font_feature_dir=str(outputs_dir),
            k_range=range(2, 11),
            random_state=42,
            gap_B=10,
        )
        logger.info("Clustering done. %d variants processed.", len(clustering_results))
    except Exception as exc:
        logger.error("Clustering failed: %s", exc)
        sys.exit(1)

    # Phase 1b: Embeddings
    logger.info("=== Phase 1b: Embeddings ===")
    try:
        from src.embed import compute_embeddings
        embed_results = compute_embeddings(
            font_feature_dir=str(outputs_dir),
            random_state=42,
        )
        logger.info("Embeddings done. %d variants processed.", len(embed_results))
    except Exception as exc:
        logger.error("Embeddings failed: %s", exc)
        sys.exit(1)

    # Phase 1c: Similarity
    logger.info("=== Phase 1c: Similarity ===")
    try:
        from src.similarity import compute_similarity
        sim_results = compute_similarity(
            font_feature_dir=str(outputs_dir),
            k_neighbors=(5, 10, 20),
        )
        logger.info("Similarity done. %d variants processed.", len(sim_results))
    except Exception as exc:
        logger.error("Similarity failed: %s", exc)
        sys.exit(1)

    # Phase 1d: Discovery
    logger.info("=== Phase 1d: Discovery ===")
    try:
        from src.discover import find_google_disagreed_pairs, compute_within_class_distances

        sim_dir = outputs_dir / "similarity"
        neighbors_csvs = sorted(sim_dir.glob("neighbors_A_mean_cosine.csv")) if sim_dir.exists() else []
        if not neighbors_csvs:
            neighbors_csvs = sorted(sim_dir.glob("neighbors_A_mean_*.csv")) if sim_dir.exists() else []
        if not neighbors_csvs:
            neighbors_csvs = sorted(sim_dir.glob("neighbors_*.csv")) if sim_dir.exists() else []

        if neighbors_csvs:
            neighbors_csv = neighbors_csvs[0]
            per_font, agg, pairs = find_google_disagreed_pairs(str(neighbors_csv), k=5)
            logger.info(
                "Discovery: aggregate cross-category rate=%.3f, %d fonts, %d cross-pairs",
                agg, len(per_font), len(pairs),
            )

            # Within-class distances
            pairwise_npzs = sorted(
                sim_dir.glob("pairwise_A_mean_cosine.npz")
            ) if sim_dir.exists() else []
            if not pairwise_npzs:
                pairwise_npzs = sorted(
                    sim_dir.glob("pairwise_A_mean_*.npz")
                ) if sim_dir.exists() else []
            if pairwise_npzs:
                within = compute_within_class_distances(str(pairwise_npzs[0]))
                logger.info("Within-class distances: %s", within)
        else:
            logger.warning("No neighbors CSV found — skipping discovery.")
    except Exception as exc:
        logger.error("Discovery failed: %s", exc)
        sys.exit(1)

    logger.info("Phase 1 complete.")


if __name__ == "__main__":
    main()
