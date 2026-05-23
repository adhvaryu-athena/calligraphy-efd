"""Phase 4 driver: non-Latin script analysis.

Full execution requires the Google Fonts repository at data/google-fonts/.
Clone it with:
    git clone --depth 1 https://github.com/google/fonts.git data/google-fonts
"""
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

    google_fonts_dir = Path("data/google-fonts")

    if not google_fonts_dir.exists():
        logger.warning(
            "data/google-fonts/ not found. Phase 4 requires the Google Fonts repo.\n"
            "  Clone it with:\n"
            "    git clone --depth 1 https://github.com/google/fonts.git data/google-fonts"
        )
        return

    logger.info("=== Phase 4: Non-Latin script analysis ===")

    from src.non_latin import build_non_latin_corpus, paired_cross_script_analysis
    from src.non_latin_cluster import cluster_within_script, embed_cross_script

    Path("outputs/non_latin").mkdir(parents=True, exist_ok=True)

    for script in ("devanagari", "arabic"):
        logger.info("--- Script: %s ---", script)

        try:
            corpus_result = build_non_latin_corpus(
                google_fonts_dir=str(google_fonts_dir),
                script=script,
                n_harmonics=20,
            )
            if corpus_result is None or corpus_result.get("n_fonts", 0) == 0:
                logger.warning("No fonts found for script=%s — skipping.", script)
                continue
            logger.info("Corpus built: %d fonts with %s glyphs", corpus_result["n_fonts"], script)
        except Exception as exc:
            logger.error("build_non_latin_corpus failed for %s: %s", script, exc)
            continue

        try:
            cluster_within_script(script=script, feature_dir="outputs/non_latin")
            logger.info("Within-script clustering done for %s", script)
        except Exception as exc:
            logger.warning("cluster_within_script failed for %s: %s", script, exc)

        latin_path = Path("outputs/font_features_A_mean.npz")
        nl_path = Path(f"outputs/non_latin/{script}_features.npz")
        if latin_path.exists() and nl_path.exists():
            try:
                paired_cross_script_analysis(
                    latin_features_path=str(latin_path),
                    non_latin_features_path=str(nl_path),
                    script=script,
                )
                logger.info("Paired cross-script analysis done for %s", script)
            except Exception as exc:
                logger.warning("paired_cross_script_analysis failed for %s: %s", script, exc)

    try:
        embed_cross_script(
            scripts=("latin", "devanagari", "arabic"),
            feature_dir="outputs",
            output_dir="outputs/non_latin",
        )
        logger.info("Cross-script UMAP embedding saved.")
    except Exception as exc:
        logger.warning("embed_cross_script failed: %s", exc)

    logger.info("Phase 4 complete.")


if __name__ == "__main__":
    main()
