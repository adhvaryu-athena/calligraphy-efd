"""Phase 4 driver: non-Latin script analysis (placeholder).

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
            "data/google-fonts/ not found. Phase 4 (non-Latin analysis) requires the "
            "Google Fonts repository.\n"
            "  Clone it with:\n"
            "    git clone --depth 1 https://github.com/google/fonts.git data/google-fonts"
        )
        logger.info("Phase 4 skipped (missing google-fonts directory).")
        return

    logger.info("=== Phase 4: Non-Latin script analysis ===")

    try:
        from src.non_latin import run as run_non_latin
        logger.info("Running non-Latin detection...")
        run_non_latin(
            fonts_root=str(google_fonts_dir),
            output_dir="outputs/non_latin",
        )
        logger.info("Non-Latin analysis complete.")
    except ImportError as exc:
        logger.warning("src.non_latin not available: %s — skipping.", exc)
    except Exception as exc:
        logger.error("Non-Latin analysis failed: %s", exc)
        sys.exit(1)

    try:
        from src.non_latin_cluster import run as run_nl_cluster
        logger.info("Running non-Latin clustering...")
        run_nl_cluster(
            feature_dir="outputs",
            output_dir="outputs/non_latin",
        )
        logger.info("Non-Latin clustering complete.")
    except ImportError as exc:
        logger.warning("src.non_latin_cluster not available: %s — skipping.", exc)
    except Exception as exc:
        logger.error("Non-Latin clustering failed: %s", exc)
        sys.exit(1)

    logger.info("Phase 4 complete.")


if __name__ == "__main__":
    main()
