"""Phase 0 driver: extract EFD features in all configs and build font-level matrices."""
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fontTools.ttLib import TTFont

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.counters import extract_glyph_features
from src.aggregate import save_font_feature_matrix
from src.efd import build_feature_matrices

GLYPHS = "abcdefghijklmnopqrstuvwxyz"
N_HARMONICS = 20


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    fonts_csv = Path("outputs/fonts.csv")
    if not fonts_csv.exists():
        fonts_csv = Path("fonts.csv")
    if not fonts_csv.exists():
        logger.error("fonts.csv not found. Run the corpus build step first.")
        sys.exit(1)

    fonts_df = pd.read_csv(fonts_csv)
    logger.info("Loaded %d fonts from %s", len(fonts_df), fonts_csv)

    # Determine path column
    path_col = None
    for col in ("abs_path", "filepath", "font_path"):
        if col in fonts_df.columns:
            path_col = col
            break
    if path_col is None:
        logger.error(
            "No font path column found in fonts.csv. "
            "Expected one of: abs_path, filepath, font_path. "
            "Available columns: %s",
            list(fonts_df.columns),
        )
        sys.exit(1)

    # For each font+glyph, extract features in configs A, B, C
    glyph_feature_dict: dict = {"A": {}, "B": {}, "C": {}}
    n_processed = 0
    n_skipped = 0

    for _, row in fonts_df.iterrows():
        font_name = row["font_name"]
        font_path = row.get(path_col, "")

        if not font_path or not Path(str(font_path)).exists():
            logger.warning("Font not found: %s (path=%s)", font_name, font_path)
            n_skipped += 1
            continue

        try:
            font = TTFont(str(font_path))
        except Exception as exc:
            logger.warning("Cannot open font %s: %s", font_name, exc)
            n_skipped += 1
            continue

        cmap = font.getBestCmap()
        glyph_set = font.getGlyphSet()

        for config in ("A", "B", "C"):
            glyph_feature_dict[config][font_name] = []

        for char in GLYPHS:
            glyph_name = cmap.get(ord(char)) if cmap else None
            if glyph_name is None:
                continue
            try:
                glyph = glyph_set[glyph_name]
            except KeyError:
                continue

            for config in ("A", "B", "C"):
                try:
                    result = extract_glyph_features(
                        glyph, font, config=config, n_harmonics=N_HARMONICS
                    )
                    if result is not None:
                        glyph_feature_dict[config][font_name].append(result["features"])
                except Exception as exc:
                    logger.debug(
                        "extract_glyph_features failed for %s char=%s config=%s: %s",
                        font_name, char, config, exc,
                    )

        n_processed += 1
        logger.debug("Processed font: %s", font_name)

    n_with_features = len(
        [k for k, v in glyph_feature_dict["A"].items() if v]
    )
    logger.info(
        "Feature extraction complete: %d processed, %d skipped, %d have Config A features",
        n_processed, n_skipped, n_with_features,
    )

    if n_with_features == 0:
        logger.error(
            "No fonts with extractable features found. "
            "Check that font paths in fonts.csv are correct and fonts have Latin glyphs."
        )
        sys.exit(1)

    # Save font-level matrices
    output_dir = Path("outputs")
    try:
        saved = save_font_feature_matrix(
            fonts_df,
            glyph_feature_dict,
            output_dir=output_dir,
            configs=("A", "B", "C"),
            modes=("mean", "median", "concat", "weighted"),
        )
        logger.info("Saved %d font feature matrices:", len(saved))
        for key, path in saved.items():
            logger.info("  %s → %s", key, path)
    except Exception as exc:
        logger.error("save_font_feature_matrix failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
