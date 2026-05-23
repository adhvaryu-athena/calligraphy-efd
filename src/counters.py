"""Counter-aware glyph feature extraction (Configs B and C).

Config A: outer EFD only (delegated to efd.py).
Config B: outer EFD + [n_tc, counter_area_ratio, n_comp]. Shape: (4*n,)
          where n is the harmonic order.  The outer EFD gives 4*n-3 dims,
          plus 3 scalar features = 4*n total.
Config C: outer EFD + counter EFD (n_tc==1 only) + scalars. Shape: (8*n-3,).
          Falls back to Config B silently for n_tc != 1, padding with zeros
          to reach (8*n-3,) shape.  The returned dict has "fallback": True in
          those cases.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from src.efd import compute_efd_features
from src.outlines import (
    classify_subpaths,
    resample_arc_length,
    DEFAULT_N_POINTS,
    DEFAULT_POINTS_PER_SEGMENT,
)

_logger = logging.getLogger(__name__)


def extract_glyph_features(
    glyph,
    font,
    config: str = "A",
    n_harmonics: int = 20,
    n_points: int = DEFAULT_N_POINTS,
) -> Optional[Dict]:
    """Return a feature dict for one glyph under the specified config.

    Parameters
    ----------
    glyph       : glyph object from font.getGlyphSet()
    font        : the TTFont object
    config      : "A", "B", or "C"
    n_harmonics : harmonic order N; EFD feature length = 4*N-3 (normalized)
    n_points    : number of arc-length-resampled contour points

    Returns
    -------
    Config A : {"features": np.array shape (4*N-3,), "config": "A",
                "n_tc": int, "n_comp": int}
    Config B : {"features": np.array shape (4*N,),   "config": "B",
                "n_tc": int, "n_comp": int, "counter_area_ratio": float}
    Config C : {"features": np.array shape (8*N-3,), "config": "C",
                "n_tc": int, "n_comp": int, "counter_area_ratio": float,
                "fallback": bool}
    None on extraction failure.
    """
    if config not in ("A", "B", "C"):
        raise ValueError(f"config must be 'A', 'B', or 'C', got {config!r}")

    info = classify_subpaths(glyph, font)
    if info is None:
        _logger.debug("extract_glyph_features: classify_subpaths returned None")
        return None

    outer_arr = info["outer"]
    if outer_arr is None:
        _logger.debug("extract_glyph_features: outer contour is None")
        return None

    outer_resampled = resample_arc_length(outer_arr, n_points=n_points)
    if outer_resampled is None:
        _logger.debug("extract_glyph_features: resample_arc_length returned None for outer")
        return None

    try:
        outer_efd = compute_efd_features(outer_resampled, order=n_harmonics, normalize=True)
    except Exception as exc:
        _logger.warning("extract_glyph_features: outer EFD failed: %s", exc)
        return None

    n_tc = info["n_tc"]
    n_comp = info["n_comp"]
    outer_area = info["outer_area"]
    total_counter_area = info["total_counter_area"]
    counter_area_ratio = (
        total_counter_area / outer_area if outer_area > 0.0 else 0.0
    )

    # ---- Config A ----
    if config == "A":
        return {
            "features": outer_efd,
            "config": "A",
            "n_tc": n_tc,
            "n_comp": n_comp,
        }

    # Scalar block shared by B and C: [n_tc, counter_area_ratio, n_comp]
    scalars = np.array([float(n_tc), counter_area_ratio, float(n_comp)], dtype=float)

    # ---- Config B ----
    if config == "B":
        features_b = np.concatenate([outer_efd, scalars])  # shape (4*N,)
        return {
            "features": features_b,
            "config": "B",
            "n_tc": n_tc,
            "n_comp": n_comp,
            "counter_area_ratio": counter_area_ratio,
        }

    # ---- Config C ----
    # Target shape: outer_efd (4N-3) + counter_efd (4N-3) + scalars (3) = 8N-3
    target_len = 8 * n_harmonics - 3

    if n_tc == 1:
        counter_arr = info["true_counters"][0]
        counter_resampled = resample_arc_length(counter_arr, n_points=n_points)
        if counter_resampled is None:
            _logger.warning(
                "extract_glyph_features: resample failed for true counter — falling back to B"
            )
            fallback = True
            features_c = np.zeros(target_len, dtype=float)
            b_len = len(outer_efd) + len(scalars)  # 4N
            features_c[:b_len] = np.concatenate([outer_efd, scalars])
        else:
            try:
                counter_efd = compute_efd_features(
                    counter_resampled, order=n_harmonics, normalize=True
                )
                features_c = np.concatenate([outer_efd, counter_efd, scalars])
                fallback = False
            except Exception as exc:
                _logger.warning(
                    "extract_glyph_features: counter EFD failed (%s) — falling back to B", exc
                )
                fallback = True
                features_c = np.zeros(target_len, dtype=float)
                b_len = len(outer_efd) + len(scalars)
                features_c[:b_len] = np.concatenate([outer_efd, scalars])
    else:
        # n_tc != 1 — fall back to Config B, padded to target_len
        fallback = True
        features_c = np.zeros(target_len, dtype=float)
        b_len = len(outer_efd) + len(scalars)  # 4N
        features_c[:b_len] = np.concatenate([outer_efd, scalars])

    return {
        "features": features_c,
        "config": "C",
        "n_tc": n_tc,
        "n_comp": n_comp,
        "counter_area_ratio": counter_area_ratio,
        "fallback": fallback,
    }


def compute_pattern_intensity(font_glyph_results: List[Optional[Dict]]) -> float:
    """Return max_n_components as a scalar float.

    Parameters
    ----------
    font_glyph_results : list of dicts returned by extract_glyph_features
                         (None entries are safely ignored)

    Returns
    -------
    float — maximum n_comp value seen across all glyphs, 0.0 if none found.
    """
    max_comp = max(
        (r["n_comp"] for r in font_glyph_results if r is not None and "n_comp" in r),
        default=0,
    )
    return float(max_comp)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    # Minimal smoke test: load first available font from outputs/fonts.csv
    try:
        import pandas as pd
        from fontTools.ttLib import TTFont

        csv_path = Path("outputs/fonts.csv")
        if not csv_path.exists():
            csv_path = Path("fonts.csv")
        if not csv_path.exists():
            print("No fonts.csv found — skipping self-test.")
            sys.exit(0)

        df = pd.read_csv(csv_path)
        row = df.dropna(subset=["abs_path"] if "abs_path" in df.columns else ["filepath"]).iloc[0]
        font_path = row.get("abs_path", row["filepath"])
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        glyph_name = cmap.get(ord("o"))
        if glyph_name is None:
            print("Glyph 'o' not found — skipping.")
            sys.exit(0)
        glyph = font.getGlyphSet()[glyph_name]

        for cfg in ("A", "B", "C"):
            result = extract_glyph_features(glyph, font, config=cfg, n_harmonics=10)
            if result is None:
                print(f"Config {cfg}: extraction failed")
            else:
                print(
                    f"Config {cfg}: shape={result['features'].shape}, "
                    f"n_tc={result['n_tc']}, n_comp={result['n_comp']}, "
                    f"fallback={result.get('fallback', False)}"
                )

        intensity = compute_pattern_intensity([
            extract_glyph_features(glyph, font, config="B", n_harmonics=10)
        ])
        print(f"Pattern intensity (max n_comp): {intensity}")

    except Exception as e:
        print(f"Self-test error: {e}")
        raise
