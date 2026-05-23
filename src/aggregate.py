"""Font-level feature aggregation.

Aggregates per-glyph EFD feature vectors to a single font-level vector.

The glyph_feature_dict passed to save_font_feature_matrix() is expected to
have the structure:

    {config: {font_name: [np.array, ...]}}

where each list entry is the "features" array from a counters.extract_glyph_features()
(or efd.compute_efd_features()) call for one glyph.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def _pad_to_matrix(glyph_features: List[np.ndarray]) -> np.ndarray:
    """Stack a list of 1-D arrays into a 2-D matrix, zero-padding shorter rows."""
    max_len = max(arr.shape[0] for arr in glyph_features)
    rows = []
    for arr in glyph_features:
        if arr.shape[0] < max_len:
            padded = np.zeros(max_len, dtype=float)
            padded[: arr.shape[0]] = arr
            rows.append(padded)
        else:
            rows.append(arr.astype(float))
    return np.vstack(rows)


def aggregate_to_font_level(
    glyph_features,
    mode: str = "mean",
) -> np.ndarray:
    """Aggregate per-glyph feature vectors to a single font-level vector.

    Parameters
    ----------
    glyph_features : list of 1-D np.arrays (may have different lengths) OR
                     2-D np.array of shape (n_glyphs, n_features)
    mode           : one of "mean", "median", "concat", "weighted"

        "mean"     — np.mean across glyphs (axis=0)
        "median"   — np.median across glyphs (axis=0)
        "concat"   — flatten all glyph vectors into one long vector
                     (arrays are zero-padded to equal length before stacking)
        "weighted" — weight each glyph vector by its L2 norm (proxy for how
                     distinctive the glyph is); weights are normalised to sum 1;
                     returns the weighted sum

    Returns
    -------
    np.array of shape (n_features,) for mean/median/weighted, or
    (n_glyphs * n_features,) for concat.
    """
    if mode not in ("mean", "median", "concat", "weighted"):
        raise ValueError(f"mode must be 'mean', 'median', 'concat', or 'weighted', got {mode!r}")

    # Normalise input
    if isinstance(glyph_features, np.ndarray) and glyph_features.ndim == 2:
        matrix = glyph_features.astype(float)
    else:
        arrays = [np.asarray(a, dtype=float).ravel() for a in glyph_features]
        if not arrays:
            _logger.warning("aggregate_to_font_level: empty glyph_features list")
            return np.array([], dtype=float)
        if mode == "concat":
            # For concat we don't need a uniform matrix
            max_len = max(a.shape[0] for a in arrays)
            padded = []
            for a in arrays:
                if a.shape[0] < max_len:
                    p = np.zeros(max_len, dtype=float)
                    p[: a.shape[0]] = a
                    padded.append(p)
                else:
                    padded.append(a)
            return np.concatenate(padded)
        matrix = _pad_to_matrix(arrays)

    if mode == "mean":
        return np.mean(matrix, axis=0)
    elif mode == "median":
        return np.median(matrix, axis=0)
    elif mode == "concat":
        # Already handled above for list input; handle ndarray case
        return matrix.flatten()
    elif mode == "weighted":
        norms = np.linalg.norm(matrix, axis=1)
        total = norms.sum()
        if total == 0.0:
            weights = np.ones(len(norms)) / max(len(norms), 1)
        else:
            weights = norms / total
        return (matrix * weights[:, np.newaxis]).sum(axis=0)


# ---------------------------------------------------------------------------
# Save font feature matrix
# ---------------------------------------------------------------------------

def save_font_feature_matrix(
    fonts_df: pd.DataFrame,
    glyph_feature_dict: Dict[str, Dict[str, List[np.ndarray]]],
    output_dir: str | Path,
    configs: Tuple[str, ...] = ("A", "B", "C"),
    modes: Tuple[str, ...] = ("mean", "median", "concat", "weighted"),
    label_col: str = "style_class",
) -> Dict[str, Path]:
    """Build and save font-level feature matrices for every (config, mode) combo.

    Parameters
    ----------
    fonts_df            : DataFrame with at least "font_name" and label_col columns
    glyph_feature_dict  : {config: {font_name: [np.array, ...]}}
                          where each array is a per-glyph feature vector
    output_dir          : directory to write .npz files
    configs             : which configs to process (only those present in the dict)
    modes               : aggregation modes to apply
    label_col           : column name in fonts_df for class labels

    Returns
    -------
    dict mapping "{config}_{mode}" → Path of saved .npz file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if label_col not in fonts_df.columns:
        # Try common alternatives
        for alt in ("label", "class", "style"):
            if alt in fonts_df.columns:
                label_col = alt
                _logger.warning("label_col not found; using %r instead", alt)
                break
        else:
            raise ValueError(
                f"Column {label_col!r} not found in fonts_df.  "
                f"Available columns: {list(fonts_df.columns)}"
            )

    saved: Dict[str, Path] = {}

    for config in configs:
        if config not in glyph_feature_dict:
            _logger.info("Config %s not found in glyph_feature_dict — skipping", config)
            continue

        font_arrays_by_name = glyph_feature_dict[config]

        for mode in modes:
            font_names_out: List[str] = []
            labels_out: List[str] = []
            font_vectors: List[np.ndarray] = []

            for _, row in fonts_df.iterrows():
                font_name = row["font_name"]
                label = row.get(label_col)
                if label is None:
                    _logger.debug("Skipping font %r — no label", font_name)
                    continue

                glyph_list = font_arrays_by_name.get(font_name)
                if not glyph_list:
                    _logger.debug(
                        "Skipping font %r in config %s — no glyph features", font_name, config
                    )
                    continue

                # Filter None entries
                valid = [a for a in glyph_list if a is not None]
                if not valid:
                    _logger.debug(
                        "Skipping font %r — all glyph feature entries are None", font_name
                    )
                    continue

                try:
                    font_vec = aggregate_to_font_level(valid, mode=mode)
                except Exception as exc:
                    _logger.warning(
                        "aggregate_to_font_level failed for font %r config %s mode %s: %s",
                        font_name, config, mode, exc,
                    )
                    continue

                font_names_out.append(font_name)
                labels_out.append(str(label))
                font_vectors.append(font_vec)

            if not font_vectors:
                _logger.warning(
                    "No font vectors produced for config=%s mode=%s — skipping save",
                    config, mode,
                )
                continue

            # Pad to uniform width (relevant for concat mode where different
            # fonts may have contributed different numbers of glyphs)
            max_len = max(v.shape[0] for v in font_vectors)
            X_rows = []
            for v in font_vectors:
                if v.shape[0] < max_len:
                    padded = np.zeros(max_len, dtype=float)
                    padded[: v.shape[0]] = v
                    X_rows.append(padded)
                else:
                    X_rows.append(v.astype(float))

            X = np.vstack(X_rows)
            font_names_arr = np.array(font_names_out)
            labels_arr = np.array(labels_out)

            out_path = output_dir / f"font_features_{config}_{mode}.npz"
            np.savez(
                out_path,
                X=X,
                font_names=font_names_arr,
                labels=labels_arr,
            )
            _logger.info(
                "Saved config=%s mode=%s  shape=%s → %s",
                config, mode, X.shape, out_path,
            )
            key = f"{config}_{mode}"
            saved[key] = out_path

    return saved


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Build synthetic glyph features for 3 fonts, 2 glyphs each, 2 configs
    rng = np.random.default_rng(42)

    n_harmonics = 10
    dim_a = 4 * n_harmonics - 3   # 37
    dim_b = 4 * n_harmonics        # 40
    dim_c = 8 * n_harmonics - 3   # 77

    fonts_df = pd.DataFrame({
        "font_name": ["FontA", "FontB", "FontC"],
        "style_class": ["serif", "sans-serif", "serif"],
    })

    glyph_feature_dict: Dict[str, Dict[str, List[np.ndarray]]] = {
        "A": {
            "FontA": [rng.standard_normal(dim_a), rng.standard_normal(dim_a)],
            "FontB": [rng.standard_normal(dim_a)],
            "FontC": [rng.standard_normal(dim_a), rng.standard_normal(dim_a)],
        },
        "B": {
            "FontA": [rng.standard_normal(dim_b), rng.standard_normal(dim_b)],
            "FontB": [rng.standard_normal(dim_b)],
            "FontC": [rng.standard_normal(dim_b)],
        },
        "C": {
            "FontA": [rng.standard_normal(dim_c), rng.standard_normal(dim_b)],  # mixed (fallback)
            "FontB": [rng.standard_normal(dim_c)],
            "FontC": [rng.standard_normal(dim_c)],
        },
    }

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        saved = save_font_feature_matrix(
            fonts_df, glyph_feature_dict, tmpdir,
            configs=("A", "B", "C"), modes=("mean", "median", "concat", "weighted"),
        )
        for key, path in saved.items():
            data = np.load(path)
            print(f"  {key}: X.shape={data['X'].shape}, labels={data['labels'].tolist()}")

    # Quick unit tests for aggregate_to_font_level
    arrs = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    assert np.allclose(aggregate_to_font_level(arrs, mode="mean"), [2.5, 3.5, 4.5])
    assert np.allclose(aggregate_to_font_level(arrs, mode="median"), [2.5, 3.5, 4.5])
    concat_result = aggregate_to_font_level(arrs, mode="concat")
    assert concat_result.shape == (6,), f"Expected (6,), got {concat_result.shape}"

    # weighted: norms are sqrt(14) and sqrt(77); should not crash
    w_result = aggregate_to_font_level(arrs, mode="weighted")
    assert w_result.shape == (3,)

    # mixed lengths
    mixed = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
    mean_mixed = aggregate_to_font_level(mixed, mode="mean")
    assert mean_mixed.shape == (3,)

    print("All self-tests passed.")
