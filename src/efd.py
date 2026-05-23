"""Stage 3 — Elliptic Fourier Descriptor feature extraction.

Wraps `pyefd.elliptic_fourier_descriptors` to convert each (n_points, 2)
contour into a 1-D feature vector. For each harmonic order N, the result is
flattened to (4 * N) values; when normalize=True, the first 3 entries become
constants (1.0, 0.0, 0.0) — pyefd's own docs say to drop them — so the actual
feature length is 4 * N - 3.

build_feature_matrices() computes the matrix at multiple orders in one pass
so Stage 4 (classification) and Stage 5 (sensitivity curve) can re-use it
without re-extraction.

Feature Configs
---------------
This module implements Config A (outer contour EFD only).  Counter-aware
feature extraction (Configs B and C) is implemented in src/counters.py:

  Config A: outer EFD only.         Feature length = 4*N - 3.
  Config B: outer EFD + scalars.    Feature length = 4*N.
            Scalars = [n_tc, counter_area_ratio, n_comp].
  Config C: outer + counter EFD + scalars.  Feature length = 8*N - 3.
            Falls back to Config B (zero-padded) when n_tc != 1.

The run() function here accepts a config= parameter for forward compatibility
but only implements Config A natively; passing "B" or "C" logs a warning and
delegates to Config A behaviour.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pyefd import elliptic_fourier_descriptors

_logger = logging.getLogger(__name__)


DEFAULT_ORDERS: Tuple[int, ...] = (5, 10, 15, 20, 30, 40)
DEFAULT_PRIMARY_ORDER = 20


def compute_efd_features(contour: np.ndarray, order: int = DEFAULT_PRIMARY_ORDER,
                         normalize: bool = True) -> np.ndarray:
    """Compute one glyph's EFD feature vector.

    Parameters
    ----------
    contour : (n_points, 2) array of contour points.
    order   : harmonic order N. Output dimension is 4*N - 3 when normalize=True.
    normalize : if True, applies size/rotation/starting-point invariant
                normalization (Kuhl & Giardina). The first three coefficients
                become constants and are dropped.

    Returns
    -------
    1-D float array of length (4*order - 3) when normalize=True else (4*order).
    """
    coeffs = elliptic_fourier_descriptors(contour, order=order, normalize=normalize)
    flat = coeffs.flatten()
    return flat[3:] if normalize else flat


def build_feature_matrices(
    outlines: Dict[str, Dict[str, np.ndarray]],
    label_map: Dict[str, str],
    orders: Tuple[int, ...] = DEFAULT_ORDERS,
    normalize: bool = True,
) -> Dict[str, Dict]:
    """Build feature matrices at every requested harmonic order in one pass.

    Returns a dict keyed by order:
        {
          order: {
            "X": (n_samples, 4*order - 3) feature matrix,
            "y": (n_samples,) class labels (as strings),
            "groups": (n_samples,) font_name per sample (for GroupKFold),
            "glyphs": (n_samples,) glyph char per sample,
            "fonts":  (n_samples,) font_name per sample (same as groups but explicit),
          }
        }
    """
    # First pass: enumerate all (font, glyph) pairs that have outlines AND a label
    samples = []  # list of (font_name, glyph_char, label)
    for font_name, glyph_dict in outlines.items():
        label = label_map.get(font_name)
        if label is None:
            continue
        for glyph_char, contour in glyph_dict.items():
            samples.append((font_name, glyph_char, label, contour))

    if not samples:
        raise ValueError("No (font, glyph) samples with both outlines and labels.")

    n = len(samples)
    fonts = np.array([s[0] for s in samples])
    glyphs = np.array([s[1] for s in samples])
    y = np.array([s[2] for s in samples])

    # Second pass: compute features per order
    out: Dict[int, Dict] = {}
    for order in orders:
        feat_dim = 4 * order - 3 if normalize else 4 * order
        X = np.zeros((n, feat_dim), dtype=float)
        for i, (_, _, _, contour) in enumerate(samples):
            X[i] = compute_efd_features(contour, order=order, normalize=normalize)
        out[order] = {
            "X": X,
            "y": y,
            "groups": fonts,
            "glyphs": glyphs,
            "fonts": fonts,
        }
    return out


def run(
    outlines: Dict[str, Dict[str, np.ndarray]],
    label_map: Dict[str, str],
    orders: Tuple[int, ...] = DEFAULT_ORDERS,
    normalize: bool = True,
    output_path: str | Path | None = None,
    verbose: bool = True,
    config: str = "A",
) -> Dict[int, Dict]:
    """Stage 3 entry point. Returns dict keyed by harmonic order.

    If output_path is given, saves as a single .npz with arrays:
        X_{order}, y, groups, glyphs, and a meta JSON in the file's .info attr.

    Parameters
    ----------
    config : "A", "B", or "C".  Only "A" is natively implemented here.
             Passing "B" or "C" logs a warning and proceeds as Config A for
             backward compatibility.  Use src.counters.extract_glyph_features()
             for counter-aware extraction.
    """
    if config in ("B", "C"):
        _logger.warning(
            "Config %s not yet implemented in efd.run(); "
            "use counters.extract_glyph_features() directly.  Proceeding as Config A.",
            config,
        )
    feats = build_feature_matrices(outlines, label_map, orders=orders, normalize=normalize)
    if verbose:
        sample_order = next(iter(feats))
        info = feats[sample_order]
        print(f"Feature extraction: {len(info['y'])} samples")
        print(f"  classes: {sorted(set(info['y']))}")
        print(f"  fonts:   {len(set(info['fonts']))}")
        print("  feature dimension per order:")
        for o in orders:
            print(f"    N={o}: {feats[o]['X'].shape[1]} features")

    if output_path is not None:
        save_dict = {f"X_{o}": feats[o]["X"] for o in orders}
        # y, groups, glyphs are the same across orders — store once.
        first = next(iter(feats.values()))
        save_dict["y"] = first["y"]
        save_dict["groups"] = first["groups"]
        save_dict["glyphs"] = first["glyphs"]
        save_dict["orders"] = np.array(orders)
        np.savez(output_path, **save_dict)
        if verbose:
            print(f"Saved feature matrices to {output_path}")
    return feats


def load(npz_path: str | Path) -> Dict[int, Dict]:
    """Inverse of run() with output_path: rebuild the per-order dict."""
    data = np.load(npz_path, allow_pickle=True)
    orders = data["orders"].tolist()
    out: Dict[int, Dict] = {}
    for o in orders:
        out[o] = {
            "X": data[f"X_{o}"],
            "y": data["y"],
            "groups": data["groups"],
            "glyphs": data["glyphs"],
            "fonts": data["groups"],
        }
    return out
