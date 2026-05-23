"""Smoke tests for src/outlines.py.

These tests check the invariants that bugs in outline extraction would break:
 - signed_area gives correct sign for known shapes
 - select_outer_contour picks the largest sub-path
 - resample_arc_length produces uniformly-spaced points along the curve

These do NOT require any font files to run.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.outlines import signed_area, select_outer_contour, resample_arc_length


def test_signed_area_ccw_square():
    """A unit square traversed counter-clockwise has area +1."""
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert signed_area(sq) == pytest.approx(1.0)


def test_signed_area_cw_square():
    """A unit square traversed clockwise has area -1."""
    sq = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    assert signed_area(sq) == pytest.approx(-1.0)


def test_select_outer_contour_picks_larger():
    """Given a big square and a small inner square, pick the big one."""
    big = [(0, 0), (10, 0), (10, 10), (0, 10)]
    small = [(3, 3), (4, 3), (4, 4), (3, 4)]
    outer = select_outer_contour([small, big])
    assert outer is not None
    assert abs(signed_area(outer)) == pytest.approx(100.0)


def test_select_outer_handles_empty():
    assert select_outer_contour([]) is None


def test_select_outer_skips_degenerate():
    """Sub-paths with fewer than 4 points are ignored."""
    degenerate = [(0, 0), (1, 0)]  # only 2 points
    good = [(0, 0), (10, 0), (10, 10), (0, 10)]
    outer = select_outer_contour([degenerate, good])
    assert outer is not None
    assert abs(signed_area(outer)) == pytest.approx(100.0)


def test_resample_arc_length_uniformity():
    """Resampling a unit circle should give equally-spaced points around it."""
    theta = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    resampled = resample_arc_length(circle, n_points=200)
    assert resampled.shape == (200, 2)
    # All points should be at radius ~1
    radii = np.sqrt((resampled**2).sum(axis=1))
    assert np.allclose(radii, 1.0, atol=1e-3)
    # Angular spacing should be uniform
    angles = np.arctan2(resampled[:, 1], resampled[:, 0])
    diffs = np.diff(np.unwrap(angles))
    assert diffs.std() < 1e-3


def test_resample_arc_length_handles_open_loop():
    """Function should auto-close an open contour."""
    # Square traversed without repeating the start
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    out = resample_arc_length(sq, n_points=100)
    assert out is not None
    assert out.shape == (100, 2)
    # Perimeter should be approximately 4 (sum of segment lengths)
    deltas = np.diff(np.vstack([out, out[0:1]]), axis=0)
    perimeter = np.sqrt((deltas**2).sum(axis=1)).sum()
    assert perimeter == pytest.approx(4.0, abs=0.05)


def test_resample_arc_length_zero_length_returns_none():
    """Degenerate contour (all points at one location) returns None."""
    degen = np.zeros((10, 2))
    assert resample_arc_length(degen) is None
