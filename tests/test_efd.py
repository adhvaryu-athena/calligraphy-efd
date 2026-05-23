"""Smoke tests for src/efd.py.

Validates that:
 - Feature vector has the expected dimensionality after normalization.
 - A circle has near-zero higher-order harmonics (it IS one ellipse, so n=1 dominates).
 - Same shape at different scales/positions gives the same feature vector (within tolerance).
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.efd import compute_efd_features


def _make_circle(radius=1.0, center=(0, 0), n=200, start_angle=0.0):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + start_angle
    return np.column_stack([
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta),
    ])


def test_feature_dim_normalize_true():
    """N=20 normalized -> 4*20 - 3 = 77 features."""
    circle = _make_circle(n=200)
    feats = compute_efd_features(circle, order=20, normalize=True)
    assert feats.shape == (77,)


def test_feature_dim_normalize_false():
    """N=20 unnormalized -> 4*20 = 80 features."""
    circle = _make_circle(n=200)
    feats = compute_efd_features(circle, order=20, normalize=False)
    assert feats.shape == (80,)


def test_normalization_invariance_to_scale():
    """Same shape, different sizes should give the same normalized features."""
    small = _make_circle(radius=1.0, n=200)
    big = _make_circle(radius=100.0, n=200)
    f1 = compute_efd_features(small, order=10, normalize=True)
    f2 = compute_efd_features(big, order=10, normalize=True)
    assert np.allclose(f1, f2, atol=1e-6)


def test_normalization_invariance_to_translation():
    """Same shape, different positions should give the same normalized features."""
    here = _make_circle(center=(0, 0), n=200)
    over_there = _make_circle(center=(50, -30), n=200)
    f1 = compute_efd_features(here, order=10, normalize=True)
    f2 = compute_efd_features(over_there, order=10, normalize=True)
    assert np.allclose(f1, f2, atol=1e-6)


def test_circle_has_low_higher_harmonics():
    """A circle is one ellipse: harmonics 2+ should be near-zero in magnitude."""
    circle = _make_circle(n=400)
    # Unnormalized so we can read raw amplitudes.
    feats = compute_efd_features(circle, order=10, normalize=False)
    # Layout: [a1,b1,c1,d1, a2,b2,c2,d2, ...]
    h1 = np.linalg.norm(feats[0:4])
    h2 = np.linalg.norm(feats[4:8])
    h5 = np.linalg.norm(feats[16:20])
    assert h2 / h1 < 0.05, f"h2/h1 = {h2/h1:.3f} should be near zero"
    assert h5 / h1 < 0.05, f"h5/h1 = {h5/h1:.3f} should be near zero"
