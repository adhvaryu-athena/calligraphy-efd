"""Stage 2 — Glyph outline extraction.

For each (font, glyph) pair we want a single closed contour represented as a
fixed-length (n_points, 2) numpy array, ready for pyefd. The pipeline has three
steps:

    1.  Walk the glyph's Bezier description using fontTools' Pen Protocol.
        We sample N_PER_SEG points along every cubic or quadratic segment.

    2.  Pick the OUTER contour. Glyphs like 'a', 'b', 'd', 'e', 'g', 'o', 'p',
        'q' have multiple sub-paths (an outer ring plus an inner hole).
        We select the sub-path with the largest absolute signed area
        (shoelace formula). The lit review explicitly flags multi-contour
        glyphs as a known risk; this is the documented mitigation.

    3.  Resample the outer contour to exactly N_POINTS points equally spaced
        along ARC LENGTH (not Bezier parameter t). This is the subtle step
        the PRD glosses over: EFD assumes roughly uniform arc-length spacing,
        but Bezier `t` is non-uniform with respect to arc length, so we have
        to re-space explicitly. Without this, the first few harmonics are
        distorted.

Output of run(): a nested dict {font_name: {glyph_char: np.ndarray (N_POINTS, 2)}}.
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen


# Tunables
DEFAULT_N_POINTS = 200             # samples per glyph contour (PRD)
DEFAULT_POINTS_PER_SEGMENT = 25    # samples along each Bezier segment before resampling
DEFAULT_GLYPHS = "abcdefghijklmnopqrstuvwxyz"


class ContourPen(BasePen):
    """A pen that captures glyph outlines as dense lists of (x, y) points.

    Each sub-path becomes one entry in self.contours. Cubic and quadratic
    Bezier segments are sampled at `points_per_segment` parameter values
    using the standard parametric formulas. The output is intentionally
    over-sampled; arc-length resampling later reduces to the target count.
    """

    def __init__(self, glyphSet, points_per_segment: int = DEFAULT_POINTS_PER_SEGMENT):
        super().__init__(glyphSet)
        self.points_per_segment = points_per_segment
        self.contours: List[List[Tuple[float, float]]] = []
        self._current: List[Tuple[float, float]] = []

    # --- BasePen interface: we override the underscore versions ---

    def _moveTo(self, pt):
        if self._current:
            self.contours.append(self._current)
        self._current = [tuple(pt)]

    def _lineTo(self, pt):
        self._current.append(tuple(pt))

    def _curveToOne(self, pt1, pt2, pt3):
        """Cubic Bezier from the current point to pt3 via pt1, pt2."""
        p0 = self._current[-1]
        ts = np.linspace(0.0, 1.0, self.points_per_segment + 1)[1:]
        for t in ts:
            u = 1.0 - t
            x = u**3 * p0[0] + 3 * u**2 * t * pt1[0] + 3 * u * t**2 * pt2[0] + t**3 * pt3[0]
            y = u**3 * p0[1] + 3 * u**2 * t * pt1[1] + 3 * u * t**2 * pt2[1] + t**3 * pt3[1]
            self._current.append((x, y))

    def _qCurveToOne(self, pt1, pt2):
        """Quadratic Bezier from the current point to pt2 via pt1."""
        p0 = self._current[-1]
        ts = np.linspace(0.0, 1.0, self.points_per_segment + 1)[1:]
        for t in ts:
            u = 1.0 - t
            x = u**2 * p0[0] + 2 * u * t * pt1[0] + t**2 * pt2[0]
            y = u**2 * p0[1] + 2 * u * t * pt1[1] + t**2 * pt2[1]
            self._current.append((x, y))

    def _closePath(self):
        if self._current:
            self.contours.append(self._current)
        self._current = []

    def _endPath(self):
        self._closePath()


def signed_area(contour: np.ndarray) -> float:
    """Shoelace area. Positive = counter-clockwise, negative = clockwise.

    For closed contours the magnitude is the area enclosed. We use |area|
    to pick the largest sub-path as the outer contour.
    """
    pts = np.asarray(contour, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def select_outer_contour(contours: List[List[Tuple[float, float]]]) -> Optional[np.ndarray]:
    """Return the sub-path with the largest absolute signed area, as a numpy array.

    Falls back to None if `contours` is empty or all sub-paths are degenerate
    (fewer than 4 points).
    """
    arrays = [np.asarray(c, dtype=float) for c in contours if len(c) >= 4]
    if not arrays:
        return None
    areas = np.array([abs(signed_area(a)) for a in arrays])
    return arrays[int(np.argmax(areas))]


def resample_arc_length(contour: np.ndarray, n_points: int = DEFAULT_N_POINTS) -> Optional[np.ndarray]:
    """Resample a closed contour at `n_points` equally spaced along arc length.

    The input may or may not have the start point repeated at the end; we
    handle both cases. Output has shape (n_points, 2) with no repeated endpoint
    (the contour is implicitly closed by treating the last point as connected
    to the first).
    """
    pts = np.asarray(contour, dtype=float)
    # Ensure the loop is closed for interpolation
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0:1]])
    deltas = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((deltas**2).sum(axis=1))
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]
    if total_length <= 0:
        return None
    targets = np.linspace(0.0, total_length, n_points + 1)[:-1]
    xs = np.interp(targets, cum_lengths, pts[:, 0])
    ys = np.interp(targets, cum_lengths, pts[:, 1])
    return np.column_stack([xs, ys])


def extract_glyph_contour(
    font_path: str,
    glyph_char: str,
    n_points: int = DEFAULT_N_POINTS,
    points_per_segment: int = DEFAULT_POINTS_PER_SEGMENT,
) -> Optional[np.ndarray]:
    """Extract one glyph's outer contour as an (n_points, 2) array.

    Returns None if the font lacks the requested character, the glyph has no
    drawable outline (e.g., a space), or all sub-paths are degenerate.
    """
    font = TTFont(font_path)
    cmap = font.getBestCmap()
    glyph_name = cmap.get(ord(glyph_char))
    if glyph_name is None:
        return None
    glyph_set = font.getGlyphSet()
    glyph = glyph_set[glyph_name]
    pen = ContourPen(glyph_set, points_per_segment=points_per_segment)
    try:
        glyph.draw(pen)
    except Exception:
        return None
    outer = select_outer_contour(pen.contours)
    if outer is None:
        return None
    return resample_arc_length(outer, n_points=n_points)


def run(
    fonts_df: pd.DataFrame,
    glyphs: str = DEFAULT_GLYPHS,
    n_points: int = DEFAULT_N_POINTS,
    points_per_segment: int = DEFAULT_POINTS_PER_SEGMENT,
    output_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[Tuple[str, str, str]]]:
    """Stage 2 entry point. Returns (outlines_dict, failures_list).

    outlines_dict[font_name][glyph_char] -> np.ndarray of shape (n_points, 2)
    failures_list:   list of (font_name, glyph_char, reason) tuples.

    If output_path is given, pickles the outlines dict for downstream stages.
    """
    outlines: Dict[str, Dict[str, np.ndarray]] = {}
    failures: List[Tuple[str, str, str]] = []
    for _, row in fonts_df.iterrows():
        font_name = row["font_name"]
        path = row.get("abs_path", row["filepath"])
        outlines[font_name] = {}
        if not row.get("readable", True):
            failures.append((font_name, "*", "font not readable"))
            continue
        for glyph in glyphs:
            try:
                contour = extract_glyph_contour(
                    path, glyph, n_points=n_points,
                    points_per_segment=points_per_segment,
                )
            except Exception as e:
                failures.append((font_name, glyph, f"exception: {e}"))
                continue
            if contour is None:
                failures.append((font_name, glyph, "extraction returned None"))
                continue
            outlines[font_name][glyph] = contour
    if verbose:
        n_glyphs = sum(len(v) for v in outlines.values())
        print(f"Extracted {n_glyphs} glyph contours from {len(outlines)} fonts.")
        if failures:
            print(f"Failures: {len(failures)} (first 5 shown)")
            for f in failures[:5]:
                print(f"  {f}")
    if output_path is not None:
        with open(output_path, "wb") as f:
            pickle.dump(outlines, f)
        if verbose:
            print(f"Saved outlines to {output_path}")
    return outlines, failures
