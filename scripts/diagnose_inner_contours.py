"""Diagnostic 1 — inner contour distribution across the corpus.

For each glyph in fonts.csv, re-walks the font with ContourPen and counts how many
inner sub-paths the glyph has (after the degenerate-sub-path filter that
select_outer_contour applies).

Used to lock the counter-EFD extraction rule:
    n_inner == 0           -> outer-only feature vector
    n_inner == 1           -> include counter EFD (unambiguous case)
    n_inner >= 2           -> skip counter EFD; record scalar stats only

Output:
    outputs/diagnostic_inner_contours.csv  — one row per (font, glyph)
    Printed summary: distribution overall, by style class, by glyph character.

Run from repo root, inside the conda env:
    python scripts/diagnose_inner_contours.py
"""
from __future__ import annotations
import sys
import csv
from collections import defaultdict
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fontTools.ttLib import TTFont
from src.corpus import load_corpus, resolve_paths, validate_readable
from src.outlines import ContourPen, signed_area

DEFAULT_GLYPHS = "abcdefghijklmnopqrstuvwxyz"


def count_inner(font_path: str, glyph_char: str):
    """Return (n_inner_after_degenerate_filter, n_inner_raw) or (None, None)."""
    try:
        font = TTFont(font_path)
    except Exception:
        return None, None
    cmap = font.getBestCmap()
    name = cmap.get(ord(glyph_char))
    if name is None:
        return None, None
    glyph_set = font.getGlyphSet()
    glyph = glyph_set[name]
    pen = ContourPen(glyph_set)
    try:
        glyph.draw(pen)
    except Exception:
        return None, None
    if not pen.contours:
        return None, None
    valid = [c for c in pen.contours if len(c) >= 4]
    if not valid:
        return None, None
    n_inner_valid = len(valid) - 1  # one of them is the outer
    n_inner_raw = len(pen.contours) - 1
    return n_inner_valid, n_inner_raw


def main():
    df = load_corpus("fonts.csv")
    df = resolve_paths(df, ".")
    df = validate_readable(df)
    df = df[df["readable"]].reset_index(drop=True)
    print(f"Walking {len(df)} fonts x {len(DEFAULT_GLYPHS)} glyphs = "
          f"{len(df) * len(DEFAULT_GLYPHS)} extractions...")

    rows = []
    overall = defaultdict(int)
    by_class = defaultdict(lambda: defaultdict(int))
    by_glyph = defaultdict(lambda: defaultdict(int))
    failures = 0

    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  font {i+1}/{len(df)}: {row['font_name']}")
        for g in DEFAULT_GLYPHS:
            n_valid, n_raw = count_inner(row["abs_path"], g)
            if n_valid is None:
                failures += 1
                continue
            rows.append({
                "font": row["font_name"],
                "style_class": row["style_class"],
                "glyph": g,
                "n_inner_valid": n_valid,
                "n_inner_raw": n_raw,
            })
            overall[n_valid] += 1
            by_class[row["style_class"]][n_valid] += 1
            by_glyph[g][n_valid] += 1

    Path("outputs").mkdir(exist_ok=True)
    out_path = "outputs/diagnostic_inner_contours.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["font", "style_class", "glyph",
                                          "n_inner_valid", "n_inner_raw"])
        w.writeheader()
        w.writerows(rows)

    total_ok = sum(overall.values())

    print("\n" + "=" * 60)
    print(f"Overall n_inner distribution (post-degenerate-filter)")
    print("=" * 60)
    print(f"  Total successful extractions: {total_ok}")
    print(f"  Total extraction failures:    {failures}")
    for n in sorted(overall):
        pct = 100 * overall[n] / total_ok
        print(f"  n_inner = {n}: {overall[n]:>6d} glyphs  ({pct:5.1f}%)")

    print("\n" + "=" * 60)
    print("By style class")
    print("=" * 60)
    for cls in sorted(by_class):
        total = sum(by_class[cls].values())
        print(f"  {cls} ({total} glyphs)")
        for n in sorted(by_class[cls]):
            pct = 100 * by_class[cls][n] / total
            print(f"    n_inner = {n}: {by_class[cls][n]:>5d}  ({pct:5.1f}%)")

    print("\n" + "=" * 60)
    print("By glyph character — uniform (always same n_inner)")
    print("=" * 60)
    for g in DEFAULT_GLYPHS:
        n_vals = by_glyph[g]
        if len(n_vals) == 1:
            n = list(n_vals.keys())[0]
            total = sum(n_vals.values())
            print(f"  '{g}': always n_inner = {n}  (across {total} fonts)")

    print("\n" + "=" * 60)
    print("By glyph character — variable (n_inner differs across fonts)")
    print("=" * 60)
    for g in DEFAULT_GLYPHS:
        n_vals = by_glyph[g]
        if len(n_vals) > 1:
            total = sum(n_vals.values())
            dist = ", ".join(f"n={n}:{c}({100*c/total:.0f}%)"
                              for n, c in sorted(n_vals.items()))
            print(f"  '{g}': {dist}  (total {total} fonts)")

    print(f"\nWrote {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
