"""Diagnostic 1 (refined) — TRUE counter contour distribution.

Replaces the naive "n_inner = all non-outer sub-paths" rule with the
typographically correct one: true counters are sub-paths with OPPOSITE
winding direction from the outer (PostScript/TrueType convention for holes).
Same-winding non-outer sub-paths are SEPARATE COMPONENTS (e.g., i's tittle,
j's tittle, stippled patterns in decorative fonts) — geometrically real but
not counters in the typographic sense.

Output:
    outputs/diagnostic_inner_contours_refined.csv
    Columns: font, style_class, glyph, n_true_counters, n_components, n_inner_raw

Run from repo root, inside the conda env:
    python scripts/diagnose_inner_contours_refined.py
"""
from __future__ import annotations
import sys
import csv
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fontTools.ttLib import TTFont
from src.corpus import load_corpus, resolve_paths, validate_readable
from src.outlines import ContourPen, signed_area

DEFAULT_GLYPHS = "abcdefghijklmnopqrstuvwxyz"


def classify_subpaths(font_path: str, glyph_char: str):
    """Return dict with n_true_counters, n_components, n_inner_raw, or None on failure."""
    try:
        font = TTFont(font_path)
    except Exception:
        return None
    cmap = font.getBestCmap()
    name = cmap.get(ord(glyph_char))
    if name is None:
        return None
    glyph_set = font.getGlyphSet()
    glyph = glyph_set[name]
    pen = ContourPen(glyph_set)
    try:
        glyph.draw(pen)
    except Exception:
        return None
    if not pen.contours:
        return None
    valid = [c for c in pen.contours if len(c) >= 4]
    if not valid:
        return None

    # Compute signed areas
    areas = [signed_area(c) for c in valid]
    abs_areas = [abs(a) for a in areas]
    outer_idx = abs_areas.index(max(abs_areas))
    outer_sign = 1 if areas[outer_idx] >= 0 else -1

    # Walk non-outer sub-paths and classify by sign
    n_true_counters = 0
    n_components = 0
    for i, a in enumerate(areas):
        if i == outer_idx:
            continue
        sign = 1 if a >= 0 else -1
        if sign == -outer_sign:
            n_true_counters += 1
        else:
            n_components += 1

    n_inner_raw = len(pen.contours) - 1
    return {
        "n_true_counters": n_true_counters,
        "n_components": n_components,
        "n_inner_raw": n_inner_raw,
    }


def main():
    df = load_corpus("fonts.csv")
    df = resolve_paths(df, ".")
    df = validate_readable(df)
    df = df[df["readable"]].reset_index(drop=True)
    print(f"Walking {len(df)} fonts x {len(DEFAULT_GLYPHS)} glyphs = "
          f"{len(df) * len(DEFAULT_GLYPHS)} extractions...")

    rows = []
    overall_tc = defaultdict(int)         # n_true_counters
    overall_comp = defaultdict(int)       # n_components
    by_class_tc = defaultdict(lambda: defaultdict(int))
    by_glyph_tc = defaultdict(lambda: defaultdict(int))
    failures = 0

    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  font {i+1}/{len(df)}: {row['font_name']}")
        for g in DEFAULT_GLYPHS:
            res = classify_subpaths(row["abs_path"], g)
            if res is None:
                failures += 1
                continue
            rows.append({
                "font": row["font_name"],
                "style_class": row["style_class"],
                "glyph": g,
                **res,
            })
            overall_tc[res["n_true_counters"]] += 1
            overall_comp[res["n_components"]] += 1
            by_class_tc[row["style_class"]][res["n_true_counters"]] += 1
            by_glyph_tc[g][res["n_true_counters"]] += 1

    Path("outputs").mkdir(exist_ok=True)
    out_path = "outputs/diagnostic_inner_contours_refined.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["font", "style_class", "glyph",
                                          "n_true_counters", "n_components",
                                          "n_inner_raw"])
        w.writeheader()
        w.writerows(rows)

    total_ok = sum(overall_tc.values())

    print("\n" + "=" * 70)
    print("Overall n_true_counters distribution (opposite-sign sub-paths only)")
    print("=" * 70)
    print(f"  Total successful extractions: {total_ok}")
    print(f"  Total extraction failures:    {failures}")
    for n in sorted(overall_tc):
        pct = 100 * overall_tc[n] / total_ok
        print(f"  n_true_counters = {n}: {overall_tc[n]:>6d}  ({pct:5.1f}%)")

    print("\n--- Rule case breakdown (refined) ---")
    n0 = overall_tc.get(0, 0)
    n1 = overall_tc.get(1, 0)
    n2plus = sum(c for n, c in overall_tc.items() if n >= 2)
    print(f"  Case A (n_tc=0,   outer only):    {n0:>6d}  ({100*n0/total_ok:5.1f}%)")
    print(f"  Case B (n_tc=1,   include EFD):   {n1:>6d}  ({100*n1/total_ok:5.1f}%)")
    print(f"  Case C (n_tc>=2,  scalar only):   {n2plus:>6d}  ({100*n2plus/total_ok:5.1f}%)")

    print("\n" + "=" * 70)
    print("Overall n_components distribution (same-sign non-outer sub-paths)")
    print("=" * 70)
    cumulative = 0
    for n in sorted(overall_comp):
        if n == 0 and overall_comp[n] / total_ok > 0.5:
            pct = 100 * overall_comp[n] / total_ok
            print(f"  n_components = {n}: {overall_comp[n]:>6d}  ({pct:5.1f}%)")
        elif overall_comp[n] >= 50:
            pct = 100 * overall_comp[n] / total_ok
            print(f"  n_components = {n}: {overall_comp[n]:>6d}  ({pct:5.1f}%)")
        else:
            cumulative += overall_comp[n]
    if cumulative > 0:
        pct = 100 * cumulative / total_ok
        print(f"  n_components >= rare-tail: {cumulative:>6d}  ({pct:5.1f}%)")

    print("\n" + "=" * 70)
    print("By style class — n_true_counters distribution")
    print("=" * 70)
    for cls in sorted(by_class_tc):
        total = sum(by_class_tc[cls].values())
        line = f"  {cls:>14s} (n={total:5d}): "
        for n in sorted(by_class_tc[cls]):
            pct = 100 * by_class_tc[cls][n] / total
            if pct >= 0.5:
                line += f"  n={n}: {pct:4.1f}%"
        print(line)

    print("\n" + "=" * 70)
    print("By glyph character — n_true_counters distribution")
    print("=" * 70)
    print(f"  {'glyph':>6s}  {'n=0':>7s}  {'n=1':>7s}  {'n=2':>7s}  {'n>=3':>7s}   sample")
    glyph_summary = []
    for g in DEFAULT_GLYPHS:
        c = by_glyph_tc[g]
        t = sum(c.values())
        if t == 0:
            continue
        dist = {k: 100 * v / t for k, v in c.items()}
        glyph_summary.append((g, t, dist))
    glyph_summary.sort(key=lambda x: -x[2].get(1, 0))
    for g, t, dist in glyph_summary:
        n0 = dist.get(0, 0)
        n1 = dist.get(1, 0)
        n2 = dist.get(2, 0)
        n3plus = sum(v for k, v in dist.items() if k >= 3)
        print(f"  {g!r:>6s}  {n0:6.1f}%  {n1:6.1f}%  {n2:6.1f}%  {n3plus:6.1f}%   {t:>5d}")

    # Per-font: how many true-counter-bearing glyphs (n_tc>=1) does each font have?
    by_font_tc1 = defaultdict(int)
    for r in rows:
        if r["n_true_counters"] == 1:
            by_font_tc1[r["font"]] += 1

    print("\n" + "=" * 70)
    print("Per-font: number of n_true_counters=1 glyphs (Config B/C aggregation pool)")
    print("=" * 70)
    counts = sorted(by_font_tc1.values())
    if counts:
        print(f"  min={min(counts)}, max={max(counts)}, "
              f"median={counts[len(counts)//2]}, mean={sum(counts)/len(counts):.1f}")
    all_fonts = set(r["font"] for r in rows)
    no_counters = all_fonts - set(by_font_tc1.keys())
    print(f"  Fonts with ZERO n_tc=1 glyphs (Config B/C cannot use): {len(no_counters)}")
    if no_counters:
        print(f"    First 10: {sorted(no_counters)[:10]}")

    # Fonts with high n_components (the pattern/stippled tail)
    by_font_max_components = defaultdict(int)
    for r in rows:
        f = r["font"]
        by_font_max_components[f] = max(by_font_max_components[f], r["n_components"])
    top_pattern = sorted(by_font_max_components.items(), key=lambda x: -x[1])[:15]
    print("\n" + "=" * 70)
    print("Top fonts by max n_components (pattern/stippled fonts)")
    print("=" * 70)
    for name, mx in top_pattern:
        print(f"  {name:>30s}: max n_components = {mx}")

    print(f"\nWrote {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()