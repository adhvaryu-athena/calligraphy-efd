"""Corpus builder — generate fonts.csv at arbitrary scale from google/fonts metadata.

Reads METADATA.pb files under data/google-fonts/{ofl,apache,ufl}/ and samples N
fonts per class with a fixed random seed. Output: fonts.csv compatible with
src/corpus.py downstream.

Class mapping (from Google Fonts' own category labels):
    SERIF        -> serif
    SANS_SERIF   -> sans-serif
    HANDWRITING  -> calligraphic
    DISPLAY      -> display
    MONOSPACE    -> excluded (not one of our four target classes)

The student's job is to audit the auto-generated labels against labeling_protocol.md
and override any they disagree with. Google's categories are a starting point.

Usage:
    python -m src.build_corpus --per-class 30 --seed 42 --out fonts.csv
    python -m src.build_corpus --per-class 100 --seed 42 --out fonts_full.csv
"""
from __future__ import annotations
import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Optional


CLASS_FROM_GOOGLE = {
    "SERIF": "serif",
    "SANS_SERIF": "sans-serif",
    "HANDWRITING": "calligraphic",
    "DISPLAY": "display",
}


def parse_metadata_pb(path: Path) -> Optional[Dict]:
    """Lightweight parser for the relevant METADATA.pb fields.

    We avoid the protobuf dependency by line-based regex for `name`, `category`,
    and `filename`. Picks the variable font (filename contains '[') when available,
    else the Regular static, else the first filename.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m_name = re.search(r'^\s*name:\s*"([^"]+)"', text, re.MULTILINE)
    m_cat = re.search(r'^\s*category:\s*"([^"]+)"', text, re.MULTILINE)
    if not (m_name and m_cat):
        return None
    filenames = re.findall(r'^\s*filename:\s*"([^"]+\.ttf)"', text, re.MULTILINE)
    if not filenames:
        return None
    variable = [f for f in filenames if "[" in f]
    regular = [f for f in filenames if "Regular" in f and "[" not in f]
    fname = variable[0] if variable else (regular[0] if regular else filenames[0])
    return {
        "name": m_name.group(1),
        "category": m_cat.group(1),
        "filename": fname,
    }


def discover_fonts(fonts_root: Path) -> List[Dict]:
    """Walk fonts_root/{ofl,apache,ufl} and return one dict per font."""
    rows = []
    for license_dir in ("ofl", "apache", "ufl"):
        base = fonts_root / license_dir
        if not base.is_dir():
            continue
        for font_dir in sorted(base.iterdir()):
            if not font_dir.is_dir():
                continue
            meta_path = font_dir / "METADATA.pb"
            if not meta_path.is_file():
                continue
            info = parse_metadata_pb(meta_path)
            if info is None:
                continue
            ttf_path = font_dir / info["filename"]
            if not ttf_path.is_file():
                continue
            # Build the path expected by fonts.csv: relative to project root.
            project_relative = (
                Path("data/google-fonts") / license_dir / font_dir.name / info["filename"]
            )
            rows.append({
                "font_name": info["name"].replace(" ", ""),  # "EB Garamond" -> "EBGaramond"
                "google_category": info["category"],
                "filepath": str(project_relative),
                "license": license_dir,
            })
    return rows


def sample_per_class(fonts: List[Dict], per_class: int, seed: int) -> List[Dict]:
    """Group by Google category, keep only our four classes, sample N per class.

    If a class has fewer than per_class fonts available, take them all and warn.
    """
    by_cat: Dict[str, List[Dict]] = {}
    for f in fonts:
        if f["google_category"] in CLASS_FROM_GOOGLE:
            by_cat.setdefault(f["google_category"], []).append(f)
    rng = random.Random(seed)
    selected: List[Dict] = []
    for cat in ("SERIF", "SANS_SERIF", "HANDWRITING", "DISPLAY"):
        pool = by_cat.get(cat, [])
        if len(pool) < per_class:
            print(f"  WARNING: {cat} has only {len(pool)} fonts available "
                  f"(wanted {per_class}); using all.")
            take = pool
        else:
            take = rng.sample(pool, per_class)
        for row in take:
            row["style_class"] = CLASS_FROM_GOOGLE[cat]
            row["notes"] = (f"Auto-labeled from Google category {cat}; "
                            f"verify per labeling_protocol.md.")
            selected.append(row)
    return selected


def write_csv(rows: List[Dict], out_path: Path) -> None:
    """Write fonts.csv with proper CSV quoting (filepaths may contain commas)."""
    fieldnames = ["font_name", "style_class", "google_category", "filepath", "notes"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fonts-root", default="data/google-fonts",
                    help="Path to cloned google/fonts repo (default: data/google-fonts)")
    ap.add_argument("--per-class", type=int, default=30,
                    help="Number of fonts per class (default: 30; PRD's target)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for sampling (default: 42)")
    ap.add_argument("--out", default="fonts.csv",
                    help="Output CSV path (default: fonts.csv)")
    args = ap.parse_args()

    fonts_root = Path(args.fonts_root).resolve()
    if not fonts_root.is_dir():
        raise SystemExit(
            f"Fonts root not found: {fonts_root}\n"
            f"Did you run: git clone --depth 1 https://github.com/google/fonts.git "
            f"{fonts_root}"
        )

    print(f"Scanning {fonts_root} for METADATA.pb files...")
    all_fonts = discover_fonts(fonts_root)
    print(f"  Found {len(all_fonts)} fonts total.")

    by_cat = {}
    for f in all_fonts:
        by_cat[f["google_category"]] = by_cat.get(f["google_category"], 0) + 1
    print("  Per-category counts in source:")
    for cat in sorted(by_cat):
        marker = " ->" if cat in CLASS_FROM_GOOGLE else "   "
        print(f"    {marker} {cat}: {by_cat[cat]}")

    print(f"\nSampling {args.per_class} per class with seed={args.seed}...")
    selected = sample_per_class(all_fonts, args.per_class, args.seed)
    print(f"  Selected {len(selected)} fonts total.")

    write_csv(selected, Path(args.out))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
