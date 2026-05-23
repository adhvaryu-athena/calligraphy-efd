"""Diagnostic 2 — font counts per script in the cloned google/fonts repo.

Walks data/google-fonts/{ofl,apache,ufl}, reads each METADATA.pb, parses
the `subsets:` field, and tallies fonts per script. Used to lock the
Phase 4 (non-Latin) scope.

Decision rule:
    >= 40 fonts in a non-Latin script -> include in Phase 4
    20 <= count < 40                  -> include as supplementary / smaller-scale
    < 20                              -> drop for now

Output:
    outputs/diagnostic_script_counts.csv  — one row per font
    Printed summary: counts per script, breakdown per category for non-Latin.

Run from repo root, inside the conda env:
    python scripts/diagnose_script_counts.py
    python scripts/diagnose_script_counts.py /custom/path/to/google-fonts
"""
from __future__ import annotations
import sys
import csv
import re
from collections import Counter
from pathlib import Path

DEFAULT_FONTS_ROOT = Path("data/google-fonts")

SCRIPTS_OF_INTEREST = {
    "devanagari", "gujarati", "tamil", "bengali", "telugu",
    "kannada", "malayalam", "oriya", "gurmukhi",
    "arabic", "hebrew", "thai", "lao", "khmer", "myanmar", "tibetan",
    "chinese-simplified", "chinese-traditional", "japanese", "korean",
}


def parse_metadata(metadata_path: Path):
    """Return (category, [subsets]) or (None, []) if unreadable."""
    try:
        text = metadata_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, []
    m_cat = re.search(r'^\s*category:\s*"([^"]+)"', text, re.MULTILINE)
    subs = re.findall(r'^\s*subsets:\s*"([^"]+)"', text, re.MULTILINE)
    return (m_cat.group(1) if m_cat else None), subs


def main():
    fonts_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FONTS_ROOT
    if not fonts_root.exists():
        print(f"Not found: {fonts_root}")
        print("Usage: python scripts/diagnose_script_counts.py [path/to/google-fonts]")
        sys.exit(1)

    fonts = []  # (font_dir_name, category, [subsets])
    for license_dir in ("ofl", "apache", "ufl"):
        base = fonts_root / license_dir
        if not base.is_dir():
            continue
        for font_dir in sorted(base.iterdir()):
            if not font_dir.is_dir():
                continue
            meta = font_dir / "METADATA.pb"
            if not meta.is_file():
                continue
            cat, subs = parse_metadata(meta)
            fonts.append((font_dir.name, cat, subs))

    print(f"\nScanned {len(fonts)} fonts in {fonts_root}")

    # Tally fonts per subset
    subset_total = Counter()
    for _, _, subs in fonts:
        for s in subs:
            subset_total[s] += 1

    print("\n" + "=" * 70)
    print(f"Top subsets by font count (top 25)")
    print("=" * 70)
    for s, c in subset_total.most_common(25):
        marker = " *" if s in SCRIPTS_OF_INTEREST else ""
        print(f"  {s:>30s}: {c:>5d}{marker}")
    print("  (* = non-Latin script of interest for Phase 4)")

    # Detailed breakdown for scripts of interest
    print("\n" + "=" * 70)
    print(f"Non-Latin scripts of interest — per-category breakdown")
    print("=" * 70)
    for script in sorted(SCRIPTS_OF_INTEREST):
        n = subset_total.get(script, 0)
        if n == 0:
            continue
        by_cat = Counter()
        for _, cat, subs in fonts:
            if script in subs and cat:
                by_cat[cat] += 1
        total = sum(by_cat.values())
        verdict = ("INCLUDE" if total >= 40
                   else "SUPPLEMENTARY" if total >= 20
                   else "DROP")
        print(f"\n  {script}  ->  {total} fonts  [{verdict}]")
        for cat in sorted(by_cat):
            print(f"    {cat:>15s}: {by_cat[cat]}")

    # Save full CSV
    Path("outputs").mkdir(exist_ok=True)
    out_path = "outputs/diagnostic_script_counts.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["font_dir", "category", "subsets"])
        for name, cat, subs in fonts:
            w.writerow([name, cat or "", "|".join(subs)])
    print(f"\nWrote {out_path}  ({len(fonts)} rows)")


if __name__ == "__main__":
    main()
