"""Stage 1 — Font corpus construction.

Reads fonts.csv (the labeled corpus), resolves font file paths against a
configurable fonts root, and validates that every font listed actually exists
on disk and is readable by fontTools.

Output of run(): a pandas DataFrame with one row per font, augmented with an
`exists` boolean column. Any False rows are printed as warnings so the student
can correct the path or substitute a different font.
"""

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from fontTools.ttLib import TTFont, TTLibError


VALID_CLASSES = {"serif", "sans-serif", "calligraphic", "display"}


def load_corpus(csv_path: str | Path) -> pd.DataFrame:
    """Read fonts.csv. Validates required columns and class labels."""
    df = pd.read_csv(csv_path)
    required = {"font_name", "style_class", "google_category", "filepath"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fonts.csv is missing required columns: {missing}")
    bad_classes = set(df["style_class"]) - VALID_CLASSES
    if bad_classes:
        raise ValueError(
            f"fonts.csv contains invalid style_class values: {bad_classes}. "
            f"Allowed: {VALID_CLASSES}"
        )
    return df


def resolve_paths(df: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    """Resolve every filepath against the given root directory.

    The CSV stores paths relative to the project root (e.g. 'data/google-fonts/ofl/lora/Lora[wght].ttf').
    This function joins them with `root` to produce absolute paths and
    checks file existence.
    """
    root = Path(root)
    out = df.copy()
    out["abs_path"] = out["filepath"].apply(lambda p: str(root / p))
    out["exists"] = out["abs_path"].apply(lambda p: Path(p).is_file())
    return out


def validate_readable(df: pd.DataFrame) -> pd.DataFrame:
    """Try to open each font with fontTools and record whether it succeeds."""
    out = df.copy()
    readable = []
    for _, row in out.iterrows():
        if not row.get("exists", False):
            readable.append(False)
            continue
        try:
            TTFont(row["abs_path"], lazy=True)
            readable.append(True)
        except (TTLibError, Exception):
            readable.append(False)
    out["readable"] = readable
    return out


def summarize(df: pd.DataFrame) -> str:
    """Return a multi-line string summary of the corpus."""
    n_total = len(df)
    n_per_class = df["style_class"].value_counts().to_dict()
    n_exists = int(df.get("exists", pd.Series([True] * n_total)).sum())
    n_readable = int(df.get("readable", pd.Series([True] * n_total)).sum())
    lines = [
        f"Corpus: {n_total} fonts",
        f"  exists on disk: {n_exists}/{n_total}",
        f"  readable by fontTools: {n_readable}/{n_total}",
        "  per-class counts:",
    ]
    for cls in sorted(n_per_class):
        lines.append(f"    {cls}: {n_per_class[cls]}")
    if "exists" in df.columns:
        missing = df[~df["exists"]]
        if len(missing) > 0:
            lines.append("  MISSING FILES:")
            for _, r in missing.iterrows():
                lines.append(f"    - {r['font_name']}: {r['filepath']}")
    return "\n".join(lines)


def run(csv_path: str | Path, fonts_root: str | Path) -> pd.DataFrame:
    """Stage 1 entry point. Returns validated DataFrame; prints summary."""
    df = load_corpus(csv_path)
    df = resolve_paths(df, fonts_root)
    df = validate_readable(df)
    print(summarize(df))
    return df
