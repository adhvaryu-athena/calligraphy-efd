"""Phase 1d — Discovery: cross-category neighbor analysis and within-class distances."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-category neighbor analysis
# ---------------------------------------------------------------------------

def find_google_disagreed_pairs(
    neighbors_csv_path: str | Path,
    k: int = 5,
) -> Tuple[Dict[str, float], float, List[Dict]]:
    """For each font, compute how many of its top-k neighbors have a different label.

    Parameters
    ----------
    neighbors_csv_path : path to a neighbors_{variant}_{metric}.csv file
    k                  : number of top neighbors to examine

    Returns
    -------
    per_font_cross_rate : dict mapping font_name → fraction of top-k neighbors with different label
    aggregate_rate      : overall mean cross-category rate across all fonts
    pairs_list          : list of dicts describing cross-category neighbor pairs, each with:
                          font_name, font_label, neighbor_name, neighbor_label, rank
    """
    neighbors_csv_path = Path(neighbors_csv_path)
    if not neighbors_csv_path.exists():
        _logger.warning("neighbors CSV not found: %s", neighbors_csv_path)
        return {}, float("nan"), []

    df = pd.read_csv(neighbors_csv_path)

    # Determine available rank columns up to k
    rank_cols = [f"rank_{r}" for r in range(1, k + 1) if f"rank_{r}" in df.columns]
    if not rank_cols:
        _logger.warning("No rank columns found in %s", neighbors_csv_path)
        return {}, float("nan"), []

    if "label" not in df.columns:
        _logger.warning("No 'label' column in %s", neighbors_csv_path)
        return {}, float("nan"), []

    # Build label lookup
    label_lookup: Dict[str, str] = dict(zip(df["font_name"].astype(str), df["label"].astype(str)))

    per_font_cross_rate: Dict[str, float] = {}
    all_pairs: List[Dict] = []

    for _, row in df.iterrows():
        font_name = str(row["font_name"])
        font_label = str(row["label"])
        cross_count = 0
        for rank_idx, col in enumerate(rank_cols, start=1):
            neighbor = str(row[col]) if pd.notna(row[col]) else ""
            if not neighbor:
                continue
            neighbor_label = label_lookup.get(neighbor, "")
            if neighbor_label and neighbor_label != font_label:
                cross_count += 1
                all_pairs.append({
                    "font_name": font_name,
                    "font_label": font_label,
                    "neighbor_name": neighbor,
                    "neighbor_label": neighbor_label,
                    "rank": rank_idx,
                })
        per_font_cross_rate[font_name] = cross_count / len(rank_cols)

    if per_font_cross_rate:
        aggregate_rate = float(np.mean(list(per_font_cross_rate.values())))
    else:
        aggregate_rate = float("nan")

    # --- Save outputs ---
    out_dir = Path("outputs") / "discovery"
    out_dir.mkdir(parents=True, exist_ok=True)

    # All cross-category pairs
    pairs_df = pd.DataFrame(all_pairs)
    cross_pairs_path = out_dir / "cross_category_pairs.csv"
    pairs_df.to_csv(cross_pairs_path, index=False)
    _logger.info("Saved cross-category pairs → %s", cross_pairs_path)

    # Aggregate stats
    stats = {
        "aggregate_cross_category_rate": aggregate_rate,
        "n_fonts": len(per_font_cross_rate),
        "n_cross_pairs": len(all_pairs),
        "k": k,
        "source_csv": str(neighbors_csv_path),
    }
    stats_path = out_dir / "aggregate_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    _logger.info("Saved aggregate stats → %s", stats_path)

    # Illustrative pairs: highest-confidence (rank=1, different label), one per label-pair combo
    rank1_pairs = [p for p in all_pairs if p["rank"] == 1]
    seen_label_pairs = set()
    illustrative = []
    for pair in rank1_pairs:
        label_pair = (pair["font_label"], pair["neighbor_label"])
        if label_pair not in seen_label_pairs:
            seen_label_pairs.add(label_pair)
            illustrative.append(pair)

    illus_df = pd.DataFrame(illustrative)
    illus_path = out_dir / "illustrative_pairs.csv"
    illus_df.to_csv(illus_path, index=False)
    _logger.info("Saved illustrative pairs → %s", illus_path)

    return per_font_cross_rate, aggregate_rate, all_pairs


# ---------------------------------------------------------------------------
# Within-class distance analysis
# ---------------------------------------------------------------------------

def compute_within_class_distances(
    pairwise_npz_path: str | Path,
    labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute mean pairwise distance within each class.

    Parameters
    ----------
    pairwise_npz_path : path to a pairwise_{variant}_{metric}.npz file
    labels            : optional list of labels; if None, loads from the npz file

    Returns
    -------
    dict mapping label → mean within-class pairwise distance
    """
    pairwise_npz_path = Path(pairwise_npz_path)
    if not pairwise_npz_path.exists():
        _logger.warning("Pairwise NPZ not found: %s", pairwise_npz_path)
        return {}

    try:
        data = np.load(pairwise_npz_path, allow_pickle=True)
    except Exception as exc:
        _logger.warning("Failed to load %s: %s", pairwise_npz_path, exc)
        return {}

    D = data["D"]
    if labels is None:
        if "labels" in data:
            labels = data["labels"].tolist()
        else:
            _logger.warning("No labels provided or found in %s", pairwise_npz_path)
            return {}

    labels_arr = np.array(labels)
    unique_labels = np.unique(labels_arr)
    within_class_dist: Dict[str, float] = {}

    for label in unique_labels:
        idx = np.where(labels_arr == label)[0]
        if len(idx) < 2:
            within_class_dist[str(label)] = float("nan")
            continue
        # Extract within-class submatrix (upper triangle excluding diagonal)
        sub = D[np.ix_(idx, idx)]
        # Mean of upper-triangle (excluding diagonal)
        upper_tri = sub[np.triu_indices(len(idx), k=1)]
        within_class_dist[str(label)] = float(np.mean(upper_tri))

    _logger.info(
        "Within-class distances for %d classes from %s",
        len(within_class_dist), pairwise_npz_path.name,
    )
    return within_class_dist


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Phase 1d discovery analysis")
    parser.add_argument(
        "--neighbors-csv",
        default=None,
        help="Path to neighbors CSV (default: outputs/similarity/neighbors_A_mean_cosine.csv)",
    )
    parser.add_argument(
        "--pairwise-npz",
        default=None,
        help="Path to pairwise NPZ (default: outputs/similarity/pairwise_A_mean_cosine.npz)",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for cross-category analysis")
    args = parser.parse_args()

    sim_dir = Path("outputs") / "similarity"

    neighbors_csv = args.neighbors_csv
    if neighbors_csv is None:
        candidates = sorted(sim_dir.glob("neighbors_A_mean_*.csv")) if sim_dir.exists() else []
        if candidates:
            neighbors_csv = str(candidates[0])
        else:
            # Fall back to any neighbors csv
            all_csvs = sorted(sim_dir.glob("neighbors_*.csv")) if sim_dir.exists() else []
            if all_csvs:
                neighbors_csv = str(all_csvs[0])
            else:
                _logger.error("No neighbors CSV found in %s. Run similarity.py first.", sim_dir)
                raise SystemExit(1)

    _logger.info("Using neighbors CSV: %s", neighbors_csv)
    per_font, agg, pairs = find_google_disagreed_pairs(neighbors_csv, k=args.k)
    _logger.info("Aggregate cross-category rate: %.3f across %d fonts", agg, len(per_font))

    # Within-class distances
    pairwise_npz = args.pairwise_npz
    if pairwise_npz is None:
        # Derive from neighbors CSV path
        csv_stem = Path(neighbors_csv).stem  # neighbors_{variant}_{metric}
        metric_and_variant = csv_stem[len("neighbors_"):]  # e.g. "A_mean_cosine"
        pairwise_candidate = sim_dir / f"pairwise_{metric_and_variant}.npz"
        if pairwise_candidate.exists():
            pairwise_npz = str(pairwise_candidate)
        else:
            all_npzs = sorted(sim_dir.glob("pairwise_*.npz")) if sim_dir.exists() else []
            if all_npzs:
                pairwise_npz = str(all_npzs[0])

    if pairwise_npz:
        _logger.info("Using pairwise NPZ: %s", pairwise_npz)
        within_dists = compute_within_class_distances(pairwise_npz)
        _logger.info("Within-class distances: %s", within_dists)

        out_dir = Path("outputs") / "discovery"
        out_dir.mkdir(parents=True, exist_ok=True)
        within_path = out_dir / "within_class_distances.json"
        with open(within_path, "w") as f:
            json.dump({
                "source_npz": str(pairwise_npz),
                "within_class_distances": within_dists,
            }, f, indent=2)
        _logger.info("Saved within-class distances → %s", within_path)
    else:
        _logger.warning("No pairwise NPZ found — skipping within-class distance analysis")
