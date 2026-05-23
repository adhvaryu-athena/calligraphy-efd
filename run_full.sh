#!/usr/bin/env bash
# Full-corpus run script for the rig.
#
# Usage:
#   ./run_full.sh                  # default: 100 fonts/class, seed 42
#   ./run_full.sh 30               # PRD's 30 fonts/class
#   ./run_full.sh 50 7             # 50 fonts/class, seed 7
#
# Pre-requisites:
#   1. Python 3.10+
#   2. pip install -r requirements.txt
#   3. git clone --depth 1 https://github.com/google/fonts.git data/google-fonts
#
# Produces in outputs/:
#   fonts.csv, outlines.pkl, efd_features.npz, classification_results.json
#   figures/fig1..fig4, diag_confusion_matrix, fig_permutation_importance

set -euo pipefail

PER_CLASS="${1:-100}"
SEED="${2:-42}"

echo "==========================================="
echo "Calligraphy & Math — Full pipeline run"
echo "  per_class=$PER_CLASS  seed=$SEED"
echo "==========================================="

if [ ! -d "data/google-fonts/ofl" ]; then
  echo "ERROR: data/google-fonts/ not found."
  echo "  Run: git clone --depth 1 https://github.com/google/fonts.git data/google-fonts"
  exit 1
fi

mkdir -p outputs/figures

echo
echo "--- Stage 0: build fonts.csv from METADATA.pb ---"
python -m src.build_corpus \
    --fonts-root data/google-fonts \
    --per-class "$PER_CLASS" \
    --seed "$SEED" \
    --out fonts.csv

echo
echo "--- Stages 1-5: full pipeline ---"
python - << 'PYEOF'
import sys, pickle, json, os
sys.path.insert(0, '.')
from src import corpus, outlines, efd, classify, visualize

# Stage 1
df = corpus.run('fonts.csv', fonts_root='.')
df.to_csv('outputs/fonts.csv', index=False)

# Stage 2
outlines_dict, failures = outlines.run(df, output_path='outputs/outlines.pkl')

# Stage 3
label_map = dict(zip(df.font_name, df.style_class))
feats = efd.run(outlines_dict, label_map, output_path='outputs/efd_features.npz')

# Stage 4 — grid search RF + permutation importance enabled
results = classify.run(feats,
                       output_path='outputs/classification_results.json',
                       grid_search_rf=True,
                       compute_permutation=True)

# Stage 5 — figures
paths = visualize.run(feats, outlines_dict, label_map, results,
                       output_dir='outputs/figures')
for k, p in paths.items():
    print(f'  {k}: {p}')
PYEOF

echo
echo "Done. Outputs in outputs/"
ls outputs/
ls outputs/figures/
