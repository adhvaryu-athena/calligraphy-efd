# Calligraphy & Math — EFD Typeface Classification

End-to-end Python pipeline for the research project *Classifying Typeface Styles Using Elliptic Fourier Descriptors of Glyph Outlines*. Uses classical shape descriptors (no neural network, no GPU) to classify open-source fonts into serif, sans-serif, calligraphic, and display categories.

## Quick start

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/google/fonts.git data/google-fonts
./run_full.sh                  # 100 fonts/class, seed 42, ~5 min on a laptop
```

That generates `fonts.csv`, runs all five pipeline stages, and writes outputs to `outputs/` including five publication-ready PNG figures.

## What it does

Five-stage pipeline:

1. **Build corpus** — `src/build_corpus.py` reads `METADATA.pb` files from `data/google-fonts/`, samples N fonts per class with a fixed random seed, and writes `fonts.csv`.
2. **Validate** (`src/corpus.py`) — confirms every font is on disk and readable by fontTools.
3. **Extract outlines** (`src/outlines.py`) — for each lowercase a-z in each font: walk Bezier description, select outer contour by largest absolute signed area, resample to 200 points equally spaced along arc length.
4. **Compute EFD features** (`src/efd.py`) — `pyefd.elliptic_fourier_descriptors(normalize=True)` at six harmonic orders (5, 10, 15, 20, 30, 40). After normalization the first three coefficients are constants and dropped, giving 4N-3 features per glyph.
5. **Classify** (`src/classify.py`) — three classifiers (nearest centroid, random forest with GridSearchCV, PCA + logistic regression) × two CV protocols (per-glyph stratified, per-font GroupKFold) × bootstrap 95% CI on macro-F1. Random forest also gets permutation importance alongside MDI.
6. **Visualize** (`src/visualize.py`) — four publication figures plus confusion matrix.

## Project layout

```
.
├── README.md                       # This file
├── requirements.txt
├── fonts.csv                       # Generated; or curated by hand. Pipeline input.
├── labeling_protocol.md            # How fonts are assigned to classes
├── run_full.sh                     # One-command full-corpus run
├── src/
│   ├── build_corpus.py             # Stage 0: METADATA.pb -> fonts.csv
│   ├── corpus.py                   # Stage 1: validate fonts.csv
│   ├── outlines.py                 # Stage 2: glyph -> 200-point contour
│   ├── efd.py                      # Stage 3: contour -> EFD features
│   ├── classify.py                 # Stage 4: three classifiers + grid search + perm imp
│   └── visualize.py                # Stage 5: four PRD figures + confusion matrix
├── notebooks/
│   └── pipeline.ipynb              # Orchestrating Colab notebook
├── tests/                          # Smoke tests
├── data/google-fonts/              # Gitignored. User-cloned.
└── outputs/                        # Pipeline artifacts (initial commit contains the calibration run)
```

## Two cross-validation protocols, and why

The PRD specifies per-glyph stratified k-fold. We additionally report per-font GroupKFold, because 26 glyphs from the same font are *not* statistically independent (they share a designer and a style). GroupKFold ensures all glyphs from a font stay together in train OR test. The GroupKFold number is the headline; the stratified number is reported for comparison and to make the inflation explicit.

## Calibration vs full run

The `outputs/` directory in the initial commit contains a 20-font *calibration run* used to validate the pipeline. Headline calibration result: random forest macro-F1 = 0.40 [95% CI 0.36 – 0.44] under per-font GroupKFold, vs 0.25 chance baseline.

To re-run at the PRD's intended scale (or larger), see "Running the full corpus" below.

## Running the full corpus

Default `run_full.sh` does 100 fonts/class (400 total), seed 42:

```bash
./run_full.sh
```

PRD's exact 30/class (120 total):

```bash
./run_full.sh 30
```

Custom seed for ablation:

```bash
./run_full.sh 100 7
```

## Running on a remote rig

The pipeline is filesystem-portable. On the rig:

```bash
# 1. Clone the repo
git clone https://github.com/adhvaryu-athena/calligraphy-efd.git
cd calligraphy-efd

# 2. Set up Python
pip install -r requirements.txt

# 3. Get the font corpus (1.2 GB shallow clone)
git clone --depth 1 https://github.com/google/fonts.git data/google-fonts

# 4. Run
./run_full.sh 100
```

When done, `outputs/` contains all artifacts. Ship them back to your main workspace however you like (rsync, scp, Drive Desktop sync, or just `git diff outputs/`).

## Key design choices

- **Outer contour selection by signed area.** Multi-contour glyphs (a, b, d, e, g, o, p, q) use the sub-path with the largest absolute signed area. See `outlines.py: select_outer_contour`.
- **Arc-length resampling.** Bezier `t`-uniform sampling produces non-uniform arc-length spacing, which distorts low harmonics. We resample to 200 points equally spaced along arc length before EFD. See `outlines.py: resample_arc_length`.
- **Normalization.** `pyefd.elliptic_fourier_descriptors(normalize=True)` gives translation/rotation/scale/starting-point invariance. The first three coefficients become constants and are dropped.
- **Grid search.** Random forest tuned over `max_depth ∈ {5, 10, None}` × `min_samples_leaf ∈ {1, 2, 5}` via 3-fold inner CV.
- **Bootstrap 95% CI** on macro-F1 by resampling held-out predictions 1000×. Percentile method.
- **Permutation importance** alongside MDI to address MDI's high-cardinality bias (Strobl et al. 2007).

## Reproducibility

All random seeds default to 42. The pipeline is deterministic given the same `fonts.csv` and the same Google Fonts repo state. For a fully pinned run, also pin the Google Fonts commit SHA.

## Citation foundations

Kuhl & Giardina (1982) for the EFD math, Pedregosa et al. (2011) for scikit-learn, Breiman (2001) and Strobl et al. (2007) for random forests + importance, Iwata & Ukai (2002) and Neto et al. (2006) for the EFD→PCA→classifier workflow precedent in morphometrics. Full reference list in the project's literature review.
