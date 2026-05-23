#!/usr/bin/env bash
# One-time GitHub repo setup. Run from inside the `code/` folder.
#
# Pre-reqs:
#   - `gh` CLI installed and authenticated as adhvaryu-athena (already done per your message)
#   - `git` installed
#
# What this does:
#   1. Initializes a git repo here
#   2. Creates a public GitHub repo at adhvaryu-athena/calligraphy-efd
#   3. Pushes the initial commit (everything except data/google-fonts/, per .gitignore)

set -euo pipefail

REPO_NAME="calligraphy-efd"
REPO_OWNER="adhvaryu-athena"

echo "==========================================="
echo "Initializing repo: $REPO_OWNER/$REPO_NAME"
echo "==========================================="

# Sanity: are we in the code/ folder?
if [ ! -f "fonts.csv" ] || [ ! -d "src" ]; then
  echo "ERROR: run this script from inside the code/ folder."
  exit 1
fi

# Sanity: gh CLI ready?
if ! command -v gh &> /dev/null; then
  echo "ERROR: gh CLI not installed. brew install gh"
  exit 1
fi

if ! gh auth status &> /dev/null; then
  echo "ERROR: gh CLI not authenticated. gh auth login"
  exit 1
fi

# Init git if needed
if [ ! -d ".git" ]; then
  git init
  git branch -M main
fi

# Stage everything (respecting .gitignore)
git add -A
git status --short

# Initial commit
git commit -m "Initial commit: EFD typeface classification pipeline

- Five-stage pipeline (corpus, outlines, EFD, classification, visualize)
- 20-font calibration run with reference outputs
- Random forest with grid search and permutation importance
- Two CV protocols (stratified per-glyph + GroupKFold per-font)
- Bootstrap 95% CI on macro-F1
- Reproducible from a single ./run_full.sh"

# Create remote
if gh repo view "$REPO_OWNER/$REPO_NAME" &> /dev/null; then
  echo "Repo $REPO_OWNER/$REPO_NAME already exists; setting as remote..."
  git remote add origin "https://github.com/$REPO_OWNER/$REPO_NAME.git" 2>/dev/null || true
else
  gh repo create "$REPO_OWNER/$REPO_NAME" \
    --public \
    --source=. \
    --description="Classifying typeface styles with Elliptic Fourier Descriptors. Classical shape descriptors over Google Fonts; no GPU required." \
    --push
  echo "Created and pushed: https://github.com/$REPO_OWNER/$REPO_NAME"
  exit 0
fi

# If we got here, repo already existed — just push.
git push -u origin main

echo
echo "Done. Repo: https://github.com/$REPO_OWNER/$REPO_NAME"
