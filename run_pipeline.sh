#!/usr/bin/env bash
# Calligraphy EFD pipeline orchestration script.
#
# Usage:
#   ./run_pipeline.sh [--phase 0|1|2|3|4|6] [--config A|B|C] [--skip-existing]
#
# Options:
#   --phase N        Run only phase N (default: run all phases 0-4 in order)
#   --config A|B|C   Restrict to a single feature config (default: all)
#   --skip-existing  Skip phases whose primary output already exists
#
# Phase overview:
#   0  Extract EFD features (all configs) and build font-level matrices
#   1  Clustering, embeddings, pairwise similarity, discovery
#   2  Cluster stability analysis and ablation table
#   3  Cluster validation (vs Google Fonts categories)
#   4  Non-Latin script analysis (requires data/google-fonts/)
#   6  Web tool (Streamlit) — prints launch command only
#
# Pre-requisites:
#   pip install -r requirements.txt
#   (For phases 0+: fonts.csv must be present in outputs/ or project root)
#   (For phase 4: git clone --depth 1 https://github.com/google/fonts.git data/google-fonts)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

RUN_PHASE=""          # empty = run all
CONFIG=""             # empty = all configs
SKIP_EXISTING=0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)
            RUN_PHASE="${2:-}"
            shift 2
            ;;
        --config)
            CONFIG="${2:-}"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=1
            shift
            ;;
        -h|--help)
            head -30 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_phase() {
    local phase="$1"
    local description="$2"
    local script="$3"

    if [[ -n "$RUN_PHASE" && "$RUN_PHASE" != "$phase" ]]; then
        return 0
    fi

    log "==========================================="
    log "Phase $phase: $description"
    log "==========================================="

    eval "$script"
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR: Phase $phase failed with exit code $exit_code"
        exit $exit_code
    fi
    log "Phase $phase complete."
    echo
}

check_skip() {
    local output_file="$1"
    if [[ $SKIP_EXISTING -eq 1 && -f "$output_file" ]]; then
        log "SKIP: $output_file already exists (--skip-existing)"
        return 0  # signal "should skip"
    fi
    return 1  # signal "should run"
}

mkdir -p outputs/figures

# ---------------------------------------------------------------------------
# Phase 0: Feature extraction
# ---------------------------------------------------------------------------

PHASE0_PRIMARY="outputs/font_features_A_mean.npz"
_phase0_cmd="python scripts/run_phase0.py"
if [[ -n "$CONFIG" ]]; then
    log "Note: --config $CONFIG is informational for Phase 0; run_phase0.py extracts all configs."
fi

if ! check_skip "$PHASE0_PRIMARY"; then
    run_phase 0 "Extract EFD features and build font-level matrices" "$_phase0_cmd"
else
    [[ -z "$RUN_PHASE" || "$RUN_PHASE" == "0" ]] && log "Phase 0: skipped (output exists)"
fi

# ---------------------------------------------------------------------------
# Phase 1: Clustering, embeddings, similarity, discovery
# ---------------------------------------------------------------------------

PHASE1_PRIMARY="outputs/clustering_results.json"
_phase1_cmd="python scripts/run_phase1.py"

if ! check_skip "$PHASE1_PRIMARY"; then
    run_phase 1 "Clustering, embeddings, similarity, discovery" "$_phase1_cmd"
else
    [[ -z "$RUN_PHASE" || "$RUN_PHASE" == "1" ]] && log "Phase 1: skipped (output exists)"
fi

# ---------------------------------------------------------------------------
# Phase 3 (classification): existing run_full.sh logic + validation
# ---------------------------------------------------------------------------

PHASE3_PRIMARY="outputs/validation/cluster_validation_results.json"
_phase3_classify_check="outputs/classification_results.json"

run_phase 3 "Cluster validation" "python scripts/run_phase3.py"

# ---------------------------------------------------------------------------
# Phase 2: Stability and ablation
# ---------------------------------------------------------------------------

PHASE2_PRIMARY="outputs/stability/bootstrap_ari.json"
_phase2_cmd="python scripts/run_phase2.py"

if ! check_skip "$PHASE2_PRIMARY"; then
    run_phase 2 "Cluster stability analysis and ablation table" "$_phase2_cmd"
else
    [[ -z "$RUN_PHASE" || "$RUN_PHASE" == "2" ]] && log "Phase 2: skipped (output exists)"
fi

# ---------------------------------------------------------------------------
# Phase 4: Non-Latin (placeholder — requires google-fonts)
# ---------------------------------------------------------------------------

run_phase 4 "Non-Latin script analysis (placeholder)" "python scripts/run_phase4.py"

# ---------------------------------------------------------------------------
# Phase 6: Web tool
# ---------------------------------------------------------------------------

if [[ -z "$RUN_PHASE" || "$RUN_PHASE" == "6" ]]; then
    log "==========================================="
    log "Phase 6: Web tool"
    log "==========================================="
    echo ""
    echo "  To launch the Streamlit font explorer:"
    echo ""
    echo "      streamlit run src/web_tool.py"
    echo ""
    log "Phase 6: no computation needed (web tool is on-demand)."
    echo
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

if [[ -z "$RUN_PHASE" ]]; then
    log "==========================================="
    log "All phases complete."
    log "Outputs in: outputs/"
    log "==========================================="
    ls outputs/ 2>/dev/null || true
fi
