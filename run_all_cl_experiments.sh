#!/usr/bin/env bash
# =============================================================================
# Run ALL Continual Learning Experiments Sequentially
# =============================================================================
# Each experiment runs to completion before the next one starts.
# On failure the script logs the error and continues with the next experiment.
#
# Usage:
#   ./run_all_cl_experiments.sh /path/to/clips
#   ./run_all_cl_experiments.sh /path/to/clips --dry-run   # print commands only
#
# Extra Hydra overrides can be appended:
#   ./run_all_cl_experiments.sh /path/to/clips seed=123 trainer.precision=16-mixed
# =============================================================================
set -euo pipefail

# ── Argument handling ────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <data_dir> [--dry-run] [extra hydra overrides...]"
    echo ""
    echo "Examples:"
    echo "  $0 /data/precomputed_clips"
    echo "  $0 /data/precomputed_clips --dry-run"
    echo "  $0 /data/precomputed_clips seed=123"
    exit 1
fi

DATA_DIR="$1"; shift

DRY_RUN=false
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    else
        EXTRA_ARGS+=("$arg")
    fi
done

# ── Experiments to run (in order) ────────────────────────────────────────────
EXPERIMENTS=(
    "cl_upper_bound_cross_validation"
    "cl_upper_bound"
    "cl_lower_bound"
    "cl_ac_vit"
    "cl_ac_hope"


)

# ── Logging helpers ──────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="outputs/run_all_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
fail() { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }

# ── Summary tracking ────────────────────────────────────────────────────────
declare -A RESULTS
TOTAL=${#EXPERIMENTS[@]}
PASSED=0
FAILED=0
FAILED_LIST=()

# ── Main loop ────────────────────────────────────────────────────────────────
log "Starting ${TOTAL} CL experiments sequentially"
log "Data directory: ${DATA_DIR}"
log "Extra overrides: ${EXTRA_ARGS[*]:-<none>}"
log "Logs directory:  ${LOG_DIR}"
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    NUM=$((i + 1))
    EXP_LOG="${LOG_DIR}/${EXP}.log"

    CMD="uv run src/cl_train.py experiment=${EXP} paths.data_dir=${DATA_DIR} ${EXTRA_ARGS[*]:-}"

    log "[${NUM}/${TOTAL}] ${EXP}"
    log "  Command: ${CMD}"
    log "  Log:     ${EXP_LOG}"

    if $DRY_RUN; then
        warn "  (dry-run — skipped)"
        RESULTS[$EXP]="skipped"
        echo ""
        continue
    fi

    START_SEC=$SECONDS

    # Run the experiment; tee output to log file while still showing it
    if $CMD 2>&1 | tee "$EXP_LOG"; then
        ELAPSED=$(( SECONDS - START_SEC ))
        ok "${EXP} completed in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
        RESULTS[$EXP]="success ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
        PASSED=$((PASSED + 1))
    else
        ELAPSED=$(( SECONDS - START_SEC ))
        fail "${EXP} FAILED after $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s  (see ${EXP_LOG})"
        RESULTS[$EXP]="FAILED"
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$EXP")
    fi
    echo ""
done

# ── Final summary ────────────────────────────────────────────────────────────
echo ""
log "═══════════════════════════════════════════════════════════"
log "  SUMMARY  (${PASSED} passed / ${FAILED} failed / ${TOTAL} total)"
log "═══════════════════════════════════════════════════════════"
for EXP in "${EXPERIMENTS[@]}"; do
    STATUS="${RESULTS[$EXP]:-unknown}"
    if [[ "$STATUS" == FAILED ]]; then
        fail "  ${EXP}: ${STATUS}"
    elif [[ "$STATUS" == skipped ]]; then
        warn "  ${EXP}: ${STATUS}"
    else
        ok "  ${EXP}: ${STATUS}"
    fi
done
echo ""

# Save summary to file
{
    echo "CL Experiment Run — ${TIMESTAMP}"
    echo "Data: ${DATA_DIR}"
    echo "Overrides: ${EXTRA_ARGS[*]:-<none>}"
    echo ""
    for EXP in "${EXPERIMENTS[@]}"; do
        echo "${EXP}: ${RESULTS[$EXP]:-unknown}"
    done
} > "${LOG_DIR}/summary.txt"

log "Summary saved to ${LOG_DIR}/summary.txt"

# Exit with failure if any experiment failed
if [[ $FAILED -gt 0 ]]; then
    fail "Failed experiments: ${FAILED_LIST[*]}"
    exit 1
fi
