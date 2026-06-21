#!/usr/bin/env bash
# Run the standalone TorchSISSO baseline in the background with nohup.
#
# Usage:
#   bash scripts/run_torchsisso_nohup.sh
#
# Optional overrides:
#   CONDA_ENV=organoboronates-torchsisso bash scripts/run_torchsisso_nohup.sh
#   OUTPUT_DIR=models/TorchSISSO_extended bash scripts/run_torchsisso_nohup.sh
#   USE_GPU=1 bash scripts/run_torchsisso_nohup.sh
#
# Monitor:
#   tail -f logs/torchsisso_<timestamp>.log
#
# Stop:
#   kill $(cat logs/torchsisso_<timestamp>.pid)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-organoboronates-torchsisso}"
DATA="${DATA:-example/B_dataset.csv}"
TARGET="${TARGET:-activation_energy}"
OUTPUT_DIR="${OUTPUT_DIR:-models/TorchSISSO}"
LOG_DIR="${LOG_DIR:-logs}"
USE_GPU="${USE_GPU:-0}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/torchsisso_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/torchsisso_${TIMESTAMP}.pid"

# Activate conda when available. This works in non-interactive nohup shells.
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    echo "Warning: conda was not found. Using current Python environment." >&2
fi

CMD=(
    python -u example/train_torchsisso.py
    --data "$DATA"
    --target "$TARGET"
    --output_dir "$OUTPUT_DIR"
    --n_expansion 1 2
    --n_term 1 2
    --k 20 50
    --operators + - "*" / "pow(2)" ln
    --initial_screening spearman 0.95
)

if [[ "$USE_GPU" == "1" || "$USE_GPU" == "true" || "$USE_GPU" == "True" ]]; then
    CMD+=(--use_gpu)
fi

{
    echo "Repository: $REPO_ROOT"
    echo "Conda env:  $CONDA_ENV"
    echo "Data:       $DATA"
    echo "Target:     $TARGET"
    echo "Output dir: $OUTPUT_DIR"
    echo "Log file:   $LOG_FILE"
    echo "Started at: $(date)"
    echo
    echo "Command:"
    printf ' %q' "${CMD[@]}"
    echo
    echo
} > "$LOG_FILE"

nohup "${CMD[@]}" >> "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "TorchSISSO job started."
echo "PID:      $PID"
echo "PID file: $PID_FILE"
echo "Log file: $LOG_FILE"
echo
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo
echo "Stop with:"
echo "  kill \$(cat $PID_FILE)"
