#!/usr/bin/env bash
# Pull checkpoints and logs from a Vast.ai instance back to local.
#
# Usage:
#   ./vast/pull.sh <ssh-host> [model-dir]
#
# Examples:
#   ./vast/pull.sh root@123.45.67.89
#   ./vast/pull.sh root@123.45.67.89:12345 models/v2_bigger

set -euo pipefail

HOST="${1:?Usage: $0 <ssh-host> [model-dir]}"
MODEL_DIR="${2:-models/v2_vast}"
REMOTE_DIR="/workspace/chess-nn"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$REPO_ROOT/$MODEL_DIR"

if [[ ! -d "$DST" ]]; then
  echo "Error: local model directory not found: $DST" >&2
  exit 1
fi

# Parse optional port
if [[ "$HOST" == *:* ]]; then
  SSH_HOST="${HOST%:*}"
  SSH_PORT="${HOST##*:}"
else
  SSH_HOST="$HOST"
  SSH_PORT="22"
fi

echo "==> Pulling checkpoints and logs from $SSH_HOST → $DST"

rsync -avz --progress \
  -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
  "$SSH_HOST:$REMOTE_DIR/checkpoints/" \
  "$DST/checkpoints/"

rsync -avz --progress \
  -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
  "$SSH_HOST:$REMOTE_DIR/logs/" \
  "$DST/logs/"

# Also pull training.log if it exists (from nohup run)
rsync -avz --progress \
  -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
  --ignore-missing-args \
  "$SSH_HOST:$REMOTE_DIR/training.log" \
  "$DST/logs/training_stdout.log" 2>/dev/null || true

echo ""
echo "==> Done. Checkpoints are in: $DST/checkpoints/"
echo "    Best model: $DST/checkpoints/best_model.pt"

# Auto-measure Elo for v3_vast checkpoints (#22 E4). Only runs if the
# measure_elo.sh script exists in the pulled model directory.
if [[ -x "$DST/scripts/measure_elo.sh" ]]; then
  echo ""
  echo "==> Auto-running Elo gauntlet (this may take 30-60 min)..."
  "$DST/scripts/measure_elo.sh" || echo "[warn] gauntlet exited non-zero"
fi
