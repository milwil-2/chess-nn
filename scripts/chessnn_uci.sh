#!/bin/bash
# UCI entry point for lichess-bot (and any other UCI harness).
#
# Wraps `python models/v3_vast/run.py engine` with the right interpreter and
# CWD so the engine can find its config, checkpoints, syzygy tables, and
# transposition cache via relative paths.
#
# Override knobs (env vars):
#   CHESSNN_CHECKPOINT    relative-to-v3_vast checkpoint path (default: checkpoints/best_model.pt)
#   CHESSNN_SIMS          MCTS sims per move (default: 100)
#   CHESSNN_PYTHON        interpreter (default: the v1_history8 venv that has torch)

set -euo pipefail

REPO_ROOT="/Users/milan/Desktop/projects/chess-nn"
PY="${CHESSNN_PYTHON:-$REPO_ROOT/models/v1_history8/.venv/bin/python}"
CKPT="${CHESSNN_CHECKPOINT:-checkpoints/best_model.pt}"
SIMS="${CHESSNN_SIMS:-100}"

cd "$REPO_ROOT/models/v3_vast"
exec "$PY" run.py engine --checkpoint "$CKPT" --sims "$SIMS"
