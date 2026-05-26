#!/usr/bin/env bash
# measure_elo.sh — run the full local gauntlet (30 games × 5 SF levels) and
# print the resulting Elo number. Intended for automation (e.g. vast/pull.sh
# appends this after a new checkpoint lands).
#
# Usage: scripts/measure_elo.sh [extra args passed to run_gauntlet.py]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V3_DIR="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-/opt/homebrew/bin/python3.12}"

cd "$V3_DIR"

LOG="$V3_DIR/logs/measure_elo_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$V3_DIR/logs"

echo "[measure_elo] starting gauntlet, log -> $LOG"

"$PY" "$V3_DIR/run_gauntlet.py" --games-per-level 30 --tc 10+0.1 "$@" \
  2>&1 | tee "$LOG"

# Extract the final Elo line and surface it
if grep -q "Final aggregate Elo:" "$LOG"; then
  ELO_LINE="$(grep 'Final aggregate Elo:' "$LOG" | tail -1)"
  echo
  echo "[measure_elo] result: $ELO_LINE"
else
  echo "[measure_elo] WARNING: no 'Final aggregate Elo:' line in log" >&2
fi
