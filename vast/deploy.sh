#!/usr/bin/env bash
# Deploy a model variant to a Vast.ai instance and start training.
#
# Usage:
#   ./vast/deploy.sh <ssh-host> [model-dir]
#
# Examples:
#   ./vast/deploy.sh root@123.45.67.89                      # deploys models/v2_vast (default)
#   ./vast/deploy.sh root@123.45.67.89:12345 models/v2_vast
#
# IMPORTANT for v2_vast (1M games): rent an instance with 200GB+ disk space.
# In Vast's search UI, filter by "disk space" before renting.
# The RTX 4090 template works best — 24 GB VRAM, Ada Lovelace (bf16 Tensor Cores).
#
# The SSH host is printed on your Vast.ai instance page (Connect tab).

set -euo pipefail

HOST="${1:?Usage: $0 <ssh-host> [model-dir]}"
MODEL_DIR="${2:-models/v3_vast}"
REMOTE_DIR="/workspace/chess-nn"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/$MODEL_DIR"

if [[ ! -d "$SRC" ]]; then
  echo "Error: model directory not found: $SRC" >&2
  exit 1
fi

# Parse optional port from host string (root@host:port → host -p port)
if [[ "$HOST" == *:* ]]; then
  SSH_HOST="${HOST%:*}"
  SSH_PORT="${HOST##*:}"
else
  SSH_HOST="$HOST"
  SSH_PORT="22"
fi

echo "==> Deploying $MODEL_DIR → $SSH_HOST:$REMOTE_DIR"

# Sync model directory (exclude local artifacts that shouldn't be on remote)
rsync -avz --progress \
  -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude "checkpoints/" \
  --exclude "logs/" \
  --exclude ".venv/" \
  --exclude "data/raw/" \
  --exclude "data/processed/" \
  "$SRC/" \
  "$SSH_HOST:$REMOTE_DIR/"

echo ""
echo "==> Installing dependencies on remote..."
ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_HOST" bash <<'REMOTE'
  cd /workspace/chess-nn
  # pgn-extract: C-based PGN filter (~300K games/min vs Python's ~3K/min)
  apt-get install -y --quiet pgn-extract
  # Ubuntu installs it to /usr/games, not on the default PATH — symlink it
  ln -sf /usr/games/pgn-extract /usr/local/bin/pgn-extract
  pgn-extract --help 2>&1 | head -1
  # Skip torch install if already present (saves ~5 min on 4090 template)
  python3 -c "import torch" 2>/dev/null || pip install --quiet torch
  pip install --quiet python-chess numpy tqdm requests zstandard
  # Ensure `python` resolves to python3 (some templates omit the alias)
  ln -sf "$(which python3)" /usr/local/bin/python 2>/dev/null || true
  mkdir -p checkpoints logs data/raw data/processed
  echo "Dependencies installed. $(python3 --version), torch $(python3 -c 'import torch; print(torch.__version__)')"
REMOTE

echo ""
echo "==> Ready. To start training (run on the instance):"
echo ""
echo "    # Interactive:"
echo "    ssh -p $SSH_PORT $SSH_HOST"
echo "    cd /workspace/chess-nn"
echo "    python3 data/download_data.py   # ~30 min — 1M games, pgn-extract C filter + parallel encode pool"
echo "    python3 run.py supervised       # ~20 hours on 4090"
echo ""
echo "    # Or detached (survives SSH disconnect):"
echo "    ssh -p $SSH_PORT $SSH_HOST 'cd /workspace/chess-nn && nohup bash -c \"set -o pipefail; python3 -u data/download_data.py 2>&1 | tee logs/download.log && python3 -u run.py supervised 2>&1 | tee logs/train.log\" > logs/pipeline.log 2>&1 &'"
echo "    # Monitor:      ssh -p $SSH_PORT $SSH_HOST 'tail -f /workspace/chess-nn/logs/download.log'"
echo "    # Fresh start:  ssh -p $SSH_PORT $SSH_HOST 'cd /workspace/chess-nn && python3 data/download_data.py --fresh'"
