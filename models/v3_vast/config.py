"""
Configuration for Vast.ai CUDA training — v3 (pgn-extract C filter + parallel encode pool).

Differences from v2:
  - N_ENCODE_WORKERS: dedicated encode worker pool separate from the 12 filter workers,
    saturating remaining CPU cores for tensor construction
  - Data pipeline: curl | zstdcat | pgn-extract → game_queue → encode pool → .npz
"""

import os
import subprocess
import torch


def _usable_cpu_count() -> int:
    """Return actual usable cores, respecting cgroup limits.

    os.cpu_count() reads /proc/cpuinfo and ignores cgroup CPU quotas, so it
    returns the full host count (e.g. 112) even when a container is allocated
    only 14 cores. `nproc` honours cgroup v1/v2 cpu.cfs_quota_us and is the
    correct tool here.
    """
    try:
        return int(subprocess.check_output(["nproc"], text=True).strip())
    except Exception:
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


_N_FILTER_WORKERS = 12   # one per Lichess month — keep in sync with LICHESS_MONTHS

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Architecture ---
NUM_RESIDUAL_BLOCKS = 10
NUM_FILTERS = 128
INPUT_PLANES = 105            # 8 history frames × 12 piece planes + 9 meta planes
POLICY_OUTPUT_SIZE = 4672

# --- Training ---
BATCH_SIZE = 2048             # 4090 has 24 GB — ~12 GB at this batch, tensor cores fully utilized
LEARNING_RATE = 0.004         # Linear scaling: 4× batch vs 512-baseline → 4× LR
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
GRADIENT_CLIP = 1.0
VALUE_LOSS_WEIGHT = 1.0       # WDL cross-entropy is already on same scale as policy loss
DATA_WORKERS = 4              # conservative — GPU pipeline doesn't need many loaders
# Leave 2 cores for OS + main process on top of the 12 filter workers.
# _usable_cpu_count() uses nproc so it honours cgroup quotas (os.cpu_count() does not).
N_ENCODE_WORKERS = max(2, _usable_cpu_count() - _N_FILTER_WORKERS - 2)

# --- Data ---
MIN_RATING = 1800
MIN_MOVES = 10
TRAIN_SPLIT = 0.90
VAL_SPLIT = 0.05
TEST_SPLIT = 0.05

# --- Stockfish supervision ---
SF_LOSS_WEIGHT      = 0.3   # weight of auxiliary Stockfish policy loss
SF_ANNOTATE_FRACTION = 0.20  # fraction of positions annotated per chunk

# --- MCTS / Self-Play noise (shaped Dirichlet, KataGo) ---
# At the root, Dirichlet noise is restricted to children whose prior is at
# least mean(P) * DIRICHLET_SHAPE_FLOOR. Set to 0.0 to disable shaping (flat
# AlphaZero noise). 0.1 keeps noise on the top ~half of legal moves and stops
# self-play from exploring obvious junk like early king pushes.
DIRICHLET_SHAPE_FLOOR = 0.1

# --- Root policy softmax temperature (B1, KataGo) ---
# At the MCTS root, divide policy logits by this T before softmax. Higher T
# = flatter priors = MCTS willing to search more children. Reduces the
# "38% out-of-top-8 picks" symptom by giving the search a less peaked
# distribution to work from. 1.0 disables. KataGo uses 1.25 → 1.1; constant
# 1.25 is the right starting point for chess (~30 legal moves midgame).
ROOT_POLICY_TEMPERATURE = 1.25

# --- Reinforcement Learning ---
RL_GAMES_PER_ITER  = 25
RL_SIMULATIONS     = 100
RL_CHUNK_SIZE      = 5
RL_HISTORY_FILES   = 5
RL_EPOCHS          = 5
RL_LR              = 1e-4
RL_EVAL_GAMES      = 20
RL_WIN_THRESHOLD   = 0.55

# --- Paths ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

# --- Inference-only MCTS helpers (engine + viz; NEVER wired into self-play) ---
OPENING_BOOK_PATH = os.path.join(PROJECT_DIR, "data", "book.bin")
SYZYGY_PATH = os.path.join(PROJECT_DIR, "data", "syzygy")
MCTS_CACHE_PATH = os.path.join(PROJECT_DIR, "data", "mcts_cache.json")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
