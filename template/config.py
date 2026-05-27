"""
Central configuration for the chess neural network.

All hyperparameters and paths live here so you can tune the model
from one place without hunting through multiple files.
"""

import os
import torch

# --- Device Selection ---
# PyTorch can run on CPU, NVIDIA GPU (CUDA), or Apple Silicon GPU (MPS).
# MPS = Metal Performance Shaders, Apple's GPU compute framework.
# We auto-detect the best available device.
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# --- Model Architecture ---
NUM_RESIDUAL_BLOCKS = 10      # Depth of the network (more = stronger but slower)
NUM_FILTERS = 128             # Width of each layer (more = more capacity)
INPUT_PLANES = 105            # 8 history frames × 12 piece planes + 9 meta planes
POLICY_OUTPUT_SIZE = 4672     # 73 move types × 64 source squares

# --- Training ---
BATCH_SIZE = 256              # Positions per gradient update (256 fits M3 8GB)
LEARNING_RATE = 0.001         # Initial learning rate for Adam optimizer
WEIGHT_DECAY = 1e-4           # L2 regularization to prevent overfitting
NUM_EPOCHS = 15               # Full passes through the training data
GRADIENT_CLIP = 1.0           # Max gradient norm (prevents exploding gradients)
VALUE_LOSS_WEIGHT = 0.5       # How much to weight value loss vs policy loss

# --- Data ---
MIN_RATING = 2000             # Only learn from games where both players are 2000+
MIN_MOVES = 10                # Skip very short games (likely abandoned)
TRAIN_SPLIT = 0.90
VAL_SPLIT = 0.05
TEST_SPLIT = 0.05

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
# distribution to work from. 1.0 disables. KataGo uses 1.25 → 1.1; the
# I8 sweep (#26) over T in {1.10, 1.25, 1.50, 2.00} at SF1600+SF1800
# showed T=1.50 wins by ~730 Elo over T=1.25 (which lost every game).
# Sweep override: set CHESSNN_ROOT_POLICY_TEMPERATURE=<float> in env to
# override at import time without editing this file.
ROOT_POLICY_TEMPERATURE = float(os.environ.get("CHESSNN_ROOT_POLICY_TEMPERATURE", 1.50))

# I2: ply-aware annealing of the root softmax T from start → end across
# the first `_ANNEAL_PLY` plies (KataGo decays T by ~0.15 across opening→
# endgame). Code path is shipped but disabled by default because the
# A/B match (annealed 1.50→1.35 vs constant 1.50, 20 games at SF1600+1800)
# showed annealed lost by ~104 Elo. Set END != START to re-enable.
ROOT_POLICY_TEMPERATURE_END = float(os.environ.get(
    "CHESSNN_ROOT_POLICY_TEMPERATURE_END", ROOT_POLICY_TEMPERATURE))
ROOT_POLICY_TEMPERATURE_ANNEAL_PLY = int(os.environ.get(
    "CHESSNN_ROOT_POLICY_TEMPERATURE_ANNEAL_PLY", 40))

# --- Reinforcement Learning ---
RL_GAMES_PER_ITER  = 25     # Self-play games generated each iteration
RL_SIMULATIONS     = 200    # MCTS simulations per move (more = stronger but slower)
RL_CHUNK_SIZE      = 5      # Flush self-play RAM to disk every N games (lower = less RAM)
RL_HISTORY_FILES   = 5      # How many past iteration files to train on (rolling window)
RL_EPOCHS          = 5      # Training epochs per iteration on new self-play data
RL_LR              = 1e-4   # Learning rate — smaller than supervised (fine-tuning)
RL_EVAL_GAMES      = 20     # Head-to-head games to compare new vs old model
RL_WIN_THRESHOLD   = 0.55   # New model must win >55% to be kept

# --- Paths ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

# --- Inference-only MCTS helpers (engine + viz; NEVER wired into self-play) ---
# Defined after PROJECT_DIR since they depend on it. Each path lives under
# the variant's data/ subdir; helpers gracefully no-op when the file/dir is
# absent so these constants are safe to ship without the data being present.
# Opening book: optional polyglot .bin file. Falls back to a hardcoded
# mini-book if the file is absent.
OPENING_BOOK_PATH = os.path.join(PROJECT_DIR, "data", "book.bin")

# Syzygy tablebases (3-4-5 piece, ~1 GB). Download from
# http://tablebase.sesse.net/syzygy/3-4-5/ and point this at the directory
# containing the .rtbw / .rtbz files. Empty/missing path → probe disabled.
SYZYGY_PATH = os.path.join(PROJECT_DIR, "data", "syzygy")

# Transposition cache: cross-game persisted visit counts at MCTS roots.
# Auto-loaded on engine/viz startup; saved on clean exit.
MCTS_CACHE_PATH = os.path.join(PROJECT_DIR, "data", "mcts_cache.json")
