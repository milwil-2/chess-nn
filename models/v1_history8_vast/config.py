"""
Configuration for Vast.ai CUDA training.

Differences from local M3 version:
  - CUDA device (no MPS)
  - Batch size 512 (2× local) — leverages large VRAM
  - Learning rate 0.002 (linear scaling with batch size)
  - DATA_WORKERS 4 — async data loading overlaps with GPU compute
  - bf16 mixed precision enabled in train.py (uses Tensor Cores)
"""

import os
import torch

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Architecture ---
NUM_RESIDUAL_BLOCKS = 10
NUM_FILTERS = 128
INPUT_PLANES = 102            # 8 history frames × 12 piece planes + 6 meta planes
POLICY_OUTPUT_SIZE = 4672

# --- Training ---
BATCH_SIZE = 512              # 2× local — fits comfortably in 24GB+ VRAM
LEARNING_RATE = 0.002         # Linear scaling: 2× batch → 2× LR
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
GRADIENT_CLIP = 1.0
VALUE_LOSS_WEIGHT = 0.5
DATA_WORKERS = 4              # Async data loading — safe with CUDA (unlike MPS)

# --- Data ---
MIN_RATING = 2000
MIN_MOVES = 10
TRAIN_SPLIT = 0.90
VAL_SPLIT = 0.05
TEST_SPLIT = 0.05

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
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
