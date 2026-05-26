"""
Configuration for Vast.ai CUDA training — v2 (SE blocks + WDL head + 105 input planes).

Differences from local M3 version:
  - CUDA device (no MPS)
  - Batch size 512 (2× local) — leverages large VRAM
  - Learning rate 0.002 (linear scaling with batch size)
  - DATA_WORKERS 16 — 112 cores available, 16 workers keep GPU pipeline full
  - bf16 mixed precision enabled in train.py (uses Tensor Cores on Ampere/Ada)
"""

import os
import torch

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
DATA_WORKERS = 16             # 112 CPU cores available — 16 workers keeps GPU pipeline full

# --- Data ---
MIN_RATING = 1500
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
