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
NUM_RESIDUAL_BLOCKS = 5       # Depth of the network (more = stronger but slower)
NUM_FILTERS = 128             # Width of each layer (more = more capacity)
INPUT_PLANES = 18             # 12 piece planes + 4 castling + 1 en passant + 1 turn
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
