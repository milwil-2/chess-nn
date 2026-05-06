# Chess Neural Network — Agent Context

AlphaZero-style chess engine built in Python. PyTorch CNN trained on Lichess master games, with MCTS self-play RL loop and a Pygame visualizer.

---

## Current State

- **Supervised training**: complete through epoch 10/15. Checkpoints saved as `checkpoints/epoch_01.pt` … `epoch_10.pt` and `best_model.pt`.
- **RL self-play loop**: code exists (`chess_nn/train_rl.py`, `chess_nn/mcts.py`, `chess_nn/self_play.py`), not yet run.
- **Pygame visualizer**: complete (`viz/app.py`). Shows board, move probability arrows, activation heatmaps, value bar, tactic highlights.
- **Lichess bot**: scaffolded in `bot/`, not yet tested online.

---

## Project Structure

```
chess-nn/
├── config.py                   # ALL hyperparameters and paths — edit here, not in training code
├── run.py                      # Pipeline entry point (see Commands below)
├── requirements.txt
├── chess_nn/
│   ├── board_encoding.py       # board_to_tensor() → ndarray (18, 8, 8)
│   ├── move_encoding.py        # move_to_index(), index_to_move(), get_legal_move_indices()
│   ├── model.py                # ChessNet: 5 residual blocks, policy + value heads
│   ├── dataset.py              # ChessDataset (lazy chunk loading), ChunkBatchSampler, make_dataloaders()
│   ├── train.py                # Supervised training loop
│   ├── train_rl.py             # RL self-play training loop
│   ├── mcts.py                 # Monte Carlo Tree Search
│   ├── self_play.py            # Generate self-play games using MCTS
│   ├── evaluate.py             # select_move(), move accuracy metrics
│   ├── tactics.py              # detect_tactics() — fork/pin/skewer detection for viz
│   └── utils.py                # save_checkpoint(), load_checkpoint(), log_training_step()
├── data/
│   ├── download_data.py        # Stream-download + filter Lichess PGN (clears old data first)
│   └── raw/                    # filtered_games.pgn lives here (gitignored)
├── data/processed/             # chunk_0000.npz … chunk_NNNN.npz (gitignored, ~450MB each)
├── checkpoints/                # *.pt model files (gitignored)
├── logs/                       # training_log.jsonl (gitignored)
├── viz/
│   ├── app.py                  # Main Pygame game loop
│   ├── board_renderer.py       # Board + piece rendering
│   ├── move_arrows.py          # Probability arrows overlay
│   ├── heatmap.py              # Activation heatmap overlay
│   ├── value_bar.py            # Win probability bar
│   ├── network_viz.py          # Network activation visualizer (separate window)
│   └── web_server.py           # WebSocket-based web visualizer
└── bot/
    ├── lichess_bot.py          # Lichess event loop (berserk library)
    └── engine.py               # Model wrapped as chess engine
```

---

## Architecture

| Property | Value |
|----------|-------|
| Input | `(18, 8, 8)` tensor — 12 piece planes + 4 castling + en passant + turn |
| Residual blocks | 5 |
| Filters | 128 |
| Policy output | 4672 (73 move types × 64 squares, AlphaZero-style) |
| Value output | tanh scalar, +1 = current player wins |
| Parameters | ~2.5M |
| Device | Auto-detected: MPS (Apple Silicon) > CUDA > CPU |

Board is always encoded from the **current player's perspective** (flipped for Black).

---

## Key Conventions

- **All hyperparameters in `config.py`** — batch size, learning rate, RL settings, paths. Never hardcode in training files.
- **Data pipeline**: `download_data.py` → `chess_nn/dataset.py` → `train.py`
- **Lazy chunk loading**: `ChessDataset` loads one `.npz` chunk (~450MB) at a time. Do not change it to eager loading — the full dataset is ~5GB.
- **ChunkBatchSampler**: all indices in a batch come from the same chunk. Required for performance — do not replace with standard random sampler.
- **`num_workers=0`** in DataLoaders — MPS + multiprocessing causes issues.
- **MPS warmup**: first ~20 training batches are slow (Metal shader JIT compilation). Normal.
- **Checkpoints**: `save_checkpoint()` / `load_checkpoint()` in `utils.py`. Always pass `scheduler` arg.
- **Policy encoding**: `move_to_index()` takes `(move, board)` — needs board for context (en passant, promotion). Do not call with move alone.
- **Legal move masking**: policy logits are always masked with `~legal_masks` → `-inf` before softmax. Never skip this.

---

## Commands

```bash
# Run from repo root with venv active
python run.py status                        # Show checkpoints + data state
python run.py supervised                    # Train on Lichess data (resumes if epochs exist)
python run.py supervised --resume           # Explicitly resume from latest epoch_XX.pt
python run.py rl                            # Self-play RL loop (10 iterations default)
python run.py rl --iterations 3 --games 10 # Fewer iterations for testing
python run.py viz                           # Launch Pygame visualizer
python run.py viz --web                     # Launch WebSocket web visualizer
python run.py selfplay                      # Generate self-play data only (no training)

# Data pipeline (run once before supervised training)
python data/download_data.py               # Download + filter Lichess games (clears old data)
python chess_nn/dataset.py                 # Convert PGN → .npz chunks
```

---

## Data Flow

```
Lichess PGN (stream download)
  → download_data.py → data/raw/filtered_games.pgn
  → dataset.py → data/processed/chunk_NNNN.npz  (100k positions each)
  → ChessDataset (lazy) → ChunkBatchSampler → DataLoader
  → train.py → checkpoints/epoch_XX.pt + best_model.pt
  → train_rl.py → self-play loop → RL-improved model
```

---

## Environment

- Python 3.12, virtualenv at `.venv/`
- Apple Silicon (MPS). Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` if OOM.
- Key deps: `torch>=2.1`, `python-chess`, `pygame`, `berserk`, `numpy`, `zstandard`
- No `torch.compile()` — not supported reliably on MPS.
