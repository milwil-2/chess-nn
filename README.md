# chess-nn

An AlphaZero-style chess engine built from scratch in PyTorch: a self-trained
convolutional policy/value network driven by Monte-Carlo Tree Search. The
project is organized so multiple model **variants** can be trained
independently and played head-to-head, making it easy to measure whether a
given change actually improves strength.

The engine learns from human games (supervised) and then from its own play
(reinforcement learning), and can be driven through a Pygame visualizer, a UCI
adapter (for cutechess / Lichess), or direct head-to-head matches.

> For a dense, internal-facing reference of every constant and code path, see
> [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md). This README is the high-level tour.

---

## Current architecture

The current best model is **`v3_vast`**. The network (`chess_nn/model.py`) is a
~5.6M-parameter residual CNN:

| Component | Detail |
|---|---|
| **Input** | 105 planes — 8 history frames × 12 piece planes + 9 meta planes (8×8 board) |
| **Body** | 3×3 conv → BN → ReLU, then **10 residual blocks × 128 filters** with Squeeze-and-Excitation (reduction 4) |
| **Policy head** | 1×1 conv → 8 planes → dense → **4672 logits** (AlphaZero 73-plane × 64-square move encoding) |
| **Value head** | 1×1 conv → 4 planes → dense 256 → **3-class WDL** (win/draw/loss from side-to-move POV) |
| **Search** | PUCT MCTS, `C_PUCT = 1.4`, root Dirichlet noise (α 0.3, ε 0.35) |

**Board encoding** uses a current-player perspective flip — the board is
mirrored when Black is to move, so "my pieces" always occupy a fixed set of
planes. The 8-frame history lets the network see piece trajectories and
repetitions; metadata (castling, en passant, 50-move clock, repetition flags)
is encoded from the current frame.

**Value head is WDL** (3-class softmax) rather than a single scalar;
`wdl_to_scalar()` collapses it to `P(win) − P(loss) ∈ [−1, 1]` for search.

---

## Version history

The model evolved through four trained variants plus a pre-history prototype.
Each `models/<variant>/` directory is a **self-contained snapshot** of the
`template/` codebase at creation time (created by `new_model.py`), so older
variants stay runnable as fixed baselines.

| Variant | Input | Trained on | Key parameters | What changed |
|---|---|---|---|---|
| *(prototype)* | 18 planes (1 frame) | M3 Mac | — | Original single-frame board encoding. Survives only as the Colab bootstrap checkpoint. |
| **v1_history8** | 102 planes (8×12 + 6 meta) | M3 Mac (local) | batch 256, LR 1e-3, MIN_RATING 2000 | First **8-frame history** model; 10 SE-residual blocks; WDL value head. |
| **v1_history8_vast** | 102 planes | Vast.ai GPU | batch 512, LR 2e-3 | Same architecture, ported to cloud GPU with linear LR scaling. |
| **v2_vast** | 105 planes (8×12 + 9 meta) | Vast.ai GPU | batch 2048, LR 4e-3, VALUE_LOSS_WEIGHT 1.0, MIN_RATING 1500 | **Expanded metadata** (added side-to-move + 2 repetition planes); large-batch training; rebalanced value loss. |
| **v3_vast** ⭐ | 105 planes | Vast.ai GPU | batch 2048, LR 4e-3, MIN_RATING 1800 | **Current best.** Adds Stockfish supervision, tactical oversampling, and a suite of inference-time search helpers (below). |

The progression was driven by a recurring symptom — the model played visibly
weak moves (early king walks, hung pieces) — and most of the work targets that.

---

## Improvements made

### Training data & supervision (`v3_vast`)
- **Higher rating floor** — `MIN_RATING` raised to 1800 so the network learns
  from fewer blunders (was 2000 local, dropped to 1500 in v2, settled at 1800).
- **Stockfish supervision** — 20% of positions are annotated with Stockfish's
  best move (depth 12) and added as an auxiliary policy loss (`SF_LOSS_WEIGHT = 0.3`),
  grounding the policy on engine-quality targets.
- **Tactical oversampling** — positions containing a hanging piece or fork
  (detected via `tactics.py`) are sampled **3×** during training so the network
  sees more tactically sharp positions.

### Network architecture
- **8-frame position history** (vs single-frame prototype) — exposes piece
  trajectories and repetition.
- **Squeeze-and-Excitation** residual blocks — channel-wise attention, cheap
  and consistently helpful.
- **WDL value head** — three-class win/draw/loss instead of a scalar, better
  calibrated for draw-heavy chess.
- **Widened policy/value head bottlenecks** and dropout reduced to 0.1.

### Search (MCTS)
- **Subtree reuse** — the search tree under the chosen move is carried into the
  next move instead of rebuilding from scratch (free simulations).
- **Shaped Dirichlet noise** (KataGo) — root exploration noise is restricted to
  plausible moves (`DIRICHLET_SHAPE_FLOOR`), so self-play stops polluting the
  training data with low-prior junk like early king pushes.
- **Root policy softmax temperature** — policy logits are flattened by `T`
  (swept to **1.50**) before the root softmax, letting search consider tactical
  alternatives the raw policy was over-confident about.
- **Stalemate guard** and a **cooperative stop flag** for cancelable,
  time-bounded search.

### Inference-time helpers (engine & visualizer; never used in self-play)
- **Opening book** — weighted Polyglot lookups (codekiddy book) for principled,
  varied openings.
- **Syzygy endgame tablebases** — 3-4-5-man perfect-play probing with
  distance-to-zero tie-breaking.
- **Cross-game transposition cache** — persisted MCTS visit counts, segmented
  per-checkpoint, auto-saved, mixed into priors to warm-start known positions.
- **Blunder filter** — at the root, prune moves that hang material unless the
  move is itself a capture/check.

### Tooling
- **UCI adapter** (`uci_engine.py`) — speaks full UCI so the engine can be
  driven by cutechess, `python-chess`, or a Lichess bot bridge.
- **Elo gauntlet** (`run_gauntlet.py`) — plays the engine vs Stockfish at
  calibrated `UCI_Elo` levels and logs ratings with confidence intervals.
- **Value-probe harness** — measures value-head MAE across position categories
  (symmetric / advantage / opening / endgame) to detect side-to-move bias.

See [`template/docs/improvements.md`](template/docs/improvements.md) for the
research-backed roadmap of what's shipped and what's planned next (opponent-policy
head, forced playouts, playout-cap randomization, transformer body).

---

## Repository layout

```
chess-nn/
  template/              canonical codebase — copied per-variant by new_model.py
    chess_nn/            model, encodings, MCTS, training, inference helpers
    viz/                 Pygame + web visualizers
    docs/improvements.md research roadmap
  models/<variant>/      self-contained snapshots (v1_history8, v2_vast, v3_vast, ...)
  scripts/               chessnn_uci.sh (UCI launcher), plot_elo.py
  match.py               head-to-head between any two variants
  new_model.py           create a new variant from the template
  PROJECT_SUMMARY.md     dense internal reference
```

Each variant carries its own `config.py`, checkpoints, and data — so editing
`template/` does **not** affect already-created variants.

---

## Quickstart

```bash
# Create a new variant from the template
python new_model.py my_variant
cd models/my_variant

# Download + encode Lichess training data
python data/download_data.py

# Supervised pretraining, then RL self-play
python run.py supervised
python run.py rl

# Watch it play
python run.py viz
```

### Matching two variants

```bash
# From repo root — 20 games with MCTS
python match.py models/v2_vast models/v3_vast

# More games / sims, or fast policy-only comparison
python match.py models/v2_vast models/v3_vast --games 40 --sims 100
python match.py models/v2_vast models/v3_vast --fast
```

### Running as a UCI engine

```bash
# Drives v3_vast's best_model.pt over the UCI protocol
scripts/chessnn_uci.sh
# overridable: CHESSNN_CHECKPOINT, CHESSNN_SIMS, CHESSNN_PYTHON
```

---

## Measuring strength

Strength is tracked two ways:

1. **Local gauntlet vs Stockfish** — `models/v3_vast/run_gauntlet.py` plays the
   engine against Stockfish at several `UCI_Elo` levels and writes ratings (with
   CIs) to `logs/elo_history.csv`. Plot with `scripts/plot_elo.py`.
2. **Lichess bot** — the UCI adapter can be bridged to
   [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) to play rated
   or casual games against the live bot pool.

Current `v3_vast` strength is roughly **~700 Elo** on the local Stockfish
gauntlet (wide confidence interval) — weak, which is exactly what the next
training cycle (higher-quality data + the planned architecture changes) aims to
fix.

---

## Status & caveats

- The `models/<variant>/data/` and `checkpoints/` directories, Syzygy tables,
  opening books, and the MCTS cache are **gitignored** (large / regenerable).
- The inference helpers (book, Syzygy, cache, blunder filter) are **never** wired
  into self-play — training always sees the raw network so the learned policy
  isn't contaminated by hand-coded heuristics.
- GPU training runs on Vast.ai (`vast/` scripts); local inference/visualization
  runs on Apple Silicon (MPS) or CPU.
