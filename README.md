# chess-nn

AlphaZero-style chess neural network. Multiple model variants can be trained independently and matched against each other.

## Structure

```
chess-nn/
  template/        base code — copy this to create a new model variant
  models/          trained model variants live here (each is self-contained)
  match.py         run head-to-head games between any two variants
  new_model.py     create a new variant from the template
```

## Creating a model

```bash
python new_model.py v1_baseline
cd models/v1_baseline

# Download + process training data
python data/download_data.py
python chess_nn/dataset.py

# Train
python run.py supervised
python run.py rl

# Visualize
python run.py viz
```

## Matching models

```bash
# From repo root — MCTS, 20 games
python match.py models/v1_baseline models/v2_wider_heads

# More games, more simulations
python match.py models/v1_baseline models/v2_wider_heads --games 40 --sims 100

# Fast policy-only (no MCTS) — good for quick comparisons
python match.py models/v1_baseline models/v2_wider_heads --fast

# Use a specific checkpoint
python match.py models/v1_baseline models/v2_wider_heads --checkpoint rl_best_model.pt
```

## Customizing a variant

Edit `models/<name>/config.py` before training. Key architecture knobs:

| Setting | Default | Effect |
|---------|---------|--------|
| `NUM_RESIDUAL_BLOCKS` | 5 | Deeper = stronger, slower |
| `NUM_FILTERS` | 128 | Wider = more capacity |
| `RL_SIMULATIONS` | 200 | MCTS strength during self-play |

See `template/docs/improvements.md` for a prioritized list of changes and their expected impact.
