# Vast.ai Training Guide

## One-time setup

1. Create a Vast.ai account at vast.ai and add billing.
2. Rent a GPU instance — RTX 3090/4090 or A100 recommended (look for bf16 support = Ampere+).
3. Choose the **PyTorch** template image so CUDA/torch are pre-installed.
4. Note the SSH connection string from the **Connect** tab, e.g. `root@123.45.67.89` or `root@123.45.67.89:12345`.

## Deploy and train

```bash
# From the repo root:
./vast/deploy.sh root@<host>:<port>

# SSH in and run (or use the nohup one-liner printed by deploy.sh):
ssh -p <port> root@<host>
cd /workspace/chess-nn
python data/download_data.py       # ~15 min — downloads 200k games
python chess_nn/dataset.py         # ~30 min — converts to .npz chunks
python run.py supervised           # starts training (10 epochs, ~3-5h on 3090)
```

## Pull results when done

```bash
./vast/pull.sh root@<host>:<port>
# Checkpoints land in models/v1_history8_vast/checkpoints/
# best_model.pt is the one with lowest combined val loss
```

## Run against local model

```bash
python match.py models/v1_history8 models/v1_history8_vast --games 20
```

## Estimated times (RTX 3090, 200k games → ~8M positions)

| Stage              | Time       |
|--------------------|------------|
| Download 200k PGN  | ~15 min    |
| dataset.py         | ~25 min    |
| Training (10 ep)   | ~3-4 hours |
| **Total**          | **~4-5 h** |

Cost at ~$0.30/hr for a 3090: **~$1.50 total**.
