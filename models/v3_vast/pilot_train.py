"""P2 pilot (GH #6): train one epoch on chunk_0000_whitepov.npz to test
the value-head bias hypothesis.

What this script does:
  * loads checkpoints/best_model.pt as the starting weights (so the probe
    eval measures DELTA from the current baseline, not a from-scratch run)
  * trains exactly one epoch on the re-labeled (W-POV) chunk
  * saves the resulting checkpoint to checkpoints/pilot_whitepov_e1.pt

What this script intentionally does NOT do:
  * touch best_model.pt
  * touch any other shipped checkpoint
  * write a second chunk back to data/processed/

Run on MPS locally (~10 min for one chunk) or CUDA on Vast.
"""

from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import (  # noqa: E402
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEVICE,
    LEARNING_RATE,
    POLICY_OUTPUT_SIZE,
    VALUE_LOSS_WEIGHT,
    WEIGHT_DECAY,
)
from chess_nn.model import ChessNet  # noqa: E402
from chess_nn.utils import load_checkpoint, save_checkpoint  # noqa: E402


CHUNK = os.environ.get(
    "PILOT_CHUNK",
    os.path.join(HERE, "data", "processed", "chunk_0000_whitepov.npz"),
)
INIT_CKPT = os.environ.get(
    "PILOT_INIT_CKPT",
    os.path.join(CHECKPOINT_DIR, "best_model.pt"),
)
OUT_CKPT_NAME = os.environ.get("PILOT_OUT_CKPT_NAME", "pilot_whitepov_e1.pt")


def _device_for_training():
    """Prefer MPS on Apple Silicon when CUDA isn't available."""
    if DEVICE.type == "cuda":
        return DEVICE
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    if not os.path.exists(CHUNK):
        print(f"ERROR: chunk not found: {CHUNK}", file=sys.stderr)
        print("       Run relabel_chunk.py first.", file=sys.stderr)
        sys.exit(1)

    device = _device_for_training()
    print(f"device:        {device}")
    print(f"chunk:         {CHUNK}")
    print(f"init ckpt:     {INIT_CKPT}")
    print(f"output ckpt:   {os.path.join(CHECKPOINT_DIR, OUT_CKPT_NAME)}")

    model = ChessNet().to(device)
    if os.path.exists(INIT_CKPT):
        epoch_start, prev_loss = load_checkpoint(INIT_CKPT, model)
        print(f"  → inited from {INIT_CKPT} (epoch {epoch_start}, loss {prev_loss:.4f})")
    else:
        print(f"  → init ckpt not found; starting from random init")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    data = np.load(CHUNK)
    boards_np = data["boards"]  # (N, 105, 8, 8) float32
    values_np = data["values"]  # (N,) int64
    policies_np = data["policies"]  # (N,) int64 — move played
    legal_packed = data.get("legal_masks")  # (N, ceil(POLICY/8)) uint8
    N = len(boards_np)
    print(f"\ntraining on {N:,} positions, 1 epoch, batch_size={BATCH_SIZE}")

    # Pre-unpack legal masks into one big bool array so the hot loop stays cheap.
    if legal_packed is not None:
        legal_masks_np = np.unpackbits(legal_packed, axis=1)[:, :POLICY_OUTPUT_SIZE].astype(bool)
    else:
        legal_masks_np = None

    # Shuffle once so the pilot epoch isn't biased by chunk order.
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(N)

    model.train()
    losses = []
    pol_losses = []
    val_losses = []
    n_batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.monotonic()

    for step in range(n_batches):
        idx = perm[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        if len(idx) == 0:
            continue

        b_boards = torch.from_numpy(boards_np[idx].copy()).to(device)
        b_values = torch.from_numpy(values_np[idx].astype(np.int64).copy()).to(device)
        b_policies = torch.from_numpy(policies_np[idx].astype(np.int64).copy()).to(device)
        if legal_masks_np is not None:
            b_mask = torch.from_numpy(legal_masks_np[idx].copy()).to(device)
        else:
            b_mask = None

        policy_logits, value_logits = model(b_boards)

        if b_mask is not None:
            masked_logits = policy_logits.masked_fill(~b_mask, float("-inf"))
        else:
            masked_logits = policy_logits

        # Hard-target cross-entropy on the move that was actually played.
        pol_loss = F.cross_entropy(masked_logits, b_policies)
        val_loss = F.cross_entropy(value_logits, b_values)
        loss = pol_loss + VALUE_LOSS_WEIGHT * val_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        pol_losses.append(pol_loss.item())
        val_losses.append(val_loss.item())

        if step % 25 == 0 or step == n_batches - 1:
            recent_loss = sum(losses[-10:]) / min(10, len(losses))
            recent_pol = sum(pol_losses[-10:]) / min(10, len(pol_losses))
            recent_val = sum(val_losses[-10:]) / min(10, len(val_losses))
            print(
                f"  step {step:>4}/{n_batches}  "
                f"loss={recent_loss:.3f}  pol={recent_pol:.3f}  val={recent_val:.3f}"
            )

    elapsed = time.monotonic() - t0
    final_loss = sum(losses[-10:]) / min(10, len(losses))
    print(
        f"\nfinal loss (last 10 batches): {final_loss:.4f}  "
        f"(pol {sum(pol_losses[-10:])/min(10,len(pol_losses)):.4f}, "
        f"val {sum(val_losses[-10:])/min(10,len(val_losses)):.4f})"
    )
    print(f"elapsed: {int(elapsed//60)}m {int(elapsed%60):02d}s")

    save_checkpoint(model, optimizer, epoch=1, loss=final_loss, filename=OUT_CKPT_NAME)


if __name__ == "__main__":
    main()
