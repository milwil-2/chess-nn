"""
Training loop: teach the network to predict moves and evaluate positions.

Two losses are combined:
  - Policy loss: cross-entropy between predicted move probabilities and the actual move played
                 (like teaching a student: "in this position, this move was correct")
  - Value loss:  MSE between predicted win probability and actual game outcome
                 (like teaching: "this position eventually led to a win/loss/draw")

Total loss = policy_loss + 0.5 * value_loss
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    GRADIENT_CLIP, VALUE_LOSS_WEIGHT, CHECKPOINT_DIR
)
from chess_nn.model import ChessNet
from chess_nn.utils import save_checkpoint, log_training_step
from chess_nn.dataset import make_dataloaders

WARMUP_STEPS = 1000     # Ramp LR from 0 → LEARNING_RATE over first 1000 steps
LABEL_SMOOTHING = 0.1   # Fraction of probability redistributed uniformly across legal moves
DIRICHLET_ALPHA = 0.3   # Concentration — low = sparse/focused, AlphaZero uses 0.3 for chess
DIRICHLET_EPS   = 0.25  # Fraction of target replaced with Dirichlet noise during training


def get_lr_scale(step: int) -> float:
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    return 1.0


def make_policy_targets(policy_targets, legal_masks, add_noise: bool):
    """
    Build soft policy targets over legal moves only.

    Without noise: (1-smooth)*one_hot + smooth*uniform_over_legal
    With noise:    mix in Dirichlet noise so the model sees that less common
                   legal moves sometimes get played (human error / rare lines).

    Why Dirichlet?  α=0.3 produces sparse samples — most mass on 2-3 moves but
    nonzero everywhere legal.  This mimics how humans sometimes choose second-best
    moves without being purely random.
    """
    B, N = legal_masks.shape
    n_legal = legal_masks.float().sum(dim=1, keepdim=True).clamp(min=1)
    uniform_legal = legal_masks.float() / n_legal
    one_hot = F.one_hot(policy_targets, num_classes=N).float()
    soft = (1.0 - LABEL_SMOOTHING) * one_hot + LABEL_SMOOTHING * uniform_legal

    if add_noise:
        concentration = torch.full((N,), DIRICHLET_ALPHA, device=legal_masks.device)
        noise = torch.distributions.Dirichlet(concentration).sample((B,))
        noise = noise * legal_masks.float()
        noise = noise / noise.sum(dim=1, keepdim=True).clamp(min=1e-8)
        soft = (1.0 - DIRICHLET_EPS) * soft + DIRICHLET_EPS * noise

    return soft


def train(chunk_paths: list[str]):
    print(f"Training on device: {DEVICE}")

    model = ChessNet().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    train_loader, val_loader, _ = make_dataloaders(chunk_paths)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # --- Training ---
        model.train()
        train_policy_loss = 0.0
        train_value_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]",
                    unit="batch", dynamic_ncols=True)

        for boards, policy_targets, value_targets, legal_masks in pbar:
            boards = boards.to(DEVICE)
            policy_targets = policy_targets.to(DEVICE)
            value_targets = value_targets.to(DEVICE)
            legal_masks = legal_masks.to(DEVICE)

            if global_step < WARMUP_STEPS:
                scale = get_lr_scale(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = LEARNING_RATE * scale

            policy_logits, value_pred = model(boards)
            masked_logits = policy_logits.masked_fill(~legal_masks, float("-inf"))

            soft_targets = make_policy_targets(policy_targets, legal_masks, add_noise=True)
            policy_loss = F.kl_div(
                F.log_softmax(masked_logits, dim=1), soft_targets, reduction="batchmean"
            )
            value_loss = F.mse_loss(value_pred.squeeze(1), value_targets)
            total_loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            train_policy_loss += policy_loss.item()
            train_value_loss += value_loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                log_training_step(
                    epoch, global_step,
                    policy_loss.item(), value_loss.item(), total_loss.item()
                )

            # Live loss display in the progress bar
            pbar.set_postfix(
                policy=f"{train_policy_loss / n_batches:.4f}",
                value=f"{train_value_loss / n_batches:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.5f}",
            )

        pbar.close()
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_policy_loss = 0.0
        val_value_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [val]  ",
                        unit="batch", dynamic_ncols=True)

        with torch.no_grad():
            for boards, policy_targets, value_targets, legal_masks in val_pbar:
                boards = boards.to(DEVICE)
                policy_targets = policy_targets.to(DEVICE)
                value_targets = value_targets.to(DEVICE)
                legal_masks = legal_masks.to(DEVICE)

                policy_logits, value_pred = model(boards)
                masked_logits = policy_logits.masked_fill(~legal_masks, float("-inf"))
                soft_targets = make_policy_targets(policy_targets, legal_masks, add_noise=False)
                val_policy_loss += F.kl_div(
                    F.log_softmax(masked_logits, dim=1), soft_targets, reduction="batchmean"
                ).item()
                val_value_loss += F.mse_loss(value_pred.squeeze(1), value_targets).item()

                top5_pred = masked_logits.topk(5, dim=1).indices
                correct_top1 += (top5_pred[:, 0] == policy_targets).sum().item()
                correct_top5 += (top5_pred == policy_targets.unsqueeze(1)).any(dim=1).sum().item()
                total += len(policy_targets)

        val_pbar.close()

        n_val = len(val_loader)
        avg_val_policy = val_policy_loss / n_val
        avg_val_value = val_value_loss / n_val
        top1_acc = correct_top1 / total * 100
        top5_acc = correct_top5 / total * 100
        elapsed = time.time() - epoch_start
        mins, secs = divmod(int(elapsed), 60)

        print(f"\n{'='*52}")
        print(f"  Epoch {epoch}/{NUM_EPOCHS}  ({mins}m {secs}s)")
        print(f"  Val policy loss : {avg_val_policy:.4f}")
        print(f"  Val value loss  : {avg_val_value:.4f}")
        print(f"  Top-1 accuracy  : {top1_acc:.1f}%")
        print(f"  Top-5 accuracy  : {top5_acc:.1f}%")
        remaining = (NUM_EPOCHS - epoch) * elapsed
        r_mins, r_secs = divmod(int(remaining), 60)
        print(f"  ETA remaining   : ~{r_mins}m {r_secs}s")

        # Save best model
        val_total = avg_val_policy + VALUE_LOSS_WEIGHT * avg_val_value
        if val_total < best_val_loss:
            best_val_loss = val_total
            save_checkpoint(model, optimizer, epoch, val_total, "best_model.pt")
            print(f"  New best model saved (val_loss={val_total:.4f})")

        # Always save latest
        save_checkpoint(model, optimizer, epoch, val_total, f"epoch_{epoch:02d}.pt")

    print("\nTraining complete.")
    return model


if __name__ == "__main__":
    import glob
    chunk_paths = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "processed", "chunk_*.npz"
    )))
    if not chunk_paths:
        print("No processed data found. Run data/download_data.py first, then chess_nn/dataset.py.")
        sys.exit(1)
    train(chunk_paths)
