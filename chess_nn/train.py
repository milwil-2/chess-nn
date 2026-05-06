"""
Training loop: teach the network to predict moves and evaluate positions.

Two losses are combined:
  - Policy loss: cross-entropy between predicted move probabilities and the actual move played
                 (like teaching a student: "in this position, this move was correct")
  - Value loss:  MSE between predicted win probability and actual game outcome
                 (like teaching: "this position eventually led to a win/loss/draw")

Total loss = policy_loss + 0.5 * value_loss
"""

import glob
import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    GRADIENT_CLIP, VALUE_LOSS_WEIGHT, CHECKPOINT_DIR
)
from chess_nn.model import ChessNet
from chess_nn.utils import save_checkpoint, load_checkpoint, log_training_step
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
        concentration = torch.full((N,), DIRICHLET_ALPHA)  # CPU — MPS lacks _sample_dirichlet
        noise = torch.distributions.Dirichlet(concentration).sample((B,)).to(legal_masks.device)
        noise = noise * legal_masks.float()
        noise = noise / noise.sum(dim=1, keepdim=True).clamp(min=1e-8)
        soft = (1.0 - DIRICHLET_EPS) * soft + DIRICHLET_EPS * noise

    return soft


def train(chunk_paths: list[str], resume: bool = False):
    print(f"Training on device: {DEVICE}")

    model = ChessNet().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    start_epoch = 0
    if resume:
        epoch_ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pt")))
        if epoch_ckpts:
            start_epoch, _ = load_checkpoint(epoch_ckpts[-1], model, optimizer, scheduler)
            print(f"Resuming from epoch {start_epoch} → continuing to epoch {NUM_EPOCHS}")
        else:
            print("No epoch checkpoints found — starting from scratch")

    if start_epoch >= NUM_EPOCHS:
        print(f"Already completed {NUM_EPOCHS} epochs. Done.")
        return model

    train_loader, val_loader, _ = make_dataloaders(chunk_paths)

    best_val_loss = float("inf")
    global_step = start_epoch * len(train_loader)  # Approximate step count for LR warmup

    # How often to print a status line (every 10% of an epoch, or every 100 batches)
    total_train_batches = len(train_loader)
    print_every = max(100, total_train_batches // 10)

    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        print(f"\n{'='*54}")
        print(f"  Epoch {epoch} / {NUM_EPOCHS}")
        if epoch == 1:
            print("  Note: first ~20 batches are slow while MPS warms up.")
        print(f"{'='*54}")

        # --- Training ---
        model.train()
        train_policy_loss = 0.0
        train_value_loss  = 0.0
        n_batches = 0
        MPS_WARMUP = 20  # batches to skip before trusting the ETA

        for boards, policy_targets, value_targets, legal_masks in train_loader:
            boards         = boards.to(DEVICE)
            policy_targets = policy_targets.to(DEVICE)
            value_targets  = value_targets.to(DEVICE)
            legal_masks    = legal_masks.to(DEVICE)

            if global_step < WARMUP_STEPS:
                scale = get_lr_scale(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = LEARNING_RATE * scale

            policy_logits, value_pred = model(boards)
            masked_logits = policy_logits.masked_fill(~legal_masks, float("-inf"))

            soft_targets = make_policy_targets(policy_targets, legal_masks, add_noise=True)
            log_probs = F.log_softmax(masked_logits, dim=1).masked_fill(~legal_masks, 0.0)
            policy_loss = -(soft_targets * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred.squeeze(1), value_targets)
            total_loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            train_policy_loss += policy_loss.item()
            train_value_loss  += value_loss.item()
            n_batches  += 1
            global_step += 1

            if global_step % 100 == 0:
                log_training_step(
                    epoch, global_step,
                    policy_loss.item(), value_loss.item(), total_loss.item()
                )

            if n_batches % print_every == 0 or n_batches == total_train_batches:
                pct     = n_batches / total_train_batches * 100
                elapsed = time.time() - epoch_start
                avg_policy = train_policy_loss / n_batches
                avg_value  = train_value_loss  / n_batches
                lr_now     = optimizer.param_groups[0]["lr"]

                # Only show ETA after MPS has warmed up
                if n_batches > MPS_WARMUP:
                    secs_per_batch = elapsed / n_batches
                    remaining_batches = total_train_batches - n_batches
                    eta_secs = secs_per_batch * remaining_batches
                    eta_m, eta_s = divmod(int(eta_secs), 60)
                    eta_str = f"ETA {eta_m}m {eta_s:02d}s"
                else:
                    eta_str = "ETA warming up..."

                e_m, e_s = divmod(int(elapsed), 60)
                print(
                    f"  {n_batches:>5}/{total_train_batches}  [{pct:5.1f}%]"
                    f"  policy {avg_policy:.3f}  value {avg_value:.3f}"
                    f"  lr {lr_now:.5f}  |  {e_m}m {e_s:02d}s elapsed  {eta_str}"
                )

        scheduler.step()

        # --- Validation ---
        print("\n  Running validation...")
        model.eval()
        val_policy_loss = 0.0
        val_value_loss  = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        with torch.no_grad():
            for boards, policy_targets, value_targets, legal_masks in val_loader:
                boards         = boards.to(DEVICE)
                policy_targets = policy_targets.to(DEVICE)
                value_targets  = value_targets.to(DEVICE)
                legal_masks    = legal_masks.to(DEVICE)

                policy_logits, value_pred = model(boards)
                masked_logits = policy_logits.masked_fill(~legal_masks, float("-inf"))
                soft_targets  = make_policy_targets(policy_targets, legal_masks, add_noise=False)
                log_probs = F.log_softmax(masked_logits, dim=1).masked_fill(~legal_masks, 0.0)
                val_policy_loss += (-(soft_targets * log_probs).sum(dim=1).mean()).item()
                val_value_loss += F.mse_loss(value_pred.squeeze(1), value_targets).item()

                top5_pred = masked_logits.topk(5, dim=1).indices
                correct_top1 += (top5_pred[:, 0] == policy_targets).sum().item()
                correct_top5 += (top5_pred == policy_targets.unsqueeze(1)).any(dim=1).sum().item()
                total += len(policy_targets)

        n_val       = len(val_loader)
        avg_val_policy = val_policy_loss / n_val
        avg_val_value  = val_value_loss  / n_val
        top1_acc = correct_top1 / total * 100
        top5_acc = correct_top5 / total * 100

        elapsed = time.time() - epoch_start
        e_m, e_s = divmod(int(elapsed), 60)

        print(f"\n  --- Epoch {epoch} results ---")
        print(f"  Time this epoch    : {e_m}m {e_s:02d}s")
        print(f"  Policy loss        : {avg_val_policy:.4f}  (lower = better move picks)")
        print(f"  Value loss         : {avg_val_value:.4f}  (lower = better win-rate estimate)")
        print(f"  Move accuracy top1 : {top1_acc:.1f}%   top5: {top5_acc:.1f}%")

        remaining_epochs = NUM_EPOCHS - epoch
        if remaining_epochs > 0:
            r_m, r_s = divmod(int(elapsed * remaining_epochs), 60)
            print(f"  ETA for remaining  : ~{r_m}m {r_s:02d}s  ({remaining_epochs} epochs left)")

        # Save best model
        val_total = avg_val_policy + VALUE_LOSS_WEIGHT * avg_val_value
        if val_total < best_val_loss:
            best_val_loss = val_total
            save_checkpoint(model, optimizer, epoch, val_total, "best_model.pt", scheduler)
            print(f"  >> New best model saved  (combined loss {val_total:.4f})")

        save_checkpoint(model, optimizer, epoch, val_total, f"epoch_{epoch:02d}.pt", scheduler)

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
