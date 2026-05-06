"""
Reinforcement learning training loop — AlphaZero style.

Each iteration:
  1. Generate self-play games using the current model + MCTS
  2. Train on those games (policy loss + value loss)
  3. Evaluate new model vs old model (play N head-to-head games)
  4. Keep the winner, discard the loser
  5. Repeat

Why do we evaluate and potentially discard?
  Neural networks can occasionally get worse after an update if the self-play data
  was unlucky or the learning rate was too high. Head-to-head evaluation catches this.
  If the new model wins >55% of games, it's genuinely better. Otherwise keep the old one.

With 200 simulations and a small model, each iteration takes ~10-20 minutes on M-chip.
The model will gradually improve with each iteration.
"""

import gc
import os
import sys
import glob
import copy
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHECKPOINT_DIR, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, VALUE_LOSS_WEIGHT, DEVICE,
    RL_GAMES_PER_ITER, RL_SIMULATIONS, RL_CHUNK_SIZE,
    RL_HISTORY_FILES, RL_EPOCHS, RL_LR, RL_EVAL_GAMES, RL_WIN_THRESHOLD,
)
from chess_nn.model import ChessNet
from chess_nn.utils import save_checkpoint, load_checkpoint
from chess_nn.self_play import generate_games
from chess_nn.mcts import MCTS
import bisect
import chess


class SelfPlayDataset(Dataset):
    """
    Dataset for self-play data. Keeps each file's arrays separate instead of
    concatenating them — halves peak RAM vs np.concatenate (no second copy during merge).

    Policy targets are MCTS visit distributions (shape 4672), not single move indices.
    """

    def __init__(self, npz_paths: list[str], max_files: int = RL_HISTORY_FILES):
        paths = sorted(npz_paths)[-max_files:]
        self._boards: list[np.ndarray] = []
        self._policies: list[np.ndarray] = []
        self._values: list[np.ndarray] = []
        self._cumulative = [0]

        for path in paths:
            data = np.load(path)
            self._boards.append(data["boards"])
            self._policies.append(data["policies"])
            self._values.append(data["values"])
            self._cumulative.append(self._cumulative[-1] + len(data["boards"]))

        print(f"RL dataset: {self._cumulative[-1]:,} positions from {len(paths)} file(s)")

    def __len__(self):
        return self._cumulative[-1]

    def __getitem__(self, idx):
        # bisect finds which file contains this global index
        file_idx = bisect.bisect_right(self._cumulative, idx) - 1
        local_idx = idx - self._cumulative[file_idx]
        return (
            torch.from_numpy(self._boards[file_idx][local_idx].copy()),
            torch.from_numpy(self._policies[file_idx][local_idx].copy()),
            torch.tensor(self._values[file_idx][local_idx]),
        )


def train_on_selfplay(model, dataset: SelfPlayDataset, epochs: int = RL_EPOCHS, lr: float = RL_LR) -> float:
    """
    Train the model on self-play data for a few epochs.
    Returns the final average loss.

    Note: policy loss here uses KL divergence (not cross-entropy) because
    the target is a probability distribution from MCTS, not a single move index.
    KL divergence measures how different two probability distributions are.
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    model.to(DEVICE)
    model.train()
    final_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for boards, policy_targets, value_targets in loader:
            boards = boards.to(DEVICE)
            policy_targets = policy_targets.to(DEVICE)
            value_targets = value_targets.to(DEVICE)
            policy_logits, value_pred = model(boards)

            # Policy loss: KL divergence between MCTS distribution and network output
            # log_softmax gives log-probabilities from the network
            # MCTS distribution is already probabilities (sums to 1)
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()

            # Value loss: MSE between predicted win probability and actual outcome
            value_loss = F.mse_loss(value_pred.squeeze(1), value_targets)

            loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        print(f"  RL epoch {epoch+1}/{epochs}: loss={avg:.4f}")
        final_loss = avg

    return final_loss


def evaluate_models(new_model, old_model, num_games: int = RL_EVAL_GAMES,
                    simulations: int = 50) -> float:
    """
    Play head-to-head games between new and old model.
    Uses fewer simulations than training (50 vs 200) for speed.
    Returns new model win rate.

    Each model plays half the games as White, half as Black (to eliminate colour bias).
    """
    new_mcts = MCTS(new_model, num_simulations=simulations)
    old_mcts = MCTS(old_model, num_simulations=simulations)

    new_wins = draws = old_wins = 0

    for game_idx in range(num_games):
        board = chess.Board()
        new_is_white = (game_idx % 2 == 0)

        while not board.is_game_over():
            if (board.turn == chess.WHITE) == new_is_white:
                move = new_mcts.search(board, temperature=0)
            else:
                move = old_mcts.search(board, temperature=0)
            if move in board.legal_moves:
                board.push(move)
            else:
                board.push(random.choice(list(board.legal_moves)))

        result = board.result()
        if result == "1-0":
            if new_is_white:
                new_wins += 1
            else:
                old_wins += 1
        elif result == "0-1":
            if new_is_white:
                old_wins += 1
            else:
                new_wins += 1
        else:
            draws += 1

    del new_mcts, old_mcts

    win_rate = new_wins / num_games
    print(f"  Evaluation: new={new_wins} draws={draws} old={old_wins}  win_rate={win_rate:.2f}")
    return win_rate


def run_rl_loop(num_iterations: int = 10, start_checkpoint: str = "best_model.pt",
                games_per_iter: int = None, num_simulations: int = None,
                eval_games: int = None, rl_epochs: int = None):
    """
    Main RL loop. Runs `num_iterations` cycles of self-play → train → evaluate.

    Optional overrides (default to config.py values if not set):
      games_per_iter    — self-play games per iteration
      num_simulations   — MCTS simulations per move
      eval_games        — head-to-head games for model comparison
      rl_epochs         — training epochs per iteration
    """
    games_per_iter  = games_per_iter  or RL_GAMES_PER_ITER
    num_simulations = num_simulations or RL_SIMULATIONS
    eval_games      = eval_games      or RL_EVAL_GAMES
    rl_epochs       = rl_epochs       or RL_EPOCHS
    selfplay_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "processed", "self_play"
    )
    os.makedirs(selfplay_dir, exist_ok=True)

    # Load current best model
    model = ChessNet()
    checkpoint_path = os.path.join(CHECKPOINT_DIR, start_checkpoint)
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model)
        print(f"Starting from checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found — starting from scratch (random weights)")
    model.eval()

    for iteration in range(num_iterations):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration + 1} / {num_iterations}")
        print(f"{'='*50}")

        # Step 1: Generate self-play data — keep model on CPU so MCTS tensor ops stay cheap
        print("\n[1] Generating self-play games...")
        model.cpu()
        selfplay_path = generate_games(
            model,
            num_games=games_per_iter,
            num_simulations=num_simulations,
            output_dir=selfplay_dir,
            iteration=iteration,
        )
        gc.collect()

        # Step 2: Train a copy of the model on new self-play data (uses DEVICE / MPS)
        print("\n[2] Training on self-play data...")
        new_model = copy.deepcopy(model)

        all_selfplay = sorted(glob.glob(os.path.join(selfplay_dir, "selfplay_iter*.npz")))
        dataset = SelfPlayDataset(all_selfplay)
        train_on_selfplay(new_model, dataset, epochs=rl_epochs)  # moves new_model to DEVICE internally
        del dataset
        new_model.cpu()
        new_model.eval()
        gc.collect()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Step 3: Evaluate new model vs old (both on CPU for MCTS)
        print("\n[3] Evaluating new model vs old...")
        old_model = copy.deepcopy(model)
        old_model.eval()
        win_rate = evaluate_models(new_model, old_model, num_games=eval_games)
        del old_model
        gc.collect()

        # Step 4: Keep winner
        if win_rate >= RL_WIN_THRESHOLD:
            print(f"\n  New model promoted (win rate {win_rate:.0%} >= {RL_WIN_THRESHOLD:.0%})")
            del model
            model = new_model
            save_checkpoint(model, torch.optim.Adam(model.parameters()), iteration, win_rate,
                            f"rl_best_model.pt")
            save_checkpoint(model, torch.optim.Adam(model.parameters()), iteration, win_rate,
                            f"rl_iter_{iteration:03d}.pt")
        else:
            print(f"\n  Old model kept (new win rate {win_rate:.0%} < {RL_WIN_THRESHOLD:.0%})")
            del new_model
        gc.collect()

    print("\nRL training complete.")
    return model


if __name__ == "__main__":
    run_rl_loop(num_iterations=5)
