"""
Evaluate how well the trained model plays chess.

Metrics:
  - Top-1 accuracy: did the model pick the exact same move a human played?
  - Top-5 accuracy: was the human's move in the model's top 5 choices?
  - Value MSE: how close was the win probability estimate to the real outcome?
  - Play test: win rate vs. a random-move player
"""

import sys
import os
import random
import torch
import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE
from chess_nn.model import ChessNet
from chess_nn.board_encoding import board_to_tensor
from chess_nn.move_encoding import move_to_index, get_legal_move_indices, policy_to_moves
from chess_nn.utils import load_checkpoint


def evaluate_dataset(model, data_loader):
    """Compute top-1, top-5 accuracy and value MSE on a dataset split."""
    model.eval()
    correct_top1 = correct_top5 = total = 0
    value_mse_sum = 0.0

    with torch.no_grad():
        for boards, policy_targets, value_targets in data_loader:
            boards = boards.to(DEVICE)
            policy_targets = policy_targets.to(DEVICE)
            value_targets = value_targets.to(DEVICE)

            policy_logits, value_pred = model(boards)

            top5 = policy_logits.topk(5, dim=1).indices
            correct_top1 += (top5[:, 0] == policy_targets).sum().item()
            correct_top5 += (top5 == policy_targets.unsqueeze(1)).any(dim=1).sum().item()
            total += len(policy_targets)

            value_mse_sum += ((value_pred.squeeze(1) - value_targets) ** 2).sum().item()

    return {
        "top1_acc": correct_top1 / total * 100,
        "top5_acc": correct_top5 / total * 100,
        "value_mse": value_mse_sum / total,
    }


def select_move(model, board: chess.Board, temperature: float = 1.0) -> chess.Move:
    """
    Pick a move using the model's policy output.

    temperature=0: always pick the highest-probability move (greedy)
    temperature=1: sample proportionally to probabilities (more variety)
    """
    device = next(model.parameters()).device
    tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, value = model(tensor)

    policy = policy_logits.squeeze(0).cpu().numpy()
    legal_indices = get_legal_move_indices(board)

    # Mask illegal moves
    masked = np.full(4672, -1e9, dtype=np.float32)
    for idx in legal_indices:
        masked[idx] = policy[idx]

    if temperature == 0:
        chosen_idx = legal_indices[np.argmax([masked[i] for i in legal_indices])]
    else:
        # Apply temperature, then softmax
        scaled = np.array([masked[i] / temperature for i in legal_indices])
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        chosen_idx = np.random.choice(legal_indices, p=probs)

    from chess_nn.move_encoding import index_to_move
    return index_to_move(chosen_idx, board)


def play_vs_random(model, n_games: int = 50, temperature: float = 0.5) -> dict:
    """Play n games against a random-move opponent. Model plays both colors."""
    wins = draws = losses = 0

    for game_idx in range(n_games):
        board = chess.Board()
        model_is_white = (game_idx % 2 == 0)

        while not board.is_game_over():
            if board.turn == chess.WHITE and model_is_white:
                move = select_move(model, board, temperature)
            elif board.turn == chess.BLACK and not model_is_white:
                move = select_move(model, board, temperature)
            else:
                move = random.choice(list(board.legal_moves))
            board.push(move)

        result = board.result()
        if (result == "1-0" and model_is_white) or (result == "0-1" and not model_is_white):
            wins += 1
        elif result == "1/2-1/2":
            draws += 1
        else:
            losses += 1

    return {"wins": wins, "draws": draws, "losses": losses, "win_rate": wins / n_games * 100}


if __name__ == "__main__":
    import glob
    from chess_nn.dataset import make_dataloaders

    chunk_paths = sorted(glob.glob(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "chunk_*.npz")
    ))

    model = ChessNet().to(DEVICE)
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "checkpoints", "best_model.pt"
    )
    load_checkpoint(checkpoint_path, model)

    _, val_loader, test_loader = make_dataloaders(chunk_paths)

    print("Evaluating on test set...")
    metrics = evaluate_dataset(model, test_loader)
    print(f"  Top-1 accuracy: {metrics['top1_acc']:.1f}%")
    print(f"  Top-5 accuracy: {metrics['top5_acc']:.1f}%")
    print(f"  Value MSE:      {metrics['value_mse']:.4f}")

    print("\nPlaying 20 games vs. random mover...")
    results = play_vs_random(model, n_games=20)
    print(f"  Wins: {results['wins']}  Draws: {results['draws']}  Losses: {results['losses']}")
    print(f"  Win rate: {results['win_rate']:.0f}%")
