"""
Self-play: the model plays games against itself using MCTS.

For every position in every game we record:
  - board tensor          (input to the network)
  - MCTS visit distribution  (better policy target than raw network output)
  - game result           (value target: +1 win, 0 draw, -1 loss for the player to move)

Why is the MCTS distribution a better policy target than the raw network?
  The network makes one guess. MCTS runs hundreds of simulations and aggregates them.
  Training the network to imitate MCTS = the network learns to think ahead,
  even though it's just a single forward pass at inference time.

This file generates the data. train_rl.py consumes it.
"""

import gc
import os
import sys
import chess
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_DIR, RL_CHUNK_SIZE
from chess_nn.board_encoding import board_to_tensor
from chess_nn.move_encoding import move_to_index
from chess_nn.mcts import MCTS

# Temperature schedule:
# Early in the game (< TEMP_THRESHOLD moves), use temperature=1 for variety.
# Late in the game, use near-deterministic play (temperature→0).
# This mirrors AlphaZero: explore early, exploit late.
TEMP_THRESHOLD = 30


def play_game(mcts: MCTS, max_moves: int = 200) -> list[dict]:
    """
    Play one full game of the model against itself using MCTS.

    Returns a list of position records — one per move played.
    Each record has board_tensor, policy (4672-length array), and turn (chess.WHITE/BLACK).
    The value target is filled in at the end once we know the game result.
    """
    board = chess.Board()
    positions = []

    for move_num in range(max_moves):
        if board.is_game_over():
            break

        temperature = 1.0 if move_num < TEMP_THRESHOLD else 0.1

        # Get MCTS visit distribution — this is the improved policy target
        visit_dist = mcts.get_policy(board, temperature=temperature)

        # Convert visit distribution to a 4672-length array
        policy_target = np.zeros(4672, dtype=np.float32)
        for move, prob in visit_dist.items():
            if move in board.legal_moves:
                idx = move_to_index(move, board)
                policy_target[idx] = prob

        positions.append({
            "board_tensor": board_to_tensor(board),
            "policy": policy_target,
            "turn": board.turn,
        })

        # Pick the move from the distribution
        moves = list(visit_dist.keys())
        probs = np.array([visit_dist[m] for m in moves])
        probs /= probs.sum()  # Re-normalise for safety
        chosen_move = np.random.choice(moves, p=probs)
        board.push(chosen_move)

    # --- Fill in value targets ---
    # Now we know who won. Go back through every position and assign
    # the game result from that player's perspective.
    result = board.result()
    records = []
    for pos in positions:
        if result == "1-0":
            value = 1.0 if pos["turn"] == chess.WHITE else -1.0
        elif result == "0-1":
            value = -1.0 if pos["turn"] == chess.WHITE else 1.0
        else:
            value = 0.0  # Draw

        records.append({
            "board_tensor": pos["board_tensor"],
            "policy": pos["policy"],
            "value": np.float32(value),
        })

    return records, result


def generate_games(model, num_games: int, num_simulations: int = 200,
                   output_dir: str = None, iteration: int = 0,
                   chunk_size: int = RL_CHUNK_SIZE) -> str:
    """
    Generate `num_games` self-play games and save them as a .npz file.

    Games are accumulated in RAM in batches of `chunk_size`, then flushed to temporary
    files on disk. This keeps peak RAM proportional to `chunk_size` × avg_game_length
    rather than `num_games` × avg_game_length.

    Returns the path to the final merged .npz file.
    """
    if output_dir is None:
        output_dir = os.path.join(PROCESSED_DATA_DIR, "self_play")
    os.makedirs(output_dir, exist_ok=True)

    mcts = MCTS(model, num_simulations=num_simulations)
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
    total_positions = 0

    print(f"Generating {num_games} self-play games ({num_simulations} sims/move, "
          f"flushing every {chunk_size})...")

    chunk_paths = []
    all_boards, all_policies, all_values = [], [], []

    for game_idx in tqdm(range(num_games), desc="Self-play games"):
        records, result = play_game(mcts)
        results[result] = results.get(result, 0) + 1

        for r in records:
            all_boards.append(r["board_tensor"])
            all_policies.append(r["policy"])
            all_values.append(r["value"])
            total_positions += 1

        # Flush to disk every chunk_size games to keep RAM bounded
        is_last = (game_idx == num_games - 1)
        if (game_idx + 1) % chunk_size == 0 or is_last:
            chunk_path = os.path.join(
                output_dir, f"_tmp_iter{iteration:03d}_{len(chunk_paths):02d}.npz"
            )
            np.savez_compressed(
                chunk_path,
                boards=np.array(all_boards, dtype=np.float32),
                policies=np.array(all_policies, dtype=np.float32),
                values=np.array(all_values, dtype=np.float32),
            )
            chunk_paths.append(chunk_path)
            all_boards.clear()
            all_policies.clear()
            all_values.clear()
            gc.collect()

    # Merge all chunks into one file for this iteration
    final_path = os.path.join(output_dir, f"selfplay_iter{iteration:03d}.npz")
    if len(chunk_paths) == 1:
        os.rename(chunk_paths[0], final_path)
    else:
        merged = [np.load(p) for p in chunk_paths]
        np.savez_compressed(
            final_path,
            boards=np.concatenate([d["boards"] for d in merged]),
            policies=np.concatenate([d["policies"] for d in merged]),
            values=np.concatenate([d["values"] for d in merged]),
        )
        for p in chunk_paths:
            os.remove(p)

    total = num_games
    print(f"\nResults — White: {results['1-0']}/{total}  "
          f"Black: {results['0-1']}/{total}  "
          f"Draws: {results['1/2-1/2']}/{total}")
    print(f"Positions: {total_positions:,} → {final_path}")
    return final_path
