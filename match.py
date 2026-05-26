"""
Head-to-head match between two model variants.

Each model runs as an isolated subprocess via its own run.py engine command,
so models with completely different architectures work without conflict.

Usage:
  python match.py models/v1_baseline models/v2_wider_heads
  python match.py models/v1_baseline models/v2_wider_heads --games 40
  python match.py models/v1_baseline models/v2_wider_heads --sims 100
  python match.py models/v1_baseline models/v2_wider_heads --fast          # policy-only, no MCTS
  python match.py models/v1_baseline models/v2_wider_heads --checkpoint rl_best_model.pt
"""

import argparse
import os
import random
import subprocess
import sys

import chess


def start_engine(model_dir: str, checkpoint: str, sims: int, fast: bool) -> subprocess.Popen:
    ckpt_path = os.path.join(model_dir, "checkpoints", checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        os.path.join(model_dir, "run.py"),
        "engine",
        "--checkpoint", ckpt_path,
        "--sims", str(sims),
    ]
    if fast:
        cmd.append("--fast")

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def get_move(engine: subprocess.Popen, board: chess.Board) -> chess.Move:
    engine.stdin.write(f"position fen {board.fen()}\n")
    engine.stdin.write("go\n")
    engine.stdin.flush()
    line = engine.stdout.readline().strip()
    return chess.Move.from_uci(line.split()[1])


def run_match(dir_a: str, dir_b: str, games: int, sims: int, fast: bool, checkpoint: str):
    name_a = os.path.basename(dir_a.rstrip("/"))
    name_b = os.path.basename(dir_b.rstrip("/"))
    mode = "policy-only" if fast else f"MCTS {sims} sims"
    print(f"\n{name_a}  vs  {name_b}  —  {games} games  ({mode})\n")

    engine_a = start_engine(dir_a, checkpoint, sims, fast)
    engine_b = start_engine(dir_b, checkpoint, sims, fast)

    wins_a = draws = wins_b = 0

    for game_idx in range(games):
        board = chess.Board()
        a_is_white = (game_idx % 2 == 0)

        while not board.is_game_over():
            try:
                if (board.turn == chess.WHITE) == a_is_white:
                    move = get_move(engine_a, board)
                else:
                    move = get_move(engine_b, board)
            except Exception:
                move = random.choice(list(board.legal_moves))

            if move in board.legal_moves:
                board.push(move)
            else:
                board.push(random.choice(list(board.legal_moves)))

        result = board.result()
        if result == "1-0":
            if a_is_white:
                wins_a += 1
            else:
                wins_b += 1
        elif result == "0-1":
            if a_is_white:
                wins_b += 1
            else:
                wins_a += 1
        else:
            draws += 1

        side = f"{name_a}=W" if a_is_white else f"{name_b}=W"
        print(
            f"  game {game_idx + 1:>2}/{games}  {side:<24}  {result:<7}"
            f"  {name_a} {wins_a} / {draws} draws / {wins_b} {name_b}",
            flush=True,
        )

    for engine in (engine_a, engine_b):
        try:
            engine.stdin.write("quit\n")
            engine.stdin.flush()
            engine.wait(timeout=5)
        except Exception:
            engine.kill()

    total = wins_a + wins_b + draws
    print(f"\n{'='*52}")
    print(f"  {name_a:<28} {wins_a:>3} wins  ({wins_a / total * 100:.0f}%)")
    print(f"  {name_b:<28} {wins_b:>3} wins  ({wins_b / total * 100:.0f}%)")
    print(f"  Draws                          {draws:>3}")
    print(f"{'='*52}\n")

    return {"wins_a": wins_a, "draws": draws, "wins_b": wins_b}


def main():
    parser = argparse.ArgumentParser(
        description="Head-to-head match between two model variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model_a", help="Path to first model directory")
    parser.add_argument("model_b", help="Path to second model directory")
    parser.add_argument("--games", type=int, default=20, help="Number of games (default: 20)")
    parser.add_argument("--sims", type=int, default=50,
                        help="MCTS simulations per move (default: 50)")
    parser.add_argument("--fast", action="store_true",
                        help="Policy-only move selection — no MCTS, much faster")
    parser.add_argument("--checkpoint", default="best_model.pt",
                        help="Checkpoint filename to load from each model's checkpoints/ dir")

    args = parser.parse_args()

    for path in (args.model_a, args.model_b):
        if not os.path.isdir(path):
            print(f"Not a directory: {path}")
            sys.exit(1)

    run_match(
        dir_a=args.model_a,
        dir_b=args.model_b,
        games=args.games,
        sims=args.sims,
        fast=args.fast,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
