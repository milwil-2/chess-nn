"""
Pipeline entry point — run any phase of the chess NN project from one place.

Usage:
  python run.py status                       # Show what's been trained so far
  python run.py supervised                   # Phase 4: train on Lichess games
  python run.py rl                           # Phase 7: self-play RL loop (default settings)
  python run.py rl --iterations 3            # RL with only 3 iterations
  python run.py rl --games 10 --sims 100     # Fewer games + simulations (faster, less RAM)
  python run.py rl --checkpoint rl_best_model.pt  # Resume from a specific checkpoint
  python run.py selfplay                     # Generate self-play data only (no training)
  python run.py viz                          # Launch Pygame visualizer
  python run.py viz --web                    # Launch web visualizer (WebSockets)

Tweak hyperparameters in config.py — no need to edit the training code.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── helpers ──────────────────────────────────────────────────────────────────

def _selfplay_dir():
    from config import PROCESSED_DATA_DIR
    return os.path.join(PROCESSED_DATA_DIR, "self_play")


def _checkpoint_path(name: str) -> str:
    from config import CHECKPOINT_DIR
    return os.path.join(CHECKPOINT_DIR, name)


def _require_checkpoint(name: str) -> str:
    path = _checkpoint_path(name)
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        print("Run 'python run.py supervised' first to train a base model.")
        sys.exit(1)
    return path


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_status(_args):
    """Print a summary of training progress."""
    from config import CHECKPOINT_DIR, PROCESSED_DATA_DIR

    print("=== Chess NN — Training Status ===\n")

    # Checkpoints
    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt")))
    if ckpts:
        print(f"Checkpoints ({len(ckpts)}):")
        for p in ckpts:
            size_mb = os.path.getsize(p) / 1e6
            print(f"  {os.path.basename(p):<35}  {size_mb:.1f} MB")
    else:
        print("Checkpoints: none — run 'python run.py supervised' first")

    print()

    # Supervised data
    chunks = glob.glob(os.path.join(PROCESSED_DATA_DIR, "chunk_*.npz"))
    print(f"Supervised data: {len(chunks)} chunk file(s) in {PROCESSED_DATA_DIR}")

    # Self-play data
    sp_files = sorted(glob.glob(os.path.join(_selfplay_dir(), "selfplay_iter*.npz")))
    if sp_files:
        print(f"\nSelf-play iterations: {len(sp_files)}")
        for p in sp_files:
            size_mb = os.path.getsize(p) / 1e6
            print(f"  {os.path.basename(p)}  {size_mb:.1f} MB")
    else:
        print("\nSelf-play data: none — run 'python run.py rl' to generate")


def cmd_supervised(_args):
    """Train on pre-processed Lichess PGN data."""
    from config import PROCESSED_DATA_DIR

    chunk_paths = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, "chunk_*.npz")))
    if not chunk_paths:
        print("No processed data found.")
        print("Download data first: python data/download_data.py")
        print("Then process it:     python chess_nn/dataset.py")
        sys.exit(1)

    print(f"Found {len(chunk_paths)} data chunk(s). Starting supervised training...\n")
    from chess_nn.train import train
    train(chunk_paths)


def cmd_rl(args):
    """Run the RL self-play loop."""
    from chess_nn.train_rl import run_rl_loop
    run_rl_loop(
        num_iterations=args.iterations,
        start_checkpoint=args.checkpoint,
    )


def cmd_selfplay(args):
    """Generate self-play games without running the full training loop."""
    import torch
    from chess_nn.model import ChessNet
    from chess_nn.utils import load_checkpoint
    from chess_nn.self_play import generate_games

    model = ChessNet()
    ckpt = _require_checkpoint(args.checkpoint)
    load_checkpoint(ckpt, model)
    model.eval()

    # Determine next iteration index from existing files
    existing = glob.glob(os.path.join(_selfplay_dir(), "selfplay_iter*.npz"))
    iteration = len(existing)

    generate_games(
        model,
        num_games=args.games,
        num_simulations=args.sims,
        output_dir=_selfplay_dir(),
        iteration=iteration,
    )


def cmd_viz(args):
    """Launch the visualizer."""
    if args.web:
        import subprocess
        print("Starting web visualizer at http://localhost:8765")
        subprocess.run([sys.executable, "viz/web_server.py"])
    else:
        import subprocess
        subprocess.run([sys.executable, "viz/app.py"])


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chess NN pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # status
    sub.add_parser("status", help="Show training progress and saved files")

    # supervised
    sub.add_parser("supervised", help="Train on Lichess games (Phase 4)")

    # rl
    rl_p = sub.add_parser("rl", help="Self-play RL loop (Phase 7)")
    rl_p.add_argument("--iterations", type=int, default=None,
                      help="Number of RL iterations (default: from config.py)")
    rl_p.add_argument("--games", type=int, default=None,
                      help="Override RL_GAMES_PER_ITER from config.py")
    rl_p.add_argument("--sims", type=int, default=None,
                      help="Override RL_SIMULATIONS from config.py")
    rl_p.add_argument("--checkpoint", default="best_model.pt",
                      help="Starting checkpoint filename (default: best_model.pt)")

    # selfplay
    sp_p = sub.add_parser("selfplay", help="Generate self-play data (no training)")
    sp_p.add_argument("--games", type=int, default=25)
    sp_p.add_argument("--sims", type=int, default=200)
    sp_p.add_argument("--checkpoint", default="best_model.pt")

    # viz
    viz_p = sub.add_parser("viz", help="Launch Pygame or web visualizer")
    viz_p.add_argument("--web", action="store_true", help="Use WebSocket web visualizer")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Apply CLI overrides to config before dispatch
    if args.command == "rl":
        import config
        if args.games is not None:
            config.RL_GAMES_PER_ITER = args.games
        if args.sims is not None:
            config.RL_SIMULATIONS = args.sims
        if args.iterations is None:
            args.iterations = 10

    dispatch = {
        "status":     cmd_status,
        "supervised": cmd_supervised,
        "rl":         cmd_rl,
        "selfplay":   cmd_selfplay,
        "viz":        cmd_viz,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
