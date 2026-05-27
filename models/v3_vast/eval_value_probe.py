#!/usr/bin/env python3
"""Run a chess-nn checkpoint on the value-probe set; report per-category MAE.

Usage:
    python eval_value_probe.py [path/to/checkpoint.pt]

Defaults to checkpoints/best_model.pt (relative to this script's directory).

The model's value head returns WDL logits from the CURRENT-PLAYER's POV.
We convert to a scalar via wdl_to_scalar, then flip the sign when it is
black to move so that the comparison is consistently in WHITE POV.
"""

import os
import sys
import json

# Make `chess_nn` and `config` importable when running from this directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import chess
import torch

from chess_nn.model import ChessNet, wdl_to_scalar
from chess_nn.board_encoding import boards_to_tensor
from chess_nn.utils import load_checkpoint


def main():
    default_ckpt = os.path.join(THIS_DIR, "checkpoints", "best_model.pt")
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else default_ckpt

    probe_path = os.path.join(THIS_DIR, "data", "value_probe.json")
    with open(probe_path) as f:
        probes = json.load(f)

    model = ChessNet()
    load_checkpoint(checkpoint, model)
    model.eval()

    per_cat: dict = {}
    rows = []
    for p in probes:
        board = chess.Board(p["fen"])
        tensor = torch.from_numpy(boards_to_tensor([board])).unsqueeze(0).float()
        with torch.no_grad():
            _, value_logits = model(tensor)
        # wdl_to_scalar returns current-player-POV in [-1, +1]
        pred_stm_pov = float(wdl_to_scalar(value_logits))
        # Convert to white POV
        pred_white_pov = pred_stm_pov if board.turn == chess.WHITE else -pred_stm_pov
        err = abs(pred_white_pov - p["true_white_pov"])
        per_cat.setdefault(p["label"], []).append(err)
        rows.append((p["label"], p["fen"], p["true_white_pov"], pred_white_pov, err))

    print(f"Checkpoint: {checkpoint}")
    print(f"Probes:     {len(probes)} positions from {probe_path}")
    print()
    print(f"{'category':<20} {'n':>3} {'mean_abs_err':>13} {'max_err':>8}")
    for cat in ["symmetric", "white-advantage", "black-advantage", "opening", "endgame"]:
        errs = per_cat.get(cat, [])
        if errs:
            print(f"  {cat:<18} {len(errs):>3} {sum(errs)/len(errs):>13.3f} {max(errs):>8.3f}")
    overall = [e for v in per_cat.values() for e in v]
    if overall:
        print(f"  {'OVERALL':<18} {len(overall):>3} {sum(overall)/len(overall):>13.3f} {max(overall):>8.3f}")

    # Per-position dump for debugging
    print()
    print("Per-position predictions:")
    print(f"{'label':<18} {'true_wpov':>10} {'pred_wpov':>10} {'err':>7}  fen")
    for label, fen, truth, pred, err in rows:
        print(f"  {label:<16} {truth:>10.3f} {pred:>10.3f} {err:>7.3f}  {fen}")


if __name__ == "__main__":
    main()
