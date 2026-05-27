#!/usr/bin/env python3
"""Run a chess-nn checkpoint on the value-probe set; report per-category MAE.

Usage:
    python eval_value_probe.py [path/to/checkpoint.pt]

Defaults to checkpoints/best_model.pt (relative to this script's directory).

The model's value head returns WDL logits whose POV depends on how the
network was trained:
  * config.WHITE_POV_VALUE == False (default, legacy): logits are CURRENT
    player's POV. We convert via wdl_to_scalar then flip the sign on
    black-to-move so the probe comparison stays in white POV.
  * config.WHITE_POV_VALUE == True (P2 pilot, GH #6): logits are already
    in white POV — skip the flip.

To eval a pilot checkpoint that was trained with the W-POV labels, set
``CHESSNN_WHITE_POV_VALUE=1`` in the env before invoking this script:

    CHESSNN_WHITE_POV_VALUE=1 python eval_value_probe.py \
        checkpoints/pilot_whitepov_e1.pt
"""

import os
import sys
import json

# Make `chess_nn` and `config` importable when running from this directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import chess
import torch

import config
from chess_nn.model import ChessNet, wdl_to_scalar
from chess_nn.board_encoding import boards_to_tensor
from chess_nn.utils import load_checkpoint


def _white_pov_active() -> bool:
    """Resolve the white-POV flag from env (override) or config (default)."""
    env = os.environ.get("CHESSNN_WHITE_POV_VALUE")
    if env is not None:
        return env not in ("", "0", "false", "False", "no", "off")
    return bool(getattr(config, "WHITE_POV_VALUE", False))


def main():
    default_ckpt = os.path.join(THIS_DIR, "checkpoints", "best_model.pt")
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else default_ckpt

    probe_path = os.path.join(THIS_DIR, "data", "value_probe.json")
    with open(probe_path) as f:
        probes = json.load(f)

    model = ChessNet()
    load_checkpoint(checkpoint, model)
    model.eval()

    white_pov_net = _white_pov_active()
    print(f"Network value POV: {'WHITE (no flip)' if white_pov_net else 'CURRENT-PLAYER (flip on black-to-move)'}")

    per_cat: dict = {}
    rows = []
    for p in probes:
        board = chess.Board(p["fen"])
        tensor = torch.from_numpy(boards_to_tensor([board])).unsqueeze(0).float()
        with torch.no_grad():
            _, value_logits = model(tensor)
        pred_raw = float(wdl_to_scalar(value_logits))
        if white_pov_net:
            # Network output is already in white POV.
            pred_white_pov = pred_raw
        else:
            # Network output is current-player POV → flip on black-to-move.
            pred_white_pov = pred_raw if board.turn == chess.WHITE else -pred_raw
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
