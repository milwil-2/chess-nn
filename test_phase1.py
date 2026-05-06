"""
Phase 1 + 2 verification: run this to confirm encoding and model are working.
"""

import chess
import numpy as np
import torch

from chess_nn.board_encoding import board_to_tensor
from chess_nn.move_encoding import move_to_index, index_to_move, get_legal_move_indices
from chess_nn.model import ChessNet
from chess_nn.utils import count_parameters, get_device

print("=== Phase 1: Board Encoding ===")
board = chess.Board()  # Starting position
tensor = board_to_tensor(board)
print(f"Board tensor shape: {tensor.shape}")   # Should be (18, 8, 8)
assert tensor.shape == (18, 8, 8)
print(f"White pawns on rank 2: {tensor[0, 1].sum():.0f}")  # Should be 8
assert tensor[0, 1].sum() == 8, "White pawn plane wrong"
print("Board encoding OK")

print("\n=== Phase 1: Move Encoding ===")
legal_moves = list(board.legal_moves)
print(f"Legal moves in starting position: {len(legal_moves)}")  # Should be 20

# Round-trip: encode every legal move, decode it, check it matches
failures = 0
for move in legal_moves:
    idx = move_to_index(move, board)
    decoded = index_to_move(idx, board)
    if move != decoded:
        print(f"  FAIL: {move} → {idx} → {decoded}")
        failures += 1

if failures == 0:
    print(f"All {len(legal_moves)} moves encode/decode correctly")
else:
    print(f"{failures} round-trip failures")

print("\n=== Phase 2: Neural Network ===")
device = get_device()
print(f"Using device: {device}")

model = ChessNet().to(device)
count_parameters(model)

# Forward pass with a batch of 2 random boards
x = torch.randn(2, 18, 8, 8).to(device)
policy, value = model(x)
print(f"Policy output shape: {policy.shape}")  # Should be (2, 4672)
print(f"Value output shape:  {value.shape}")   # Should be (2, 1)
print(f"Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")  # Should be in [-1, 1]
assert policy.shape == (2, 4672)
assert value.shape == (2, 1)
assert value.abs().max().item() <= 1.0

# Test activation extraction for the viz heatmap
board_tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
heatmap = model.get_activations(board_tensor, layer_index=0)
print(f"Activation heatmap shape: {heatmap.shape}")  # Should be (8, 8)
assert heatmap.shape == (8, 8)

print("\nAll checks passed!")
