"""
Chess NN test suite.

Run all:          python tests.py
Run one class:    python tests.py TestTimingDiagnostics
Run smoke test:   python tests.py TestSmokeTraining
"""

import bisect
import os
import sys
import time
import tempfile
import unittest
import random
import glob

import chess
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, BATCH_SIZE, POLICY_OUTPUT_SIZE, PROCESSED_DATA_DIR,
    NUM_RESIDUAL_BLOCKS, LEARNING_RATE, WEIGHT_DECAY, VALUE_LOSS_WEIGHT,
    GRADIENT_CLIP,
)
from chess_nn.board_encoding import board_to_tensor
from chess_nn.move_encoding import (
    move_to_index, index_to_move, get_legal_move_indices, policy_to_moves,
)
from chess_nn.model import ChessNet
from chess_nn.utils import get_device, count_parameters, save_checkpoint, load_checkpoint
from chess_nn.tactics import (
    detect_tactics, find_forks, find_pins, find_hanging, find_skewers,
    identify_opening, Tactic,
)
from chess_nn.train import make_policy_targets, get_lr_scale, DIRICHLET_ALPHA

MPS_AVAILABLE = torch.backends.mps.is_available()
CHUNK_PATHS = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, "chunk_*.npz")))
HAS_DATA = len(CHUNK_PATHS) > 0


# ── Helpers ──────────────────────────────────────────────────────────────────

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def _mps_sync():
    if MPS_AVAILABLE:
        torch.mps.synchronize()


def _make_synthetic_batch(batch_size=8, device=torch.device("cpu")):
    """Realistic fake training batch using actual chess positions."""
    boards_list, policy_list, value_list, mask_list = [], [], [], []
    board = chess.Board()
    for _ in range(batch_size):
        # Play a few random moves to get variety
        for _ in range(random.randint(0, 10)):
            if board.is_game_over():
                board.reset()
            board.push(random.choice(list(board.legal_moves)))
        if board.is_game_over():
            board.reset()
        tensor = board_to_tensor(board)
        legal = get_legal_move_indices(board)
        chosen = random.choice(legal)
        mask = np.zeros(POLICY_OUTPUT_SIZE, dtype=bool)
        for idx in legal:
            mask[idx] = True
        boards_list.append(tensor)
        policy_list.append(chosen)
        value_list.append(random.uniform(-1.0, 1.0))
        mask_list.append(mask)
        board.reset()
    return (
        torch.from_numpy(np.array(boards_list, dtype=np.float32)).to(device),
        torch.tensor(policy_list, dtype=torch.long).to(device),
        torch.tensor(value_list, dtype=torch.float32).to(device),
        torch.from_numpy(np.array(mask_list)).to(device),
    )


# ── 1. Board Encoding ─────────────────────────────────────────────────────────

class TestBoardEncoding(unittest.TestCase):

    def setUp(self):
        self.board = chess.Board()
        self.tensor = board_to_tensor(self.board)

    def test_shape(self):
        self.assertEqual(self.tensor.shape, (18, 8, 8))

    def test_dtype(self):
        self.assertEqual(self.tensor.dtype, np.float32)

    def test_white_pawns_plane(self):
        self.assertEqual(self.tensor[0, 1].sum(), 8)
        self.assertEqual(self.tensor[0, 0].sum(), 0)

    def test_white_piece_planes(self):
        # Knights on b1(file=1) and g1(file=6), rank=0
        self.assertEqual(self.tensor[1, 0, 1], 1.0)
        self.assertEqual(self.tensor[1, 0, 6], 1.0)
        # King on e1 (file=4)
        self.assertEqual(self.tensor[5, 0, 4], 1.0)

    def test_castling_planes_all_set(self):
        for plane in range(12, 16):
            self.assertEqual(self.tensor[plane].sum(), 64.0, f"Plane {plane} castling wrong")

    def test_en_passant_empty_at_start(self):
        self.assertEqual(self.tensor[16].sum(), 0.0)

    def test_side_to_move_white(self):
        self.assertEqual(self.tensor[17].sum(), 64.0)

    def test_side_to_move_black_after_e4(self):
        board = chess.Board()
        board.push_uci("e2e4")
        t = board_to_tensor(board)
        self.assertEqual(t[17].sum(), 0.0)

    def test_black_perspective_flip(self):
        board = chess.Board()
        board.push_uci("e2e4")
        t = board_to_tensor(board)
        # Black pawns should now be in plane 0 (flipped: black = current player)
        # Black pawns start on rank 6 → flipped to rank 1
        self.assertEqual(t[0, 1].sum(), 8.0)

    def test_en_passant_encoding(self):
        board = chess.Board()
        board.push_uci("e2e4")
        t = board_to_tensor(board)
        # en passant target is e3 (rank=2, file=4) from white's perspective
        # from black's perspective, flipped: rank = 7-2 = 5
        self.assertEqual(t[16].sum(), 1.0)

    def test_castling_lost_after_king_move(self):
        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("e7e5")
        board.push_uci("e1e2")  # king moves — loses both castling rights for white
        t = board_to_tensor(board)
        # Now black to move; from black's perspective, planes 14-15 = white castling rights
        self.assertEqual(t[14].sum(), 0.0)
        self.assertEqual(t[15].sum(), 0.0)

    def test_empty_board_piece_planes_zero(self):
        board = chess.Board.empty()
        t = board_to_tensor(board)
        self.assertEqual(t[:12].sum(), 0.0)

    def test_deterministic(self):
        t1 = board_to_tensor(self.board)
        t2 = board_to_tensor(self.board)
        self.assertTrue(np.array_equal(t1, t2))

    def test_encoding_speed_1000(self):
        board = chess.Board()
        with Timer() as t:
            for _ in range(1000):
                board_to_tensor(board)
        print(f"\n  [timing] 1000 board encodings: {t.elapsed:.3f}s ({t.elapsed/1000*1000:.2f}ms each)")
        self.assertLess(t.elapsed, 5.0)


# ── 2. Move Encoding ──────────────────────────────────────────────────────────

class TestMoveEncoding(unittest.TestCase):

    def test_starting_position_round_trip(self):
        board = chess.Board()
        for move in board.legal_moves:
            idx = move_to_index(move, board)
            decoded = index_to_move(idx, board)
            self.assertEqual(move, decoded, f"Round-trip failed: {move} → {idx} → {decoded}")

    def test_black_moves_round_trip(self):
        board = chess.Board()
        board.push_uci("e2e4")
        for move in board.legal_moves:
            idx = move_to_index(move, board)
            decoded = index_to_move(idx, board)
            self.assertEqual(move, decoded, f"Black round-trip failed: {move}")

    def test_index_in_range(self):
        board = chess.Board()
        for move in board.legal_moves:
            idx = move_to_index(move, board)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 4672)

    def test_legal_move_indices_count_start(self):
        board = chess.Board()
        indices = get_legal_move_indices(board)
        self.assertEqual(len(indices), 20)

    def test_legal_move_indices_unique(self):
        board = chess.Board()
        indices = get_legal_move_indices(board)
        self.assertEqual(len(indices), len(set(indices)))

    def test_knight_move_plane_range(self):
        # Nf3 from starting position
        board = chess.Board()
        move = chess.Move.from_uci("g1f3")
        idx = move_to_index(move, board)
        plane = idx % 73
        self.assertGreaterEqual(plane, 56)
        self.assertLess(plane, 64)

    def test_underpromotion_plane_range(self):
        # Pawn on a7 promoting to knight
        board = chess.Board.empty()
        board.set_piece_at(chess.A7, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        move = chess.Move(chess.A7, chess.A8, promotion=chess.KNIGHT)
        if move in board.legal_moves:
            idx = move_to_index(move, board)
            plane = idx % 73
            self.assertGreaterEqual(plane, 64)
            self.assertLess(plane, 73)

    def test_policy_to_moves_all_legal(self):
        board = chess.Board()
        policy = np.random.randn(4672).astype(np.float32)
        results = policy_to_moves(policy, board, top_k=10)
        for move, prob in results:
            self.assertIn(move, board.legal_moves)

    def test_policy_to_moves_probabilities_positive(self):
        board = chess.Board()
        policy = np.random.randn(4672).astype(np.float32)
        results = policy_to_moves(policy, board, top_k=5)
        for _, prob in results:
            self.assertGreater(prob, 0.0)

    def test_complex_position_round_trip(self):
        board = chess.Board()
        moves_uci = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3"]
        for uci in moves_uci:
            board.push_uci(uci)
        for move in board.legal_moves:
            idx = move_to_index(move, board)
            decoded = index_to_move(idx, board)
            self.assertEqual(move, decoded, f"Complex position round-trip failed: {move}")


# ── 3. Model ──────────────────────────────────────────────────────────────────

class TestModel(unittest.TestCase):

    def setUp(self):
        self.model = ChessNet()

    def test_output_shapes_cpu(self):
        x = torch.randn(4, 18, 8, 8)
        policy, value = self.model(x)
        self.assertEqual(policy.shape, (4, 4672))
        self.assertEqual(value.shape, (4, 1))

    def test_value_in_range(self):
        x = torch.randn(8, 18, 8, 8)
        _, value = self.model(x)
        self.assertLessEqual(value.abs().max().item(), 1.0)

    def test_batch_size_one_eval(self):
        self.model.eval()
        x = torch.randn(1, 18, 8, 8)
        policy, value = self.model(x)
        self.assertEqual(policy.shape, (1, 4672))
        self.assertTrue(torch.isfinite(value).all())

    def test_residual_block_count(self):
        self.assertEqual(len(self.model.residual_blocks), NUM_RESIDUAL_BLOCKS)

    def test_parameter_count_positive(self):
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertGreater(n, 0)

    def test_get_activations_shape(self):
        board = chess.Board()
        t = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
        heatmap = self.model.get_activations(t, layer_index=0)
        self.assertEqual(heatmap.shape, (8, 8))

    def test_get_activations_all_layers(self):
        board = chess.Board()
        t = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
        for i in range(NUM_RESIDUAL_BLOCKS):
            hm = self.model.get_activations(t, layer_index=i)
            self.assertEqual(hm.shape, (8, 8))

    def test_get_all_activations_keys(self):
        board = chess.Board()
        t = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
        captured = self.model.get_all_activations(t)
        expected_keys = (
            ["input_conv"]
            + [f"res_block_{i}" for i in range(NUM_RESIDUAL_BLOCKS)]
            + ["policy_conv", "value_conv", "policy_logits", "value"]
        )
        for key in expected_keys:
            self.assertIn(key, captured, f"Missing key: {key}")

    def test_gradient_flow_cpu(self):
        model = ChessNet()
        x = torch.randn(4, 18, 8, 8)
        policy, value = model(x)
        loss = policy.mean() + value.mean()
        loss.backward()
        grad = model.input_conv.weight.grad
        self.assertIsNotNone(grad)
        self.assertFalse(torch.all(grad == 0))

    @unittest.skipUnless(MPS_AVAILABLE, "MPS not available")
    def test_output_shapes_mps(self):
        model = ChessNet().to("mps")
        x = torch.randn(4, 18, 8, 8).to("mps")
        policy, value = model(x)
        _mps_sync()
        self.assertEqual(policy.shape, (4, 4672))
        self.assertEqual(value.shape, (4, 1))

    @unittest.skipUnless(MPS_AVAILABLE, "MPS not available")
    def test_gradient_flow_mps(self):
        model = ChessNet().to("mps")
        x = torch.randn(4, 18, 8, 8).to("mps")
        policy, value = model(x)
        loss = policy.mean() + value.mean()
        loss.backward()
        _mps_sync()
        grad = model.input_conv.weight.grad
        self.assertIsNotNone(grad)
        self.assertFalse(torch.all(grad == 0))


# ── 4. Dataset ────────────────────────────────────────────────────────────────

@unittest.skipUnless(HAS_DATA, "No chunk files in data/processed/")
class TestDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from chess_nn.dataset import ChessDataset, make_dataloaders
        cls.dataset = ChessDataset(CHUNK_PATHS[:2])  # Use 2 chunks for speed
        cls.make_dataloaders = make_dataloaders

    def test_length_positive(self):
        self.assertGreater(len(self.dataset), 0)

    def test_getitem_returns_four_tensors(self):
        item = self.dataset[0]
        self.assertEqual(len(item), 4)

    def test_getitem_board_shape(self):
        board, policy, value, mask = self.dataset[0]
        self.assertEqual(board.shape, (18, 8, 8))

    def test_getitem_mask_shape(self):
        board, policy, value, mask = self.dataset[0]
        self.assertEqual(mask.shape, (POLICY_OUTPUT_SIZE,))

    def test_getitem_dtypes(self):
        board, policy, value, mask = self.dataset[0]
        self.assertEqual(board.dtype, torch.float32)
        self.assertEqual(policy.dtype, torch.long)
        self.assertEqual(value.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.bool)

    def test_legal_mask_has_true(self):
        _, _, _, mask = self.dataset[0]
        self.assertTrue(mask.any())

    def test_legal_mask_not_all_true(self):
        _, _, _, mask = self.dataset[0]
        self.assertFalse(mask.all())

    def test_policy_index_in_range(self):
        _, policy, _, _ = self.dataset[0]
        self.assertGreaterEqual(policy.item(), 0)
        self.assertLess(policy.item(), POLICY_OUTPUT_SIZE)

    def test_value_in_range(self):
        _, _, value, _ = self.dataset[0]
        self.assertLessEqual(abs(value.item()), 1.0)

    def test_unpackbits_round_trip(self):
        mask = np.zeros(POLICY_OUTPUT_SIZE, dtype=bool)
        mask[0] = True
        mask[100] = True
        mask[4671] = True
        packed = np.packbits(mask)
        unpacked = np.unpackbits(packed)[:POLICY_OUTPUT_SIZE].astype(bool)
        self.assertTrue(np.array_equal(mask, unpacked))

    def test_result_to_value(self):
        from chess_nn.dataset import result_to_value
        self.assertEqual(result_to_value("1-0", chess.WHITE), 1.0)
        self.assertEqual(result_to_value("1-0", chess.BLACK), -1.0)
        self.assertEqual(result_to_value("0-1", chess.WHITE), -1.0)
        self.assertEqual(result_to_value("0-1", chess.BLACK), 1.0)
        self.assertEqual(result_to_value("1/2-1/2", chess.WHITE), 0.0)
        self.assertEqual(result_to_value("1/2-1/2", chess.BLACK), 0.0)

    def test_getitem_timing_1000(self):
        n = 1000
        indices = [random.randint(0, len(self.dataset) - 1) for _ in range(n)]
        with Timer() as t:
            for i in indices:
                self.dataset[i]
        ms = t.elapsed / n * 1000
        print(f"\n  [timing] 1000 dataset __getitem__ calls: {t.elapsed:.3f}s ({ms:.2f}ms each)")
        self.assertLess(t.elapsed, 30.0)

    def test_first_batch_loading_time(self):
        from chess_nn.dataset import ChessDataset, ChunkBatchSampler
        from torch.utils.data import DataLoader, random_split
        dataset = ChessDataset(CHUNK_PATHS[:1])
        n = len(dataset)
        n_train = int(n * 0.9)
        n_rest = n - n_train
        train_set, _ = random_split(dataset, [n_train, n_rest],
                                     generator=torch.Generator().manual_seed(42))
        sampler = ChunkBatchSampler(train_set, BATCH_SIZE, shuffle=False)
        loader = DataLoader(train_set, batch_sampler=sampler, num_workers=0)
        it = iter(loader)
        with Timer() as t:
            batch = next(it)
        print(f"\n  [timing] First batch load ({BATCH_SIZE} samples): {t.elapsed:.3f}s")
        self.assertIsNotNone(batch)

    def test_unpackbits_speed(self):
        packed = np.packbits(np.zeros(POLICY_OUTPUT_SIZE, dtype=bool))
        packed[0] = 255
        n = 1000
        with Timer() as t:
            for _ in range(n):
                np.unpackbits(packed)[:POLICY_OUTPUT_SIZE].astype(bool)
        ms = t.elapsed / n * 1000
        print(f"\n  [timing] np.unpackbits ×{n}: {t.elapsed:.3f}s ({ms:.3f}ms each)")


# ── 5. Training ───────────────────────────────────────────────────────────────

class TestTraining(unittest.TestCase):

    def test_get_lr_scale_warmup(self):
        self.assertAlmostEqual(get_lr_scale(0), 0.0)
        self.assertAlmostEqual(get_lr_scale(500), 0.5)
        self.assertAlmostEqual(get_lr_scale(1000), 1.0)
        self.assertAlmostEqual(get_lr_scale(2000), 1.0)

    def test_make_policy_targets_shape(self):
        boards, policy_targets, value_targets, legal_masks = _make_synthetic_batch(8)
        soft = make_policy_targets(policy_targets, legal_masks, add_noise=False)
        self.assertEqual(soft.shape, (8, POLICY_OUTPUT_SIZE))

    def test_make_policy_targets_sums_to_one(self):
        boards, policy_targets, value_targets, legal_masks = _make_synthetic_batch(8)
        soft = make_policy_targets(policy_targets, legal_masks, add_noise=False)
        row_sums = soft.sum(dim=1)
        for s in row_sums:
            self.assertAlmostEqual(s.item(), 1.0, places=4)

    def test_make_policy_targets_with_noise_sums_to_one(self):
        boards, policy_targets, value_targets, legal_masks = _make_synthetic_batch(8)
        soft = make_policy_targets(policy_targets, legal_masks, add_noise=True)
        row_sums = soft.sum(dim=1)
        for s in row_sums:
            self.assertAlmostEqual(s.item(), 1.0, places=4)

    def test_make_policy_targets_zero_on_illegal(self):
        boards, policy_targets, value_targets, legal_masks = _make_synthetic_batch(8)
        soft = make_policy_targets(policy_targets, legal_masks, add_noise=False)
        # Illegal positions must have zero probability
        illegal = ~legal_masks
        self.assertTrue((soft[illegal] == 0.0).all())

    def test_masked_logits_illegal_minus_inf(self):
        boards, policy_targets, value_targets, legal_masks = _make_synthetic_batch(4)
        model = ChessNet()
        with torch.no_grad():
            policy_logits, _ = model(boards)
        masked = policy_logits.masked_fill(~legal_masks, float("-inf"))
        illegal = ~legal_masks
        self.assertTrue(torch.all(masked[illegal] == float("-inf")))

    def test_kl_div_loss_finite(self):
        boards, policy_targets, value_targets, legal_masks = _make_synthetic_batch(8)
        model = ChessNet()
        with torch.no_grad():
            policy_logits, _ = model(boards)
        masked = policy_logits.masked_fill(~legal_masks, float("-inf"))
        soft = make_policy_targets(policy_targets, legal_masks, add_noise=False)
        log_probs = F.log_softmax(masked, dim=1).masked_fill(~legal_masks, 0.0)
        loss = -(soft * log_probs).sum(dim=1).mean()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_single_training_step_cpu(self):
        model = ChessNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        boards, policy_targets, value_targets, legal_masks = _make_synthetic_batch(8)

        model.train()
        policy_logits, value_pred = model(boards)
        masked = policy_logits.masked_fill(~legal_masks, float("-inf"))
        soft = make_policy_targets(policy_targets, legal_masks, add_noise=False)
        log_probs = F.log_softmax(masked, dim=1).masked_fill(~legal_masks, 0.0)
        policy_loss = -(soft * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(value_pred.squeeze(1), value_targets)
        loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.input_conv.weight.grad)

    @unittest.skipUnless(MPS_AVAILABLE, "MPS not available")
    def test_single_training_step_mps(self):
        device = torch.device("mps")
        model = ChessNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        boards, policy_targets, value_targets, legal_masks = _make_synthetic_batch(8, device=device)

        model.train()
        with Timer() as t:
            policy_logits, value_pred = model(boards)
            masked = policy_logits.masked_fill(~legal_masks, float("-inf"))
            soft = make_policy_targets(policy_targets, legal_masks, add_noise=False)
            log_probs = F.log_softmax(masked, dim=1).masked_fill(~legal_masks, 0.0)
            policy_loss = -(soft * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred.squeeze(1), value_targets)
            loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            _mps_sync()

        print(f"\n  [timing] Single MPS training step (batch=8): {t.elapsed:.3f}s")
        self.assertTrue(torch.isfinite(loss))

    def test_loss_decreases_over_steps(self):
        model = ChessNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        boards, policy_targets, value_targets, legal_masks = _make_synthetic_batch(16)
        model.train()
        losses = []
        for _ in range(15):
            policy_logits, value_pred = model(boards)
            masked = policy_logits.masked_fill(~legal_masks, float("-inf"))
            soft = make_policy_targets(policy_targets, legal_masks, add_noise=False)
            log_probs = F.log_softmax(masked, dim=1).masked_fill(~legal_masks, 0.0)
            policy_loss = -(soft * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred.squeeze(1), value_targets)
            loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0], "Loss did not decrease over 15 steps")


# ── 6. Smoke Training (real data + MPS) ───────────────────────────────────────

@unittest.skipUnless(HAS_DATA and MPS_AVAILABLE, "Needs real data + MPS")
class TestSmokeTraining(unittest.TestCase):
    """
    Runs real batches through the full MPS training pipeline.
    Times every sub-step to pinpoint where the first-batch hang occurs.
    """

    @classmethod
    def setUpClass(cls):
        from chess_nn.dataset import ChessDataset, ChunkBatchSampler
        from torch.utils.data import DataLoader, random_split
        dataset = ChessDataset(CHUNK_PATHS[:2])
        n = len(dataset)
        n_train = int(n * 0.9)
        train_set, _ = random_split(dataset, [n_train, n - n_train],
                                     generator=torch.Generator().manual_seed(42))
        sampler = ChunkBatchSampler(train_set, BATCH_SIZE, shuffle=False)
        cls.loader = DataLoader(train_set, batch_sampler=sampler, num_workers=0)
        cls.model = ChessNet().to("mps")
        cls.optimizer = torch.optim.Adam(cls.model.parameters(), lr=LEARNING_RATE,
                                          weight_decay=WEIGHT_DECAY)
        cls.model.train()

    def _run_one_batch(self, it, label):
        device = torch.device("mps")
        timings = {}

        with Timer() as t:
            boards, policy_targets, value_targets, legal_masks = next(it)
        timings["data_load"] = t.elapsed

        with Timer() as t:
            boards = boards.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            legal_masks = legal_masks.to(device)
            _mps_sync()
        timings["to_device"] = t.elapsed

        with Timer() as t:
            policy_logits, value_pred = self.model(boards)
            _mps_sync()
        timings["forward"] = t.elapsed

        with Timer() as t:
            masked = policy_logits.masked_fill(~legal_masks, float("-inf"))
            soft = make_policy_targets(policy_targets, legal_masks, add_noise=True)
            log_probs = F.log_softmax(masked, dim=1).masked_fill(~legal_masks, 0.0)
            policy_loss = -(soft * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred.squeeze(1), value_targets)
            loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss
        timings["loss"] = t.elapsed

        with Timer() as t:
            self.optimizer.zero_grad()
            loss.backward()
            _mps_sync()
        timings["backward"] = t.elapsed

        with Timer() as t:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
            self.optimizer.step()
            _mps_sync()
        timings["optimizer"] = t.elapsed

        total = sum(timings.values())
        print(f"\n  [{label}] total={total:.2f}s | "
              + " | ".join(f"{k}={v:.2f}s" for k, v in timings.items()))
        return loss.item(), timings

    def test_smoke_two_batches_timing(self):
        it = iter(self.loader)
        loss1, t1 = self._run_one_batch(it, "batch1")
        loss2, t2 = self._run_one_batch(it, "batch2")
        self.assertTrue(np.isfinite(loss1), "Batch 1 loss is not finite")
        self.assertTrue(np.isfinite(loss2), "Batch 2 loss is not finite")
        print(f"\n  Batch2 forward speedup: {t1['forward']/t2['forward']:.1f}×")


# ── 7. Timing Diagnostics ─────────────────────────────────────────────────────

class TestTimingDiagnostics(unittest.TestCase):
    """Isolate each MPS operation to identify the first-batch bottleneck."""

    def test_model_creation_time(self):
        with Timer() as t:
            model = ChessNet()
        print(f"\n  [timing] ChessNet() creation: {t.elapsed:.3f}s")

    @unittest.skipUnless(MPS_AVAILABLE, "MPS not available")
    def test_model_to_mps_time(self):
        model = ChessNet()
        with Timer() as t:
            model = model.to("mps")
            _mps_sync()
        print(f"\n  [timing] model.to('mps'): {t.elapsed:.3f}s")

    @unittest.skipUnless(MPS_AVAILABLE, "MPS not available")
    def test_first_vs_second_forward_mps(self):
        model = ChessNet().to("mps")
        x = torch.randn(4, 18, 8, 8).to("mps")

        with Timer() as t1:
            _ = model(x)
            _mps_sync()
        with Timer() as t2:
            _ = model(x)
            _mps_sync()
        print(f"\n  [timing] 1st forward: {t1.elapsed:.3f}s | 2nd forward: {t2.elapsed:.3f}s "
              f"| speedup: {t1.elapsed/max(t2.elapsed,1e-6):.1f}×")

    @unittest.skipUnless(MPS_AVAILABLE, "MPS not available")
    def test_first_vs_second_backward_mps(self):
        model = ChessNet().to("mps")
        x = torch.randn(4, 18, 8, 8).to("mps")

        p, v = model(x)
        with Timer() as t1:
            (p.mean() + v.mean()).backward()
            _mps_sync()

        model.zero_grad()
        p, v = model(x)
        with Timer() as t2:
            (p.mean() + v.mean()).backward()
            _mps_sync()

        print(f"\n  [timing] 1st backward: {t1.elapsed:.3f}s | 2nd backward: {t2.elapsed:.3f}s "
              f"| speedup: {t1.elapsed/max(t2.elapsed,1e-6):.1f}×")

    @unittest.skipUnless(MPS_AVAILABLE, "MPS not available")
    def test_mps_synchronize_cost(self):
        model = ChessNet().to("mps")
        x = torch.randn(BATCH_SIZE, 18, 8, 8).to("mps")
        _ = model(x)  # warm up
        with Timer() as t:
            torch.mps.synchronize()
        print(f"\n  [timing] torch.mps.synchronize() after warm forward: {t.elapsed*1000:.1f}ms")

    @unittest.skipUnless(MPS_AVAILABLE, "MPS not available")
    def test_tensor_to_mps_time(self):
        x = torch.randn(BATCH_SIZE, 18, 8, 8)
        with Timer() as t:
            x_mps = x.to("mps")
            _mps_sync()
        print(f"\n  [timing] ({BATCH_SIZE}, 18, 8, 8) tensor to MPS: {t.elapsed*1000:.1f}ms")

    def test_dirichlet_sampling_time(self):
        from chess_nn.train import DIRICHLET_ALPHA
        boards, policy_targets, value_targets, legal_masks = _make_synthetic_batch(BATCH_SIZE)
        with Timer() as t:
            soft = make_policy_targets(policy_targets, legal_masks, add_noise=True)
        print(f"\n  [timing] make_policy_targets with Dirichlet (batch={BATCH_SIZE}): {t.elapsed*1000:.1f}ms")

    def test_unpackbits_at_batch_scale(self):
        packed = np.packbits(np.zeros(POLICY_OUTPUT_SIZE, dtype=bool))
        with Timer() as t:
            for _ in range(BATCH_SIZE):
                np.unpackbits(packed)[:POLICY_OUTPUT_SIZE].astype(bool)
        print(f"\n  [timing] unpackbits ×{BATCH_SIZE} (one full batch): {t.elapsed*1000:.1f}ms")

    @unittest.skipUnless(HAS_DATA and MPS_AVAILABLE, "Needs data + MPS")
    def test_five_batch_timing_comparison(self):
        from chess_nn.dataset import ChessDataset, ChunkBatchSampler
        from torch.utils.data import DataLoader, random_split
        dataset = ChessDataset(CHUNK_PATHS[:1])
        n = len(dataset)
        n_train = int(n * 0.9)
        train_set, _ = random_split(dataset, [n_train, n - n_train],
                                     generator=torch.Generator().manual_seed(42))
        sampler = ChunkBatchSampler(train_set, BATCH_SIZE, shuffle=False)
        loader = DataLoader(train_set, batch_sampler=sampler, num_workers=0)
        model = ChessNet().to("mps")
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        model.train()
        device = torch.device("mps")
        print()
        for i, (boards, policy_targets, value_targets, legal_masks) in enumerate(loader):
            if i >= 5:
                break
            with Timer() as t:
                boards = boards.to(device)
                policy_targets = policy_targets.to(device)
                value_targets = value_targets.to(device)
                legal_masks = legal_masks.to(device)
                policy_logits, value_pred = model(boards)
                masked = policy_logits.masked_fill(~legal_masks, float("-inf"))
                soft = make_policy_targets(policy_targets, legal_masks, add_noise=True)
                log_probs = F.log_softmax(masked, dim=1).masked_fill(~legal_masks, 0.0)
                policy_loss = -(soft * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(value_pred.squeeze(1), value_targets)
                loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _mps_sync()
            print(f"  [timing] Batch {i+1}: {t.elapsed:.2f}s  loss={loss.item():.4f}")


# ── 8. MCTS ───────────────────────────────────────────────────────────────────

class TestMCTS(unittest.TestCase):

    def setUp(self):
        from chess_nn.mcts import Node, MCTS
        self.Node = Node
        self.MCTS = MCTS

    def test_node_initial_state(self):
        node = self.Node()
        self.assertEqual(node.N, 0)
        self.assertAlmostEqual(node.W, 0.0)
        self.assertAlmostEqual(node.Q, 0.0)
        self.assertEqual(len(node.children), 0)
        self.assertFalse(node.is_expanded)

    def test_node_q_value(self):
        node = self.Node()
        node.N = 10
        node.W = 5.0
        self.assertAlmostEqual(node.Q, 0.5)

    def test_node_q_zero_when_unvisited(self):
        node = self.Node()
        self.assertAlmostEqual(node.Q, 0.0)

    def test_ucb_prefers_unvisited(self):
        node_new = self.Node(prior=0.5)
        node_visited = self.Node(prior=0.1)
        node_visited.N = 100
        node_visited.W = 50.0
        parent_visits = 200
        self.assertGreater(node_new.ucb_score(parent_visits),
                           node_visited.ucb_score(parent_visits))

    def test_best_child_selects_highest_ucb(self):
        root = self.Node(prior=1.0)
        root.N = 10
        for i, (p, n, w) in enumerate([(0.5, 1, 0.5), (0.1, 5, 2.0), (0.9, 0, 0.0)]):
            child = self.Node(prior=p)
            child.N = n
            child.W = w
            root.children[chess.Move.from_uci(f"a{i+1}a{i+2}")] = child
        _, best = root.best_child()
        scores = [(m, c.ucb_score(root.N)) for m, c in root.children.items()]
        _, top_score = max(scores, key=lambda x: x[1])
        self.assertAlmostEqual(best.ucb_score(root.N), top_score)

    def test_most_visited_child(self):
        root = self.Node(prior=1.0)
        root.N = 10
        for i, n in enumerate([3, 10, 1]):
            child = self.Node(prior=0.3)
            child.N = n
            root.children[chess.Move.from_uci(f"a{i+1}a{i+2}")] = child
        _, most = root.most_visited_child()
        self.assertEqual(most.N, 10)

    def test_visit_distribution_temperature_1(self):
        root = self.Node(prior=1.0)
        root.N = 10
        visits = [3, 6, 1]
        for i, n in enumerate(visits):
            child = self.Node(prior=0.3)
            child.N = n
            root.children[chess.Move.from_uci(f"a{i+1}a{i+2}")] = child
        dist = root.visit_distribution(temperature=1.0)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=5)

    def test_visit_distribution_temperature_0(self):
        root = self.Node(prior=1.0)
        root.N = 10
        visits = [3, 10, 1]
        for i, n in enumerate(visits):
            child = self.Node(prior=0.3)
            child.N = n
            root.children[chess.Move.from_uci(f"a{i+1}a{i+2}")] = child
        dist = root.visit_distribution(temperature=0)
        probs = list(dist.values())
        self.assertEqual(probs.count(1.0), 1)
        self.assertEqual(probs.count(0.0), 2)

    def test_terminal_value(self):
        mcts = self.MCTS(ChessNet(), num_simulations=2)
        self.assertAlmostEqual(mcts._terminal_value("1-0", chess.WHITE), 1.0)
        self.assertAlmostEqual(mcts._terminal_value("1-0", chess.BLACK), -1.0)
        self.assertAlmostEqual(mcts._terminal_value("0-1", chess.WHITE), -1.0)
        self.assertAlmostEqual(mcts._terminal_value("0-1", chess.BLACK), 1.0)
        self.assertAlmostEqual(mcts._terminal_value("1/2-1/2", chess.WHITE), 0.0)

    def test_expand_populates_children(self):
        from chess_nn.mcts import Node
        mcts = self.MCTS(ChessNet(), num_simulations=2)
        root = Node(prior=1.0)
        board = chess.Board()
        mcts._expand(root, board)
        self.assertTrue(root.is_expanded)
        self.assertGreater(len(root.children), 0)
        for move in root.children:
            self.assertIn(move, board.legal_moves)

    def test_mcts_search_returns_legal_move(self):
        model = ChessNet()
        model.eval()
        mcts = self.MCTS(model, num_simulations=5)
        board = chess.Board()
        move = mcts.search(board, temperature=1.0)
        self.assertIn(move, board.legal_moves)

    def test_backup_sign_flip(self):
        """Verify W accumulates with alternating signs at each level."""
        from chess_nn.mcts import Node
        node0 = Node(prior=1.0)
        node1 = Node(prior=0.5)
        node2 = Node(prior=0.3)
        path = [node0, node1, node2]
        value = 1.0
        for i, n in enumerate(reversed(path)):
            n.N += 1
            n.W += value if i % 2 == 0 else -value
        self.assertAlmostEqual(node2.W, 1.0)
        self.assertAlmostEqual(node1.W, -1.0)
        self.assertAlmostEqual(node0.W, 1.0)


# ── 9. Tactics ────────────────────────────────────────────────────────────────

class TestTactics(unittest.TestCase):

    def test_detect_tactics_returns_list(self):
        result = detect_tactics(chess.Board())
        self.assertIsInstance(result, list)

    def test_starting_position_no_forks_pins_hanging(self):
        board = chess.Board()
        tactics = detect_tactics(board)
        names = [t.name for t in tactics]
        self.assertNotIn("Fork", names)
        self.assertNotIn("Pin", names)
        self.assertNotIn("Hanging Piece", names)

    def test_identify_opening_e4(self):
        board = chess.Board()
        board.push_uci("e2e4")
        result = identify_opening(board)
        self.assertIsNotNone(result)
        self.assertIn("Pawn", result.description)

    def test_identify_opening_sicilian(self):
        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("c7c5")
        result = identify_opening(board)
        self.assertIsNotNone(result)
        self.assertIn("Sicilian", result.description)

    def test_identify_opening_none_on_empty_stack(self):
        board = chess.Board()
        result = identify_opening(board)
        self.assertIsNone(result)

    def test_tactic_dataclass_fields(self):
        board = chess.Board()
        board.push_uci("e2e4")
        result = identify_opening(board)
        if result:
            self.assertTrue(hasattr(result, "name"))
            self.assertTrue(hasattr(result, "squares"))
            self.assertTrue(hasattr(result, "description"))
            self.assertTrue(hasattr(result, "color"))

    def test_find_hanging_detects_undefended_piece(self):
        # FEN: White queen on d5, not defended, attacked by black knight on f6
        board = chess.Board("8/8/5n2/3Q4/8/8/8/4K1k1 w - - 0 1")
        tactics = find_hanging(board)
        hanging_squares = [t.squares[0] for t in tactics]
        # Queen on d5 should be hanging (attacked by Nf6, not defended)
        self.assertIn(chess.D5, hanging_squares)

    def test_find_forks_detects_knight_fork(self):
        # Knight on e5 attacks both king on g6 and rook on c6
        board = chess.Board("8/8/2r3k1/4N3/8/8/8/4K3 w - - 0 1")
        tactics = find_forks(board)
        self.assertGreater(len(tactics), 0)

    def test_find_pins_detects_absolute_pin(self):
        # Bishop on b2 pins knight on d4 to king on f6
        board = chess.Board("8/8/5k2/8/3n4/8/1B6/4K3 b - - 0 1")
        tactics = find_pins(board)
        self.assertGreater(len(tactics), 0)


# ── 10. Utils ─────────────────────────────────────────────────────────────────

class TestUtils(unittest.TestCase):

    def test_get_device_returns_device(self):
        device = get_device()
        self.assertIsInstance(device, torch.device)

    @unittest.skipUnless(MPS_AVAILABLE, "MPS not available")
    def test_get_device_is_mps_on_apple_silicon(self):
        device = get_device()
        self.assertEqual(device.type, "mps")

    def test_count_parameters_positive(self):
        model = ChessNet()
        n = count_parameters(model)
        self.assertGreater(n, 0)

    def test_save_load_checkpoint_round_trip(self):
        model = ChessNet()
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name
        try:
            # Save
            data = {
                "epoch": 3,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": 0.42,
            }
            torch.save(data, tmp_path)
            # Load
            model2 = ChessNet()
            checkpoint = torch.load(tmp_path, map_location="cpu", weights_only=False)
            model2.load_state_dict(checkpoint["model_state"])
            # Verify weights match
            for (k1, v1), (k2, v2) in zip(
                model.state_dict().items(), model2.state_dict().items()
            ):
                self.assertTrue(torch.equal(v1, v2), f"Mismatch in {k1}")
            self.assertEqual(checkpoint["epoch"], 3)
        finally:
            os.unlink(tmp_path)


# ── 11. Evaluate ──────────────────────────────────────────────────────────────

class TestEvaluate(unittest.TestCase):

    def setUp(self):
        from chess_nn.evaluate import select_move
        self.select_move = select_move
        self.model = ChessNet()
        self.model.eval()

    def test_select_move_returns_legal(self):
        board = chess.Board()
        move = self.select_move(self.model, board, temperature=1.0)
        self.assertIn(move, board.legal_moves)

    def test_select_move_greedy_is_deterministic(self):
        board = chess.Board()
        move1 = self.select_move(self.model, board, temperature=0)
        move2 = self.select_move(self.model, board, temperature=0)
        self.assertEqual(move1, move2)

    def test_select_move_midgame_legal(self):
        board = chess.Board()
        for uci in ["e2e4", "e7e5", "g1f3", "b8c6"]:
            board.push_uci(uci)
        move = self.select_move(self.model, board, temperature=1.0)
        self.assertIn(move, board.legal_moves)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Allow: python tests.py TestBoardEncoding  (runs just that class)
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        suite = unittest.TestLoader().loadTestsFromName(sys.argv[1], module=sys.modules[__name__])
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    else:
        unittest.main(verbosity=2)
