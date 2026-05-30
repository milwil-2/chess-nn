"""Export the trained ChessNet (v3) to ONNX for in-browser inference and emit
parity fixtures that the TypeScript reimplementation is validated against.

Truth values (top moves, WDL, plane sums) come from torch; onnxruntime is used
only for optional int8 dynamic quantization and a sanity re-check.

Run with the torch venv, from the v3_vast dir so config.py / chess_nn resolve:

    cd models/v3_vast
    /Users/milan/Desktop/projects/chess-nn/models/v1_history8/.venv/bin/python \
        /Users/milan/Desktop/projects/chess-nn/web/scripts/export_onnx.py
"""

import datetime
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
V3_DIR = os.path.join(REPO_ROOT, "models", "v3_vast")
OUT_DIR = os.path.join(REPO_ROOT, "web", "public", "model")
CHECKPOINT = os.path.join(V3_DIR, "checkpoints", "best_model.pt")

ONNX_FP32 = os.path.join(OUT_DIR, "chessnet-v3-fp32.onnx")
ONNX_FINAL = os.path.join(OUT_DIR, "chessnet-v3.onnx")
METADATA = os.path.join(OUT_DIR, "metadata.json")
FIXTURES = os.path.join(OUT_DIR, "parity_fixtures.json")

# config.py uses bare `from config import ...`; chess_nn.model imports it the
# same way, so V3_DIR must be on sys.path before importing the model.
sys.path.insert(0, V3_DIR)

import chess  # noqa: E402
from config import INPUT_PLANES, POLICY_OUTPUT_SIZE  # noqa: E402
from chess_nn.board_encoding import HISTORY_LENGTH, boards_to_tensor  # noqa: E402
from chess_nn.model import ChessNet  # noqa: E402
from chess_nn.move_encoding import policy_to_moves  # noqa: E402

OPSET = 17


def load_model() -> tuple[ChessNet, int]:
    """Checkpoint keys are prefixed `_orig_mod.` (torch.compile artifact); strip it."""
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    prefix = "_orig_mod."
    clean = {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state.items()
    }
    model = ChessNet()
    missing, unexpected = model.load_state_dict(clean, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    return model, param_count


def export_onnx(model: ChessNet) -> None:
    dummy = torch.zeros(1, INPUT_PLANES, 8, 8, dtype=torch.float32)
    # dynamo=False: the legacy TorchScript exporter embeds weights as
    # initializers in a single .onnx file. The dynamo path externalizes weights
    # to a .data sidecar and emits opset-17 graphs the int8 quantizer can't
    # shape-infer. The static net (no data-dependent control flow) traces cleanly.
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            ONNX_FP32,
            input_names=["input"],
            output_names=["policy", "value"],
            opset_version=OPSET,
            dynamic_axes={
                "input": {0: "batch"},
                "policy": {0: "batch"},
                "value": {0: "batch"},
            },
            do_constant_folding=True,
            dynamo=False,
        )


def quantize() -> str:
    """Try int8 dynamic quantization; fall back to copying the fp32 model."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=ONNX_FP32,
            model_output=ONNX_FINAL,
            weight_type=QuantType.QInt8,
        )
        return "int8"
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"[warn] int8 quantization unavailable ({exc!r}); shipping fp32.")
        import shutil

        shutil.copyfile(ONNX_FP32, ONNX_FINAL)
        return "fp32"


def validate_onnx(model: ChessNet, sample_tensor: np.ndarray) -> None:
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover
        print(f"[warn] onnxruntime not importable for validation: {exc!r}")
        return

    sess = ort.InferenceSession(ONNX_FINAL, providers=["CPUExecutionProvider"])
    inp = sample_tensor[None].astype(np.float32)
    onnx_policy, onnx_value = sess.run(["policy", "value"], {"input": inp})
    with torch.no_grad():
        t_policy, t_value = model(torch.from_numpy(inp))
    p_mad = float(np.abs(onnx_policy - t_policy.numpy()).mean())
    v_mad = float(np.abs(onnx_value - t_value.numpy()).mean())
    print(f"[validate] policy MAD={p_mad:.4g}  value MAD={v_mad:.4g} (torch vs ONNX)")


# Each fixture: (name, startFen, [uci moves applied from startFen]). The set
# is chosen to cover: perspective flip, multi-frame history, tactical capture,
# full castling rights, an endgame, and the repetition planes 103/104.
FIXTURE_SPECS = [
    ("startpos", chess.STARTING_FEN, []),
    ("after_e4", chess.STARTING_FEN, ["e2e4"]),
    ("italian_8ply", chess.STARTING_FEN,
     ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6"]),
    ("tactical_hanging_queen", "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", []),
    ("black_to_move",
     "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 3", []),
    ("castling_rights",
     "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1",
     []),
    ("kqvk_endgame", "8/8/8/4k3/8/3Q4/4K3/8 w - - 0 1", []),
    ("repetition_line", chess.STARTING_FEN,
     ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]),
]


def build_fixture(model: ChessNet, name: str, start_fen: str, moves: list[str]) -> dict:
    board = chess.Board(start_fen)
    assert board.is_valid(), f"invalid start FEN for {name}: {start_fen}"

    states = [board.copy()]
    for uci in moves:
        mv = chess.Move.from_uci(uci)
        assert mv in board.legal_moves, f"illegal move {uci} in fixture {name}"
        board.push(mv)
        states.append(board.copy())

    current = states[-1]
    # boards[0] = current, then most-recent-first; truncate to HISTORY_LENGTH.
    boards = list(reversed(states))[:HISTORY_LENGTH]

    tensor = boards_to_tensor(boards)
    assert tensor.shape == (INPUT_PLANES, 8, 8)
    plane_sums = tensor.reshape(INPUT_PLANES, 64).sum(axis=1).tolist()
    input_checksum = float(sum(plane_sums))

    with torch.no_grad():
        policy_logits, value_logits = model(torch.from_numpy(tensor)[None])

    top = policy_to_moves(
        policy_logits.squeeze(0).cpu().numpy(), current, top_k=10
    )
    top_moves = [{"uci": mv.uci(), "prob": float(prob)} for mv, prob in top]

    wdl = torch.softmax(value_logits.squeeze(0), dim=0).tolist()

    return {
        "name": name,
        "startFen": start_fen,
        "moves": moves,
        "fen": current.fen(),
        "turn": "w" if current.turn == chess.WHITE else "b",
        "planeSums": [float(x) for x in plane_sums],
        "inputChecksum": input_checksum,
        "topMoves": top_moves,
        "wdl": [float(x) for x in wdl],
    }


def build_fixtures(model: ChessNet) -> tuple[dict, np.ndarray]:
    fixtures = []
    sample_tensor = None
    for name, start_fen, moves in FIXTURE_SPECS:
        fx = build_fixture(model, name, start_fen, moves)
        fixtures.append(fx)
        if sample_tensor is None:
            b = chess.Board(start_fen)
            states = [b.copy()]
            for uci in moves:
                b.push(chess.Move.from_uci(uci))
                states.append(b.copy())
            sample_tensor = boards_to_tensor(list(reversed(states))[:HISTORY_LENGTH])

    doc = {
        "model": "chessnet-v3",
        "inputPlanes": INPUT_PLANES,
        "policySize": POLICY_OUTPUT_SIZE,
        "historyLength": HISTORY_LENGTH,
        "encodingRule": (
            "boards = reverse(states)[:8] where states = [startFen, after move1, "
            "..., current]; boards[0]=current, then most-recent-first; pass to "
            "boards_to_tensor"
        ),
        "fixtures": fixtures,
    }
    return doc, sample_tensor


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    model, param_count = load_model()
    print(f"[load] ChessNet loaded, {param_count:,} params")

    export_onnx(model)
    print(f"[export] fp32 ONNX written -> {ONNX_FP32}")

    quant_tag = quantize()
    print(f"[quant] final ONNX quantization = {quant_tag} -> {ONNX_FINAL}")

    fixtures_doc, sample_tensor = build_fixtures(model)
    with open(FIXTURES, "w") as f:
        json.dump(fixtures_doc, f, indent=2)
    print(f"[fixtures] {len(fixtures_doc['fixtures'])} fixtures -> {FIXTURES}")

    validate_onnx(model, sample_tensor)

    file_bytes = os.path.getsize(ONNX_FINAL)
    metadata = {
        "model": "chessnet-v3",
        "inputPlanes": INPUT_PLANES,
        "policySize": POLICY_OUTPUT_SIZE,
        "historyLength": HISTORY_LENGTH,
        "quantization": quant_tag,
        "paramCount": int(param_count),
        "fileBytes": int(file_bytes),
        "valueOrder": ["win", "draw", "loss"],
        "opset": OPSET,
        "exportedAt": datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    with open(METADATA, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[metadata] -> {METADATA}")

    if quant_tag == "int8" and os.path.exists(ONNX_FP32):
        os.remove(ONNX_FP32)

    print("\nDone.")
    print(f"  {ONNX_FINAL}  ({file_bytes:,} bytes, {quant_tag})")
    print(f"  {METADATA}")
    print(f"  {FIXTURES}")


if __name__ == "__main__":
    main()
