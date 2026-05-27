#!/usr/bin/env python3
"""
Build the 50-position value-bias probe set at data/value_probe.json.

Composition:
  10 symmetric    — true white-POV ~ 0 (hand-picked, source="manual")
  10 white-advantage  — Stockfish depth-20 says >= +150 cp
  10 black-advantage  — Stockfish depth-20 says <= -150 cp
  10 opening      — ply 4-12 from main lines
  10 endgame      — 7-12 pieces, above Syzygy threshold

Conversion: true_white_pov = tanh(cp / 600.0), clipped to [-1, +1].
For mate scores, use +1.0 / -1.0.
"""

import os
import json
import math
import sys
import time

import chess
import chess.engine

STOCKFISH = "/opt/homebrew/bin/stockfish"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(THIS_DIR, "data", "value_probe.json")
DEFAULT_DEPTH = 20
FALLBACK_DEPTH = 15
TIMEOUT_SEC = 10.0
MATE_SCORE = 10000


# ---------------------------- position pools ----------------------------

# 10 hand-picked symmetric positions where the true white-POV value should be ~0.
SYMMETRIC_FENS = [
    # Starting position
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0.0),
    # Mirrored: black to move from the start position (after a "null move" mirror)
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", 0.0),
    # King vs King — theoretical draw
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0.0),
    # KB vs K — theoretical draw (insufficient material)
    ("4k3/8/8/8/8/8/8/4KB2 w - - 0 1", 0.0),
    # KN vs K — theoretical draw (insufficient material)
    ("4k3/8/8/8/8/8/8/4KN2 w - - 0 1", 0.0),
    # KNN vs K — theoretical draw
    ("4k3/8/8/8/8/8/8/3NKN2 w - - 0 1", 0.0),
    # Symmetric pawn endgame — locked pawn chain, mirror image
    ("4k3/p1p1p1p1/1p1p1p1p/8/8/P1P1P1P1/1P1P1P1P/4K3 w - - 0 1", 0.0),
    # Symmetric K+P vs K+P (e2 vs e7), white to move — still ~0 (mirror)
    ("4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1", 0.0),
    # KR vs KR — theoretical draw, mirror
    ("4k2r/8/8/8/8/8/8/R3K3 w Qk - 0 1", 0.0),
    # Symmetric blocked center — both sides have the same pieces and structure
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 1", 0.0),
]

# 10 opening positions (plies 4-12 from main lines) — Stockfish-evaluated.
OPENING_FENS = [
    # Italian Game — Giuoco Piano main line (after 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5)
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    # Ruy Lopez — Berlin Defense (after 1.e4 e5 2.Nf3 Nc6 3.Bb5 Nf6)
    "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    # Open Sicilian Najdorf (after 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6)
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    # French Defense — Winawer (after 1.e4 e6 2.d4 d5 3.Nc3 Bb4)
    "rnbqk1nr/ppp2ppp/4p3/3p4/1b1PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4",
    # Queen's Gambit Declined main (after 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7)
    "rnbqk2r/ppp1bppp/4pn2/3p2B1/2PP4/2N5/PP2PPPP/R2QKBNR w KQkq - 4 5",
    # King's Indian Defense (after 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6)
    "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 5",
    # Caro-Kann main (after 1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Bf5)
    "rn1qkbnr/pp2pppp/2p5/5b2/3PN3/8/PPP2PPP/R1BQKBNR w KQkq - 2 5",
    # Slav Defense (after 1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3 dxc4)
    "rnbqkb1r/pp2pppp/2p2n2/8/2pP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",
    # English Opening — symmetric (1.c4 c5 2.Nc3 Nc6 3.g3 g6 4.Bg2 Bg7)
    "r1bqk1nr/pp1pppbp/2n3p1/2p5/2P5/2N3P1/PP1PPPBP/R1BQK1NR w KQkq - 4 5",
    # Pirc Defense (after 1.e4 d6 2.d4 Nf6 3.Nc3 g6 4.Be2 Bg7)
    "rnbqk2r/ppp1ppbp/3p1np1/8/3PP3/2N5/PPP1BPPP/R1BQK1NR w KQkq - 3 5",
]

# 10 endgame positions (7-12 pieces, just above Syzygy threshold) — Stockfish-evaluated.
ENDGAME_FENS = [
    # K+R+P vs K+R (Lucena-ish), white better
    "4k3/8/8/8/8/4P3/3R4/4K2r w - - 0 1",
    # K+R+P vs K+R (Philidor-like), drawish
    "4k3/4r3/4p3/8/8/8/3R4/4K3 w - - 0 1",
    # K+B+N vs K+P, white to play — winning technique
    "4k3/8/4p3/8/8/8/4B3/3NK3 w - - 0 1",
    # K+N+P vs K — drawish if rook pawn, here a knight pawn so it's winning
    "4k3/8/8/8/8/8/1P6/2N1K3 w - - 0 1",
    # K+Q vs K+R, classic winning endgame for white
    "4k3/8/8/8/8/8/8/3QK2r w - - 0 1",
    # K+B+P vs K+P (opposite-color bishops trend draw)
    "4k3/3b4/4p3/8/8/8/3PB3/4K3 w - - 0 1",
    # K+R vs K+B+N — drawn with correct play, slight black edge
    "4k3/8/4bn2/8/8/8/8/3RK3 w - - 0 1",
    # K+R+P vs K+R+P, opposite wings (Karpov style)
    "4k3/4r3/p7/8/8/7P/4R3/4K3 w - - 0 1",
    # K+Q+P vs K+Q — winning for the side with the extra pawn
    "4k3/8/8/8/3q4/8/2P5/3QK3 w - - 0 1",
    # K+R+B vs K+R, slight edge to side with bishop
    "4k3/8/8/8/8/8/4B3/3RK2r w - - 0 1",
]

# Candidate positions for white-advantage and black-advantage pools.
# Stockfish will filter by cp threshold; we collect more than 10 and slice.
WHITE_ADV_CANDIDATES = [
    # White up a pawn, simple middlegame
    "r1bqkbnr/ppp2ppp/2n5/3pp3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 5",
    # White up a piece (clean), endgame
    "4k3/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",
    # White up the exchange (R vs B)
    "4k3/8/8/3b4/8/8/3R4/4K3 w - - 0 1",
    # Open Sicilian where white attack is strong (Yugoslav-style)
    "r1bqk2r/pp2bppp/2nppn2/8/3NPP2/2N1B3/PPPQ2PP/R3KB1R w KQkq - 0 9",
    # White has bishop pair + space, black cramped
    "r2qkb1r/pp2nppp/3p4/2pNn3/4P3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 0 8",
    # White up a pawn in a queenless middlegame
    "r1b1kb1r/ppp1pppp/2n5/3n4/3P4/2N2N2/PPP2PPP/R1B1KB1R w KQkq - 0 7",
    # White has a passed pawn on the 7th
    "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1",
    # White has a huge attack — sacrificed material but mating
    "r1bk3r/ppp2Bpp/2n5/4N3/8/8/PPPP1PPP/R3K2R w KQ - 0 12",
    # White rook on 7th rank, passed e-pawn
    "r3k3/1R6/8/4P3/8/8/8/4K3 w - - 0 1",
    # White up two pawns
    "4k3/4p3/8/8/8/PPP5/8/4K3 w - - 0 1",
    # White Queen vs minor piece
    "4k3/8/8/8/8/8/8/Q3K1n1 w - - 0 1",
    # White up a piece in an opening line
    "r1bqkbnr/pppp1ppp/8/4p3/3nP3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4",
    # White up a clean rook (no compensation)
    "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
    # White up two minor pieces in a middlegame-style position
    "r2qk2r/ppp2ppp/8/3pp3/3PP3/2N2N2/PPP2PPP/R2QK2R w KQkq - 0 8",
    # White up a queen
    "4k3/pppp1ppp/8/8/8/8/PPP1PPPP/3QK3 w - - 0 1",
    # Open Sicilian with white attack already underway (English Attack vibes)
    "r1bqk2r/pp2bppp/2nppn2/8/3NPP2/2N1BQ2/PPP3PP/2KR1B1R w kq - 0 10",
    # White up the exchange + pawn
    "4k3/8/8/3b4/8/4P3/3R4/4K3 w - - 0 1",
]

BLACK_ADV_CANDIDATES = [
    # Black up a pawn, simple middlegame (mirror of white)
    "r1bqkb1r/ppp2ppp/2n2n2/3pp3/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 0 5",
    # Black up a piece (clean)
    "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/4K3 b kq - 0 1",
    # Black up the exchange (R vs B)
    "4k2r/3r4/8/8/8/3B4/8/4K3 b k - 0 1",
    # Black has strong attack — analog of Yugoslav-attack mirror
    "r1bqkb1r/ppp1ppp1/3p1n2/8/3NPP1n/2N1B3/PPPQ2PP/R3KB1R b KQkq - 0 9",
    # Black has bishop pair + space
    "r2qkb1r/ppp2ppp/3p4/2pNn3/3nP3/2N1B3/PPP2PPP/R2QKB1R b KQkq - 0 8",
    # Black up a pawn, queenless
    "r1b1kb1r/ppp2ppp/2n5/3n4/3p4/2N2N2/PPP2PPP/R1B1KB1R b KQkq - 0 7",
    # Black has a passed pawn on the 2nd
    "4k3/8/8/8/8/8/4p3/4K3 b - - 0 1",
    # Black up a piece, retained bishop pair, white king exposed
    "r3kb1r/pppp1ppp/8/8/8/2n5/PPP3PP/R1B1K2R b KQkq - 0 12",
    # Black rook on 2nd, passed pawn
    "4k3/8/8/8/4p3/8/4r1PP/5K2 b - - 0 1",
    # Black up two pawns
    "4k3/8/ppp5/8/8/8/4P3/4K3 b - - 0 1",
    # Black queen vs minor piece
    "1n2k3/8/8/8/8/8/4K3/7q b - - 0 1",
    # Black up a piece in an opening line
    "rnbqkb1r/pppp1ppp/5n2/4p3/3NP3/8/PPPP1PPP/RNBQKB1R b KQkq - 0 4",
    # Black up a clean rook
    "r3k3/8/8/8/8/8/8/4K3 b q - 0 1",
    # Black up two minor pieces
    "r2qk2r/ppp2ppp/8/3pp3/3PP3/2N2N2/PPP2PPP/R2QK2R b KQkq - 0 8",
    # Black up a queen
    "3qk3/ppp1pppp/8/8/8/8/PPPP1PPP/4K3 b - - 0 1",
    # Black up the exchange + pawn
    "4k3/3r4/4p3/8/8/3B4/8/4K3 b - - 0 1",
]


# ---------------------------- helpers ----------------------------

def cp_to_white_pov(cp: int) -> float:
    """tanh(cp/600), clipped to [-1, +1]."""
    v = math.tanh(cp / 600.0)
    return max(-1.0, min(1.0, v))


def analyse_with_fallback(engine, board):
    """Analyse with depth=20; fall back to depth=15 if it takes >TIMEOUT_SEC."""
    t0 = time.time()
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=DEFAULT_DEPTH, time=TIMEOUT_SEC))
        elapsed = time.time() - t0
        if elapsed > TIMEOUT_SEC:
            # Re-analyse at lower depth and mark fallback
            info = engine.analyse(board, chess.engine.Limit(depth=FALLBACK_DEPTH))
            return info, FALLBACK_DEPTH
        return info, DEFAULT_DEPTH
    except Exception as e:
        print(f"  analyse failed: {e}; retrying at depth {FALLBACK_DEPTH}")
        info = engine.analyse(board, chess.engine.Limit(depth=FALLBACK_DEPTH))
        return info, FALLBACK_DEPTH


def score_to_cp(info) -> int:
    """Pull white-POV centipawn score (mate -> +/-MATE_SCORE)."""
    return info["score"].white().score(mate_score=MATE_SCORE)


# ---------------------------- main build ----------------------------

def build_entries(engine):
    entries = []

    # --- Symmetric (manual, no Stockfish) ---
    print("Building 10 symmetric positions (manual)...")
    for fen, val in SYMMETRIC_FENS:
        entries.append({
            "fen": fen,
            "label": "symmetric",
            "true_white_pov": float(val),
            "source": "manual",
        })

    # --- Openings ---
    print("Building 10 opening positions (stockfish-d20)...")
    for i, fen in enumerate(OPENING_FENS):
        board = chess.Board(fen)
        info, depth = analyse_with_fallback(engine, board)
        cp = score_to_cp(info)
        wp = cp_to_white_pov(cp)
        entry = {
            "fen": fen,
            "label": "opening",
            "true_white_pov": round(wp, 4),
            "source": f"stockfish-d{depth}",
            "stockfish_cp": int(cp),
        }
        entries.append(entry)
        print(f"  [{i+1}/10] cp={cp:+5d} wp={wp:+.3f}  {fen}")

    # --- Endgames ---
    print("Building 10 endgame positions (stockfish-d20)...")
    for i, fen in enumerate(ENDGAME_FENS):
        board = chess.Board(fen)
        info, depth = analyse_with_fallback(engine, board)
        cp = score_to_cp(info)
        wp = cp_to_white_pov(cp)
        entry = {
            "fen": fen,
            "label": "endgame",
            "true_white_pov": round(wp, 4),
            "source": f"stockfish-d{depth}",
            "stockfish_cp": int(cp),
        }
        entries.append(entry)
        print(f"  [{i+1}/10] cp={cp:+5d} wp={wp:+.3f}  {fen}")

    # --- White advantage (filter cp >= +150) ---
    print("Building 10 white-advantage positions (stockfish-d20, requires cp >= +150)...")
    chosen = 0
    for i, fen in enumerate(WHITE_ADV_CANDIDATES):
        if chosen >= 10:
            break
        board = chess.Board(fen)
        info, depth = analyse_with_fallback(engine, board)
        cp = score_to_cp(info)
        if cp < 150:
            print(f"  SKIP (cp={cp:+d} < 150)  {fen}")
            continue
        wp = cp_to_white_pov(cp)
        entry = {
            "fen": fen,
            "label": "white-advantage",
            "true_white_pov": round(wp, 4),
            "source": f"stockfish-d{depth}",
            "stockfish_cp": int(cp),
        }
        entries.append(entry)
        chosen += 1
        print(f"  [{chosen}/10] cp={cp:+5d} wp={wp:+.3f}  {fen}")
    if chosen < 10:
        raise RuntimeError(f"Only {chosen}/10 white-advantage candidates passed cp>=150 filter; "
                           f"add more candidates.")

    # --- Black advantage (filter cp <= -150) ---
    print("Building 10 black-advantage positions (stockfish-d20, requires cp <= -150)...")
    chosen = 0
    for i, fen in enumerate(BLACK_ADV_CANDIDATES):
        if chosen >= 10:
            break
        board = chess.Board(fen)
        info, depth = analyse_with_fallback(engine, board)
        cp = score_to_cp(info)
        if cp > -150:
            print(f"  SKIP (cp={cp:+d} > -150)  {fen}")
            continue
        wp = cp_to_white_pov(cp)
        entry = {
            "fen": fen,
            "label": "black-advantage",
            "true_white_pov": round(wp, 4),
            "source": f"stockfish-d{depth}",
            "stockfish_cp": int(cp),
        }
        entries.append(entry)
        chosen += 1
        print(f"  [{chosen}/10] cp={cp:+5d} wp={wp:+.3f}  {fen}")
    if chosen < 10:
        raise RuntimeError(f"Only {chosen}/10 black-advantage candidates passed cp<=-150 filter; "
                           f"add more candidates.")

    return entries


def main():
    print(f"Opening Stockfish at {STOCKFISH}...")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    try:
        entries = build_entries(engine)
    finally:
        engine.quit()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(entries, f, indent=2)

    # Sanity check
    by_label = {}
    for e in entries:
        by_label.setdefault(e["label"], 0)
        by_label[e["label"]] += 1
    print()
    print(f"Wrote {len(entries)} entries to {OUT_PATH}")
    for label in ["symmetric", "white-advantage", "black-advantage", "opening", "endgame"]:
        print(f"  {label:<18} {by_label.get(label, 0)}")


if __name__ == "__main__":
    main()
