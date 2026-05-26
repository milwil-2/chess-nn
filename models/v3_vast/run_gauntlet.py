"""
Local gauntlet: play our UCI engine vs Stockfish at multiple UCI_Elo levels,
compute an Elo number, and append the result to logs/elo_history.csv.

NOTE: cutechess-cli is not in Homebrew (May 2026) and the GUI cask requires
Qt5, so we drive matches via python-chess's chess.engine.SimpleEngine instead.
This is functionally equivalent for our purposes (UCI tournament harness with
opening book + per-game time control + PGN output) and avoids a C++/Qt build.

Default time control: 10s + 0.1s/move increment, 30 games per Stockfish level
across 5 levels (1400/1600/1800/2000/2200). Use --games-per-level 5 for a
~10-minute smoke run.

Outputs
-------
- PGN of all games: logs/gauntlet_<timestamp>.pgn
- CSV history row : logs/elo_history.csv
  columns: timestamp, checkpoint_hash, games, tc, elo, ci_low, ci_high,
           decisive_rate, mean_plies
"""

import argparse
import csv
import datetime
import hashlib
import io
import math
import os
import random
import shutil
import sys
import time
from typing import Optional

# Force unbuffered stdout so progress lines show up promptly when piped to
# tee/measure_elo.sh.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

import chess
import chess.engine
import chess.pgn
import chess.polyglot

# Repo paths
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REPO_ROOT = HERE  # this module lives next to run.py
RUN_PY = os.path.join(HERE, "run.py")
DEFAULT_CHECKPOINT = os.path.join(HERE, "checkpoints", "best_model.pt")
DEFAULT_BOOK = os.path.join(HERE, "data", "book.bin")
LOGS_DIR = os.path.join(HERE, "logs")
ELO_HISTORY_CSV = os.path.join(LOGS_DIR, "elo_history.csv")

PY = "/opt/homebrew/bin/python3.12"
SF_LEVELS = [1400, 1600, 1800, 2000, 2200]


def find_stockfish() -> str:
    sf = shutil.which("stockfish")
    if not sf:
        sys.exit("stockfish not on PATH — run `brew install stockfish`")
    return sf


def checkpoint_hash(path: str) -> str:
    if not os.path.exists(path):
        return "no-checkpoint"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def parse_tc(tc: str) -> tuple[float, float]:
    """'10+0.1' -> (10.0, 0.1).  '30' -> (30.0, 0.0)."""
    if "+" in tc:
        base_s, inc_s = tc.split("+", 1)
        return float(base_s), float(inc_s)
    return float(tc), 0.0


def random_book_opening(book_path: str, plies: int = 8) -> chess.Board:
    """Sample a random opening line from the Polyglot book. Returns the
    board after `plies` book moves, or fewer if the book runs out."""
    board = chess.Board()
    try:
        with chess.polyglot.open_reader(book_path) as reader:
            for _ in range(plies):
                try:
                    entries = list(reader.find_all(board))
                except IndexError:
                    break
                if not entries:
                    break
                # Weight by polyglot 'weight' field; fall back to uniform.
                weights = [max(1, e.weight) for e in entries]
                entry = random.choices(entries, weights=weights, k=1)[0]
                board.push(entry.move)
                if board.is_game_over():
                    break
    except (FileNotFoundError, OSError):
        pass
    return board


def play_one_game(our_engine, sf_engine, start_board: chess.Board,
                  base_s: float, inc_s: float, our_color: chess.Color,
                  movetime_cap_s: float = 10.0) -> tuple[str, list[chess.Move], int]:
    """Play one full game. Returns (result, move_list, num_plies).

    result is one of "1-0", "0-1", "1/2-1/2" (PGN convention).
    """
    board = start_board.copy()
    moves: list[chess.Move] = []
    w_time, b_time = base_s, base_s

    while not board.is_game_over(claim_draw=True) and len(moves) < 400:
        side = board.turn
        engine = our_engine if side == our_color else sf_engine
        # Send the per-move time budget. Use the wtime/btime/winc/binc form so
        # the engine can self-manage; also enforce a hard cap on movetime.
        limit = chess.engine.Limit(
            white_clock=max(0.05, w_time),
            black_clock=max(0.05, b_time),
            white_inc=inc_s,
            black_inc=inc_s,
        )
        t0 = time.monotonic()
        try:
            result = engine.play(board, limit, game=object())
        except chess.engine.EngineTerminatedError:
            return ("0-1" if side == our_color else "1-0"), moves, len(moves)
        except Exception as exc:
            print(f"  [engine error: {exc}]", file=sys.stderr)
            return ("0-1" if side == our_color else "1-0"), moves, len(moves)
        elapsed = time.monotonic() - t0

        if result.move is None or result.move not in board.legal_moves:
            # Illegal/missing move = loss for that side
            return ("0-1" if side == our_color else "1-0"), moves, len(moves)

        # Update clocks
        if side == chess.WHITE:
            w_time = max(0.0, w_time - elapsed) + inc_s
        else:
            b_time = max(0.0, b_time - elapsed) + inc_s

        # Time forfeit
        if (side == chess.WHITE and w_time <= 0) or (side == chess.BLACK and b_time <= 0):
            return ("0-1" if side == our_color else "1-0"), moves, len(moves)

        board.push(result.move)
        moves.append(result.move)

    if board.is_checkmate():
        # Side to move is the one mated. Winner is the other side.
        winner = not board.turn
        return ("1-0" if winner == chess.WHITE else "0-1"), moves, len(moves)
    return "1/2-1/2", moves, len(moves)


def boot_our_engine(checkpoint: str, sims: int):
    return chess.engine.SimpleEngine.popen_uci([
        PY, RUN_PY, "engine",
        "--checkpoint", checkpoint,
        "--sims", str(sims),
    ])


def boot_stockfish(sf_path: str, uci_elo: int, threads: int = 1, hash_mb: int = 64):
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({
        "UCI_LimitStrength": True,
        "UCI_Elo": uci_elo,
        "Threads": threads,
        "Hash": hash_mb,
    })
    return eng


def elo_from_score(score: float, n: int) -> tuple[float, float, float]:
    """Approximate Elo difference from win-rate `score` over `n` games.

    Returns (elo_diff, ci_low, ci_high) using a ~95% Wald CI on score.
    Caps score in (eps, 1-eps) to avoid log10(0) blowups.
    """
    eps = 1e-3
    s = max(eps, min(1 - eps, score))
    elo = -400.0 * math.log10(1.0 / s - 1.0)

    if n <= 1:
        return elo, elo - 400.0, elo + 400.0

    se = math.sqrt(max(eps, s * (1 - s) / n))
    lo = max(eps, s - 1.96 * se)
    hi = min(1 - eps, s + 1.96 * se)
    elo_lo = -400.0 * math.log10(1.0 / lo - 1.0)
    elo_hi = -400.0 * math.log10(1.0 / hi - 1.0)
    return elo, elo_lo, elo_hi


def aggregate_elo(per_level: list[dict]) -> tuple[float, float, float]:
    """Inverse-variance weighted aggregate of per-level Elo estimates.

    Each entry has keys: sf_elo, wins, draws, losses, n. Our engine's Elo
    estimate from that level = sf_elo + elo_diff(score_vs_sf, n).
    """
    rows = []
    for r in per_level:
        n = r["n"]
        if n == 0:
            continue
        score = (r["wins"] + 0.5 * r["draws"]) / n
        elo, lo, hi = elo_from_score(score, n)
        est = r["sf_elo"] + elo
        # Approximate variance from CI half-width
        sigma = max(1.0, (hi - lo) / 2.0)
        rows.append((est, sigma))

    if not rows:
        return float("nan"), float("nan"), float("nan")

    # Inverse-variance weighting
    weights = [1.0 / (s * s) for _, s in rows]
    wsum = sum(weights)
    mean = sum(est * w for (est, _), w in zip(rows, weights)) / wsum
    var = 1.0 / wsum
    sigma_total = math.sqrt(var)
    return mean, mean - 1.96 * sigma_total, mean + 1.96 * sigma_total


def append_history(elo: float, lo: float, hi: float, games: int, tc: str,
                   ckpt_hash: str, decisive_rate: float, mean_plies: float) -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)
    new_file = not os.path.exists(ELO_HISTORY_CSV)
    with open(ELO_HISTORY_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "checkpoint_hash", "games", "tc",
                        "elo", "ci_low", "ci_high",
                        "decisive_rate", "mean_plies"])
        w.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            ckpt_hash,
            games,
            tc,
            f"{elo:.1f}",
            f"{lo:.1f}",
            f"{hi:.1f}",
            f"{decisive_rate:.3f}",
            f"{mean_plies:.1f}",
        ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Gauntlet: chess-nn vs Stockfish")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--book", default=DEFAULT_BOOK)
    parser.add_argument("--tc", default="10+0.1",
                        help="Time control 'base+inc' in seconds (default 10+0.1)")
    parser.add_argument("--games-per-level", type=int, default=30)
    parser.add_argument("--sims", type=int, default=200,
                        help="Default MCTS sims for engine (overridden by time controls)")
    parser.add_argument("--levels", type=int, nargs="*", default=SF_LEVELS,
                        help=f"Stockfish UCI_Elo levels (default {SF_LEVELS})")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    base_s, inc_s = parse_tc(args.tc)
    sf_path = find_stockfish()
    ckpt_hash = checkpoint_hash(args.checkpoint)

    os.makedirs(LOGS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pgn_path = os.path.join(LOGS_DIR, f"gauntlet_{ts}.pgn")

    print(f"=== Gauntlet ===")
    print(f"Checkpoint     : {args.checkpoint} ({ckpt_hash})")
    print(f"Stockfish      : {sf_path}")
    print(f"TC             : {args.tc} ({base_s}s + {inc_s}s/move)")
    print(f"Games/level    : {args.games_per_level}")
    print(f"SF levels      : {args.levels}")
    print(f"PGN output     : {pgn_path}")
    print()

    per_level_summary = []
    total_decisive = 0
    total_plies = 0
    total_games = 0

    our_engine = boot_our_engine(args.checkpoint, args.sims)
    try:
        with open(pgn_path, "w") as pgn_f:
            for sf_elo in args.levels:
                sf_engine = boot_stockfish(sf_path, sf_elo)
                wins = draws = losses = 0
                plies_sum = 0
                t_level = time.monotonic()
                try:
                    for g in range(args.games_per_level):
                        # Alternate colors per game
                        our_color = chess.WHITE if g % 2 == 0 else chess.BLACK
                        start = random_book_opening(args.book, plies=8) \
                            if os.path.exists(args.book) else chess.Board()
                        try:
                            our_engine.protocol.send_line("ucinewgame")
                        except Exception:
                            pass
                        try:
                            sf_engine.protocol.send_line("ucinewgame")
                        except Exception:
                            pass

                        result, moves, n_plies = play_one_game(
                            our_engine, sf_engine, start, base_s, inc_s,
                            our_color,
                        )
                        # Score from our POV
                        if result == "1-0":
                            ours = 1.0 if our_color == chess.WHITE else 0.0
                        elif result == "0-1":
                            ours = 1.0 if our_color == chess.BLACK else 0.0
                        else:
                            ours = 0.5
                        if ours == 1.0:
                            wins += 1
                        elif ours == 0.0:
                            losses += 1
                        else:
                            draws += 1
                        plies_sum += n_plies
                        if result != "1/2-1/2":
                            total_decisive += 1
                        total_plies += n_plies
                        total_games += 1

                        # Write PGN — build a full board from scratch including
                        # book-opening plies + game plies so chess.pgn can
                        # compute SAN correctly.
                        full_board = chess.Board()
                        if start.fen() != chess.STARTING_FEN:
                            game = chess.pgn.Game()
                            game.setup(start)
                            full_board = start.copy()
                        else:
                            # Replay book moves so headers reflect a normal opening
                            game = chess.pgn.Game()
                            for m in start.move_stack:
                                pass  # startpos: no book replay needed
                        game.headers["Event"] = f"Gauntlet vs SF UCI_Elo={sf_elo}"
                        game.headers["Site"] = "local"
                        game.headers["Date"] = datetime.date.today().isoformat()
                        game.headers["Round"] = str(g + 1)
                        game.headers["White"] = ("Chess-NN" if our_color == chess.WHITE
                                                 else f"Stockfish_{sf_elo}")
                        game.headers["Black"] = ("Chess-NN" if our_color == chess.BLACK
                                                 else f"Stockfish_{sf_elo}")
                        game.headers["Result"] = result
                        game.headers["TimeControl"] = args.tc

                        cur = game
                        for m in moves:
                            cur = cur.add_variation(m)
                        pgn_f.write(str(game) + "\n\n")
                        pgn_f.flush()

                        score_now = (wins + 0.5 * draws) / (g + 1)
                        print(f"  SF={sf_elo} g{g+1:>3}: {result:>7}  "
                              f"({'W' if our_color == chess.WHITE else 'B'}, "
                              f"{n_plies}p)  cumulative score={score_now:.2f} "
                              f"(W/D/L={wins}/{draws}/{losses})")
                finally:
                    sf_engine.quit()

                level_time = time.monotonic() - t_level
                n = wins + draws + losses
                score = (wins + 0.5 * draws) / n if n else 0.0
                elo, lo, hi = elo_from_score(score, n)
                est_elo = sf_elo + elo
                print(f"  ---- SF={sf_elo}: W/D/L = {wins}/{draws}/{losses} "
                      f"score={score:.3f}  -> our Elo ≈ {est_elo:.0f} "
                      f"({sf_elo + lo:.0f}..{sf_elo + hi:.0f})  "
                      f"[{level_time:.1f}s]")
                print()
                per_level_summary.append({
                    "sf_elo": sf_elo, "wins": wins, "draws": draws,
                    "losses": losses, "n": n,
                })
    finally:
        try:
            our_engine.quit()
        except Exception:
            pass

    final_elo, final_lo, final_hi = aggregate_elo(per_level_summary)
    decisive_rate = total_decisive / max(1, total_games)
    mean_plies = total_plies / max(1, total_games)

    print("=" * 60)
    print(f"Final aggregate Elo: {final_elo:.0f}  CI95: [{final_lo:.0f}, {final_hi:.0f}]")
    print(f"Games: {total_games}  decisive_rate={decisive_rate:.2f}  mean_plies={mean_plies:.1f}")
    print()
    print("Per-level:")
    for r in per_level_summary:
        n = r["n"] or 1
        score = (r["wins"] + 0.5 * r["draws"]) / n
        print(f"  SF={r['sf_elo']:>4}: {r['wins']:>3}W {r['draws']:>3}D {r['losses']:>3}L "
              f"score={score:.3f}")

    append_history(final_elo, final_lo, final_hi, total_games, args.tc,
                   ckpt_hash, decisive_rate, mean_plies)
    print(f"\nAppended row to {ELO_HISTORY_CSV}")
    print(f"PGN saved to     {pgn_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
