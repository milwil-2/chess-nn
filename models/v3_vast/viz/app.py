"""
Main Pygame application — play chess against the neural network and watch it think.

Layout:
  Left:    640×640 chess board (8×80px squares)
  Right:   300px sidebar — value bar, controls, layer selector

Controls:
  Click piece → click destination to move (you play White)
  A key: accept the AI's top suggested move for your side
  Space: toggle auto-play (AI vs AI)
  H key: toggle activation heatmap
  1-5 keys: select which residual block to visualise
  R key: reset game
  ESC: quit
"""

import sys
import os
import json
import warnings
from collections import deque
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
import pygame
import chess
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHECKPOINT_DIR, NUM_RESIDUAL_BLOCKS
from chess_nn.model import ChessNet
from chess_nn.board_encoding import boards_to_tensor
from chess_nn.move_encoding import policy_to_moves
from chess_nn.utils import load_checkpoint
from chess_nn.evaluate import select_move
from chess_nn.mcts import MCTS

from viz.board_renderer import BoardRenderer
from viz.move_arrows import draw_move_arrows
from viz.heatmap import draw_heatmap
from viz.value_bar import draw_value_bar
from chess_nn.tactics import detect_tactics, Tactic

SQ_SIZE   = 80
BOARD_PX  = SQ_SIZE * 8
LABEL_H   = 30          # height reserved above/below board for model name labels
SIDEBAR_W = 280
WIN_W     = BOARD_PX + SIDEBAR_W + 40
WIN_H     = BOARD_PX + 40 + 2 * LABEL_H

BG         = (40, 40, 40)
SIDEBAR_BG = (50, 50, 55)
TEXT_COLOR = (220, 220, 220)
ACCENT     = (100, 180, 100)

# Auto-play delay in milliseconds between moves
AUTOPLAY_DELAY_MS = 600


def load_model(checkpoint_name: str = "best_model.pt") -> ChessNet:
    model = ChessNet()
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    if not os.path.isabs(checkpoint_name) and not os.path.exists(checkpoint_path):
        checkpoint_path = checkpoint_name  # allow absolute or relative paths
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model)
        print(f"Model loaded from {checkpoint_path}")
    else:
        print("No checkpoint found — using untrained model.")
    model.eval()
    return model


def game_over_description(board: chess.Board) -> str:
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        return f"Checkmate — {winner} wins!"
    if board.is_stalemate():
        return "Stalemate — Draw"
    if board.is_insufficient_material():
        return "Insufficient material — Draw"
    if board.is_seventyfive_moves():
        return "75-move rule — Draw"
    if board.is_fivefold_repetition():
        return "Fivefold repetition — Draw"
    return f"Game over ({board.result()})"


STATE_FILE = "/tmp/chess_nn_state.npz"

# --- Move logging (for later analysis of bad moves) ---
import datetime
_VIZ_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)
os.makedirs(_VIZ_LOG_DIR, exist_ok=True)
MOVE_LOG_PATH = os.path.join(_VIZ_LOG_DIR, "viz_moves.log")


def _log_write(line: str = "") -> None:
    try:
        with open(MOVE_LOG_PATH, "a") as fh:
            fh.write(line.rstrip("\n") + "\n")
    except Exception:
        pass


def log_session_start(white_name: str, black_name: str, use_mcts: bool,
                      num_sims: int) -> None:
    _log_write()
    _log_write(f"=== session {datetime.datetime.now().isoformat(timespec='seconds')} ===")
    _log_write(
        f"=== white={white_name}  black={black_name}  "
        f"mcts={'on' if use_mcts else 'off'}  sims={num_sims} ==="
    )


def log_game_start(game_idx: int) -> None:
    _log_write(f"--- game {game_idx} start ---")


def log_game_end(game_idx: int, board: chess.Board, plies: int) -> None:
    reason = game_over_description(board)
    _log_write(
        f"--- game {game_idx} end  result={board.result()}  "
        f"plies={plies}  ({reason}) ---"
    )


def log_game_abandoned(game_idx: int, plies: int) -> None:
    _log_write(f"--- game {game_idx} abandoned at ply {plies} ---")


def _to_white_pov(value: float, mover_was_white: bool) -> float:
    """Convert a model value (current-player POV) to white POV for that ply."""
    return float(value) if mover_was_white else -float(value)


def log_move(game_idx: int, ply: int, mover_name: str, mover_color: str,
             chosen_move: chess.Move, chosen_san: str,
             pre_move_probs, pre_value: float, post_value: float,
             use_mcts: bool, board_after: chess.Board) -> None:
    """Append one move record. pre_value is from the mover's POV; post_value
    is from the opponent's POV (i.e. the side now to move). Both get normalized
    to white POV so an evaluation swing is directly readable."""
    pre_w = _to_white_pov(pre_value, mover_was_white=(mover_color == "W"))
    # board_after.turn is now the opponent of the mover.
    post_w = float(post_value) if board_after.turn == chess.WHITE else -float(post_value)

    rank, prior = None, None
    probs = list(pre_move_probs or [])
    for i, (m, p) in enumerate(probs):
        if m == chosen_move:
            rank, prior = i + 1, p
            break
    rank_str = f"{rank}/{len(probs)}" if rank is not None else f">{len(probs)}/{len(probs)}"
    prior_str = f"{prior:.3f}" if prior is not None else "?"
    top3 = " ".join(f"{m.uci()}({p:.2f})" for m, p in probs[:3])
    mode = "MCTS" if use_mcts else "POL "

    _log_write(
        f"G{game_idx} ply{ply:>3} {mover_color} {mover_name:<14} {mode} "
        f"{chosen_move.uci():<6} {chosen_san:<8} "
        f"vW_pre={pre_w:+.3f} vW_post={post_w:+.3f} swing={post_w - pre_w:+.3f} "
        f"rank={rank_str} prior={prior_str} top3=[{top3}] fen={board_after.fen()}"
    )


def run_inference(model: ChessNet, board: chess.Board, heatmap_layer: int):
    """Run the model on the current board. Returns (move_probs, value, heatmap)."""
    import numpy as np
    tensor = torch.from_numpy(boards_to_tensor([board])).unsqueeze(0).float()
    all_acts = model.get_all_activations(tensor)
    move_probs = policy_to_moves(all_acts["policy_logits"].squeeze(), board, top_k=8)
    heatmap = model.get_activations(tensor, layer_index=heatmap_layer).numpy()

    try:
        save_dict = {k: v for k, v in all_acts.items()}
        save_dict["board_tensor"] = boards_to_tensor([board])
        np.savez_compressed(STATE_FILE, **save_dict)
    except Exception:
        pass

    from chess_nn.model import wdl_to_scalar
    value_scalar = wdl_to_scalar(torch.tensor(all_acts["value"]))
    return move_probs, value_scalar, heatmap


def push_ai_move(model, board, board_history, mcts_obj, heatmap_layer, last_move,
                 temperature=0.5, use_mcts=True):
    """Let the AI make a move. Mutates board and board_history in place."""
    hist_list = list(board_history)
    if use_mcts:
        ai_move = mcts_obj.search(board, board_history=hist_list,
                                  temperature=temperature, enable_blunder_filter=True)
    else:
        ai_move = select_move(model, board, board_history=hist_list, temperature=temperature)
    if ai_move in board.legal_moves:
        board.push(ai_move)
        board_history.appendleft(board.copy(stack=False))
        last_move = ai_move
    if board.is_game_over():
        return last_move, [], 0.0, None, game_over_description(board)
    move_probs, value, heatmap = run_inference(model, board, heatmap_layer)
    return last_move, move_probs, value, heatmap, ""


def draw_game_over_overlay(surface, message: str, fonts):
    overlay = pygame.Surface((BOARD_PX, WIN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (20, 0))

    title_surf = fonts["large"].render(message, True, (255, 230, 80))
    sub_surf   = fonts["medium"].render("Press R to restart", True, (200, 200, 200))

    cx = 20 + BOARD_PX // 2
    cy = WIN_H // 2

    pygame.draw.rect(surface, (30, 30, 30),
                     (cx - 200, cy - 52, 400, 104), border_radius=10)
    pygame.draw.rect(surface, (255, 230, 80),
                     (cx - 200, cy - 52, 400, 104), width=2, border_radius=10)

    surface.blit(title_surf, title_surf.get_rect(center=(cx, cy - 16)))
    surface.blit(sub_surf,   sub_surf.get_rect(center=(cx, cy + 22)))


def draw_tactic_highlights(surface, tactics: list, renderer: BoardRenderer):
    """Draw coloured square highlights for detected tactics."""
    overlay = pygame.Surface((BOARD_PX, WIN_H), pygame.SRCALPHA)
    for tactic in tactics:
        r, g, b = tactic.color
        for sq in tactic.squares:
            rect = renderer.square_rect(sq)
            rel = pygame.Rect(rect.x - renderer.offset_x,
                              rect.y - renderer.offset_y,
                              renderer.sq_size, renderer.sq_size)
            pygame.draw.rect(overlay, (r, g, b, 70), rel)
            pygame.draw.rect(overlay, (r, g, b, 180), rel, width=3)
    surface.blit(overlay, (renderer.offset_x, renderer.offset_y))


def draw_model_labels(surface, white_name: str, black_name: str, fonts, board_turn=None):
    """Render model names above (Black) and below (White) the board."""
    board_left  = 20
    board_right = 20 + BOARD_PX

    # Black label — top strip
    black_bg = (30, 30, 35)
    pygame.draw.rect(surface, black_bg, (board_left, 0, BOARD_PX, LABEL_H))
    piece_dot_color = (180, 180, 180)  # dark pieces
    pygame.draw.circle(surface, piece_dot_color, (board_left + 12, LABEL_H // 2), 7)
    label = fonts["medium"].render(f"Black:  {black_name}", True, TEXT_COLOR)
    surface.blit(label, (board_left + 26, (LABEL_H - label.get_height()) // 2))
    if board_turn == chess.BLACK:
        indicator = fonts["small"].render("▶ thinking", True, ACCENT)
        surface.blit(indicator, (board_right - indicator.get_width() - 6,
                                 (LABEL_H - indicator.get_height()) // 2))

    # White label — bottom strip
    white_y = LABEL_H + BOARD_PX + 20
    white_bg = (50, 50, 45)
    pygame.draw.rect(surface, white_bg, (board_left, white_y, BOARD_PX, LABEL_H))
    pygame.draw.circle(surface, (230, 230, 215), (board_left + 12, white_y + LABEL_H // 2), 7)
    label = fonts["medium"].render(f"White:  {white_name}", True, TEXT_COLOR)
    surface.blit(label, (board_left + 26, white_y + (LABEL_H - label.get_height()) // 2))
    if board_turn == chess.WHITE:
        indicator = fonts["small"].render("▶ thinking", True, ACCENT)
        surface.blit(indicator, (board_right - indicator.get_width() - 6,
                                 white_y + (LABEL_H - indicator.get_height()) // 2))


def draw_sidebar(surface, value, turn_is_white, show_heatmap, heatmap_layer,
                 game_over_msg, auto_play, fonts, board, tactics: list, use_mcts: bool = True):
    sidebar_x = BOARD_PX + 40
    pygame.draw.rect(surface, SIDEBAR_BG, (sidebar_x - 10, 0, SIDEBAR_W + 10, WIN_H))

    draw_value_bar(surface, value, sidebar_x, 20, width=24, height=BOARD_PX - 40,
                   turn_is_white=turn_is_white)

    text_x = sidebar_x + 34

    surface.blit(fonts["large"].render("Chess NN", True, ACCENT), (text_x, 20))

    if auto_play:
        mode_str = "AUTO-PLAY  (Space to stop)"
        mode_color = (255, 200, 50)
    else:
        mode_str = "White to move" if board.turn == chess.WHITE else "Black to move (AI)"
        mode_color = TEXT_COLOR
    surface.blit(fonts["small"].render(mode_str, True, mode_color), (text_x, 56))

    if game_over_msg:
        surface.blit(fonts["medium"].render(game_over_msg, True, (255, 100, 100)), (text_x, 90))

    # Heatmap status
    surface.blit(fonts["small"].render("Heatmap (H):", True, TEXT_COLOR), (text_x, 130))
    status_color = ACCENT if show_heatmap else (150, 150, 150)
    surface.blit(fonts["medium"].render("ON" if show_heatmap else "OFF", True, status_color), (text_x, 150))
    surface.blit(fonts["small"].render(f"Layer: {heatmap_layer + 1}/{NUM_RESIDUAL_BLOCKS}  (keys 1-5)", True, TEXT_COLOR), (text_x, 175))

    # MCTS status line
    mcts_color = ACCENT if use_mcts else (150, 150, 150)
    surface.blit(fonts["small"].render(
        f"MCTS (M):  {'ON  (200 sims)' if use_mcts else 'OFF  (policy only)'}",
        True, mcts_color), (text_x, 195))

    controls = [
        ("Controls:", ACCENT),
        ("Space — start/stop autoplay", TEXT_COLOR),
        ("M  — toggle MCTS search", TEXT_COLOR),
        ("H  — toggle heatmap", TEXT_COLOR),
        ("1-5 — heatmap layer", TEXT_COLOR),
        ("R  — reset & restart", TEXT_COLOR),
        ("ESC — quit", TEXT_COLOR),
    ]
    y = 215
    for text, color in controls:
        surface.blit(fonts["small"].render(text, True, color), (text_x, y))
        y += 22

    if show_heatmap:
        y += 5
        for line, color in [
            ("Heatmap = network attention", (180, 180, 180)),
            ("Blue = low  |  Red = high", (180, 180, 180)),
        ]:
            surface.blit(fonts["small"].render(line, True, color), (text_x, y))
            y += 20

    # Tactics panel
    y += 10
    pygame.draw.line(surface, (80, 80, 90), (text_x, y), (text_x + SIDEBAR_W - 44, y))
    y += 8
    surface.blit(fonts["medium"].render("Position Analysis", True, ACCENT), (text_x, y))
    y += 22

    if not tactics:
        surface.blit(fonts["small"].render("No tactics detected", True, (120, 120, 120)), (text_x, y))
    else:
        for tactic in tactics[:6]:  # cap at 6 so we don't overflow
            r, g, b = tactic.color
            # Coloured dot
            pygame.draw.circle(surface, (r, g, b), (text_x + 5, y + 7), 5)
            # Name label
            name_surf = fonts["small"].render(tactic.name, True, (r, g, b))
            surface.blit(name_surf, (text_x + 14, y))
            y += 18
            # Description — truncate if too wide
            desc = tactic.description
            desc_surf = fonts["small"].render(desc, True, (180, 180, 180))
            if desc_surf.get_width() > SIDEBAR_W - 50:
                while desc_surf.get_width() > SIDEBAR_W - 50 and len(desc) > 4:
                    desc = desc[:-1]
                desc_surf = fonts["small"].render(desc + "…", True, (180, 180, 180))
            surface.blit(desc_surf, (text_x + 14, y))
            y += 20
            if y > WIN_H - 20:
                break


def main():
    import random
    ckpt_a = sys.argv[1] if len(sys.argv) > 1 else "best_model.pt"
    ckpt_b = sys.argv[2] if len(sys.argv) > 2 else "epoch_01.pt"

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"Chess NN — {ckpt_a} vs {ckpt_b}")
    clock = pygame.time.Clock()

    fonts = {
        "large":  pygame.font.SysFont("helvetica", 22, bold=True),
        "medium": pygame.font.SysFont("helvetica", 17),
        "small":  pygame.font.SysFont("helvetica", 14),
    }

    print(f"Loading {ckpt_a}...")
    model_a = load_model(ckpt_a)
    print(f"Loading {ckpt_b}...")
    model_b = load_model(ckpt_b)

    # Randomly assign colors once at startup
    if random.random() < 0.5:
        white_model, black_model = model_a, model_b
        white_name,  black_name  = ckpt_a,   ckpt_b
    else:
        white_model, black_model = model_b, model_a
        white_name,  black_name  = ckpt_b,   ckpt_a

    print(f"White: {white_name}  |  Black: {black_name}")

    # Construct shared inference-only helpers (opening book / Syzygy / tcache).
    # All three degrade to no-ops when their data files are absent, so the
    # viz still runs even without book.bin / syzygy/ / mcts_cache.json.
    from chess_nn.opening_book import OpeningBook
    from chess_nn.syzygy_probe import SyzygyTable
    from chess_nn.transposition import TranspositionCache
    from config import OPENING_BOOK_PATH, SYZYGY_PATH, MCTS_CACHE_PATH
    book = OpeningBook(polyglot_path=OPENING_BOOK_PATH)
    tablebase = SyzygyTable(path=SYZYGY_PATH)
    # Tag the tcache by the WHITE checkpoint's filename so visit counts
    # written by different model versions stay segmented (issue #27). Tag
    # is capped at 16 chars to keep the on-disk JSON compact.
    tcache_tag = os.path.basename(white_name)[:16]
    tcache = TranspositionCache(path=MCTS_CACHE_PATH, tag=tcache_tag)
    print(f"[viz] book={book.loaded_source}  syzygy={'ON' if tablebase.is_available else 'OFF'}  tcache={len(tcache)} positions  tag={tcache_tag!r}")

    white_mcts = MCTS(white_model, num_simulations=200,
                      book=book, tablebase=tablebase, tcache=tcache)
    black_mcts = MCTS(black_model, num_simulations=200,
                      book=book, tablebase=tablebase, tcache=tcache)

    log_session_start(white_name, black_name, use_mcts=True, num_sims=200)
    game_idx = 1
    ply_in_game = 0
    log_game_start(game_idx)
    print(f"[viz] Move log: {MOVE_LOG_PATH}")

    board = chess.Board()
    board_history = deque(maxlen=8)
    board_history.appendleft(board.copy(stack=False))
    # Board sits between the two label strips
    renderer = BoardRenderer(screen, sq_size=SQ_SIZE, offset_x=20, offset_y=LABEL_H + 20)

    last_move       = None
    game_over_msg   = ""
    show_heatmap    = False
    heatmap_layer   = 0
    auto_play       = True   # start immediately in watch mode
    use_mcts        = True
    last_auto_time  = 0
    current_tactics = []
    auto_replay     = True   # automatically start a new game on game-over (logging mode)
    AUTO_REPLAY_DELAY_MS = 2500
    game_over_at_ms = None
    should_reset    = False  # set by R-key or auto-replay trigger; consumed at end of loop tick
    print(f"[viz] auto-replay {'ON' if auto_replay else 'OFF'}  (press L to toggle)")

    active_model = white_model if board.turn == chess.WHITE else black_model
    move_probs, current_value, current_heatmap = run_inference(active_model, board, heatmap_layer)
    current_tactics = detect_tactics(board)

    running = True
    while running:
        clock.tick(30)
        now = pygame.time.get_ticks()

        # Auto-play: each model moves when it's their turn
        if auto_play and not board.is_game_over() and now - last_auto_time > AUTOPLAY_DELAY_MS:
            active_model = white_model if board.turn == chess.WHITE else black_model
            active_mcts  = white_mcts  if board.turn == chess.WHITE else black_mcts
            mover_color  = "W" if board.turn == chess.WHITE else "B"
            mover_name   = white_name if board.turn == chess.WHITE else black_name
            board_before = board.copy()
            pre_move_probs = move_probs
            pre_value = current_value
            last_move, move_probs, current_value, current_heatmap, game_over_msg = \
                push_ai_move(active_model, board, board_history, active_mcts,
                             heatmap_layer, last_move, use_mcts=use_mcts)
            current_tactics = detect_tactics(board)
            last_auto_time = now
            if last_move is not None and last_move in board_before.legal_moves:
                ply_in_game += 1
                try:
                    chosen_san = board_before.san(last_move)
                except Exception:
                    chosen_san = last_move.uci()
                log_move(
                    game_idx, ply_in_game, mover_name, mover_color,
                    last_move, chosen_san, pre_move_probs,
                    pre_value, current_value, use_mcts, board,
                )
            if board.is_game_over():
                log_game_end(game_idx, board, ply_in_game)
                auto_play = False
                game_over_at_ms = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    should_reset = True

                elif event.key == pygame.K_l:
                    auto_replay = not auto_replay
                    print(f"[viz] auto-replay {'ON' if auto_replay else 'OFF'}")

                elif event.key == pygame.K_SPACE:
                    auto_play = not auto_play
                    last_auto_time = now

                elif event.key == pygame.K_m:
                    use_mcts = not use_mcts
                    print(f"MCTS: {'ON (200 sims)' if use_mcts else 'OFF (policy only)'}")

                elif event.key == pygame.K_h:
                    show_heatmap = not show_heatmap

                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    heatmap_layer = min(event.key - pygame.K_1, NUM_RESIDUAL_BLOCKS - 1)
                    if not board.is_game_over():
                        active_model = white_model if board.turn == chess.WHITE else black_model
                        move_probs, current_value, current_heatmap = run_inference(active_model, board, heatmap_layer)

        # Auto-replay: kick off the next game after a brief pause on game-over.
        if (auto_replay and board.is_game_over() and game_over_at_ms is not None
                and now - game_over_at_ms > AUTO_REPLAY_DELAY_MS):
            should_reset = True

        # Reset block — triggered by R-key or auto-replay.
        if should_reset:
            if ply_in_game > 0 and not board.is_game_over():
                log_game_abandoned(game_idx, ply_in_game)
            game_idx += 1
            ply_in_game = 0
            log_game_start(game_idx)
            board = chess.Board()
            board_history = deque(maxlen=8)
            board_history.appendleft(board.copy(stack=False))
            white_mcts.reset()
            black_mcts.reset()
            last_move = None
            game_over_msg = ""
            auto_play = True
            last_auto_time = now
            current_tactics = []
            active_model = white_model
            move_probs, current_value, current_heatmap = run_inference(active_model, board, heatmap_layer)
            current_tactics = detect_tactics(board)
            game_over_at_ms = None
            should_reset = False

        # --- Draw ---
        screen.fill(BG)
        if show_heatmap and current_heatmap is not None:
            draw_heatmap(screen, current_heatmap, renderer)
        renderer.draw(board, None, last_move, [], flip=False)
        if current_tactics:
            draw_tactic_highlights(screen, current_tactics, renderer)
        draw_move_arrows(screen, move_probs, renderer)
        draw_model_labels(screen, white_name, black_name, fonts,
                          board.turn if not board.is_game_over() else None)
        draw_sidebar(screen, current_value, board.turn == chess.WHITE,
                     show_heatmap, heatmap_layer, game_over_msg, auto_play, fonts, board,
                     current_tactics, use_mcts)
        if game_over_msg:
            draw_game_over_overlay(screen, game_over_msg, fonts)
        pygame.display.flip()

    # Persist any cache state accumulated during this session so the next
    # viz/engine run gets a head start on the positions we just searched.
    try:
        tcache.save()
    except Exception:
        pass
    pygame.quit()


if __name__ == "__main__":
    main()
