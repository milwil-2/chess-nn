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
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
import pygame
import chess
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHECKPOINT_DIR, NUM_RESIDUAL_BLOCKS
from chess_nn.model import ChessNet
from chess_nn.board_encoding import board_to_tensor
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
SIDEBAR_W = 280
WIN_W     = BOARD_PX + SIDEBAR_W + 40
WIN_H     = BOARD_PX + 40

BG         = (40, 40, 40)
SIDEBAR_BG = (50, 50, 55)
TEXT_COLOR = (220, 220, 220)
ACCENT     = (100, 180, 100)

# Auto-play delay in milliseconds between moves
AUTOPLAY_DELAY_MS = 800


def load_model() -> ChessNet:
    model = ChessNet()
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model)
        print("Model loaded from checkpoint.")
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


def run_inference(model: ChessNet, board: chess.Board, heatmap_layer: int):
    """Run the model on the current board. Returns (move_probs, value, heatmap)."""
    import numpy as np
    tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).float()
    all_acts = model.get_all_activations(tensor)
    move_probs = policy_to_moves(all_acts["policy_logits"].squeeze(), board, top_k=8)
    heatmap = model.get_activations(tensor, layer_index=heatmap_layer).numpy()

    # Write state for the network visualizer window to read
    try:
        save_dict = {k: v for k, v in all_acts.items()}
        save_dict["board_tensor"] = board_to_tensor(board)
        np.savez_compressed(STATE_FILE, **save_dict)
    except Exception:
        pass

    return move_probs, float(all_acts["value"].squeeze()), heatmap


def push_ai_move(model, board, heatmap_layer, last_move, temperature=0.5, use_mcts=True):
    """Let the AI make a move. Returns updated (last_move, move_probs, value, heatmap, game_over_msg)."""
    if use_mcts:
        mcts = MCTS(model, num_simulations=200)
        ai_move = mcts.search(board, temperature=0)
    else:
        ai_move = select_move(model, board, temperature=temperature)
    if ai_move in board.legal_moves:
        board.push(ai_move)
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
    sub_surf   = fonts["medium"].render("Press R to play again", True, (200, 200, 200))

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


def draw_sidebar(surface, value, turn_is_white, show_heatmap, heatmap_layer,
                 game_over_msg, auto_play, fonts, board, tactics: list):
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

    controls = [
        ("Controls:", ACCENT),
        ("Click — select & move", TEXT_COLOR),
        ("A  — accept suggestion", TEXT_COLOR),
        ("Space — auto-play AI vs AI", TEXT_COLOR),
        ("H  — toggle heatmap", TEXT_COLOR),
        ("1-5 — heatmap layer", TEXT_COLOR),
        ("R  — reset game", TEXT_COLOR),
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
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Chess Neural Network")
    clock = pygame.time.Clock()

    fonts = {
        "large":  pygame.font.SysFont("helvetica", 22, bold=True),
        "medium": pygame.font.SysFont("helvetica", 17),
        "small":  pygame.font.SysFont("helvetica", 14),
    }

    model = load_model()
    board = chess.Board()
    renderer = BoardRenderer(screen, sq_size=SQ_SIZE, offset_x=20, offset_y=20)

    selected_sq     = None
    legal_targets   = []
    last_move       = None
    game_over_msg   = ""
    show_heatmap    = False
    heatmap_layer   = 0
    auto_play       = False
    last_auto_time  = 0
    current_tactics = []

    move_probs, current_value, current_heatmap = run_inference(model, board, heatmap_layer)
    current_tactics = detect_tactics(board)

    running = True
    while running:
        clock.tick(30)
        now = pygame.time.get_ticks()

        # Auto-play: trigger a move every AUTOPLAY_DELAY_MS
        if auto_play and not board.is_game_over() and now - last_auto_time > AUTOPLAY_DELAY_MS:
            last_move, move_probs, current_value, current_heatmap, game_over_msg = \
                push_ai_move(model, board, heatmap_layer, last_move)
            current_tactics = detect_tactics(board)
            last_auto_time = now
            if board.is_game_over():
                auto_play = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    board = chess.Board()
                    selected_sq, legal_targets, last_move = None, [], None
                    game_over_msg = ""
                    auto_play = False
                    current_tactics = []
                    move_probs, current_value, current_heatmap = run_inference(model, board, heatmap_layer)
                    current_tactics = detect_tactics(board)

                elif event.key == pygame.K_SPACE:
                    auto_play = not auto_play
                    last_auto_time = now
                    selected_sq, legal_targets = None, []

                elif event.key == pygame.K_a:
                    if not board.is_game_over() and move_probs:
                        suggested_move = move_probs[0][0]
                        if suggested_move in board.legal_moves:
                            board.push(suggested_move)
                            last_move = suggested_move
                            selected_sq, legal_targets = None, []
                            if board.is_game_over():
                                game_over_msg = game_over_description(board)
                                move_probs, current_value, current_heatmap = [], 0.0, None
                                current_tactics = []
                            else:
                                if board.turn == chess.BLACK:
                                    move_probs, current_value, current_heatmap = run_inference(model, board, heatmap_layer)
                                    last_move, move_probs, current_value, current_heatmap, game_over_msg = \
                                        push_ai_move(model, board, heatmap_layer, last_move)
                                else:
                                    move_probs, current_value, current_heatmap = run_inference(model, board, heatmap_layer)
                                current_tactics = detect_tactics(board)

                elif event.key == pygame.K_h:
                    show_heatmap = not show_heatmap

                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    heatmap_layer = min(event.key - pygame.K_1, NUM_RESIDUAL_BLOCKS - 1)
                    if not board.is_game_over():
                        move_probs, current_value, current_heatmap = run_inference(model, board, heatmap_layer)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if auto_play or board.is_game_over() or board.turn == chess.BLACK:
                    continue

                clicked_sq = renderer.pixel_to_square(event.pos[0], event.pos[1])
                if clicked_sq is None:
                    selected_sq, legal_targets = None, []
                    continue

                if selected_sq is not None and clicked_sq in legal_targets:
                    move = chess.Move(selected_sq, clicked_sq)
                    piece = board.piece_at(selected_sq)
                    if piece and piece.piece_type == chess.PAWN:
                        back_rank = 7 if board.turn == chess.WHITE else 0
                        if chess.square_rank(clicked_sq) == back_rank:
                            move = chess.Move(selected_sq, clicked_sq, promotion=chess.QUEEN)

                    if move in board.legal_moves:
                        board.push(move)
                        last_move = move
                        selected_sq, legal_targets = None, []

                        if board.is_game_over():
                            game_over_msg = game_over_description(board)
                            move_probs, current_value, current_heatmap = [], 0.0, None
                            current_tactics = []
                        else:
                            move_probs, current_value, current_heatmap = run_inference(model, board, heatmap_layer)
                            last_move, move_probs, current_value, current_heatmap, game_over_msg = \
                                push_ai_move(model, board, heatmap_layer, last_move)
                            current_tactics = detect_tactics(board)
                else:
                    piece = board.piece_at(clicked_sq)
                    if piece and piece.color == board.turn:
                        selected_sq = clicked_sq
                        legal_targets = [m.to_square for m in board.legal_moves
                                         if m.from_square == clicked_sq]
                    else:
                        selected_sq, legal_targets = None, []

        # --- Draw ---
        screen.fill(BG)
        if show_heatmap and current_heatmap is not None:
            draw_heatmap(screen, current_heatmap, renderer)
        renderer.draw(board, selected_sq, last_move, legal_targets, flip=False)
        if current_tactics:
            draw_tactic_highlights(screen, current_tactics, renderer)
        draw_move_arrows(screen, move_probs, renderer)
        draw_sidebar(screen, current_value, board.turn == chess.WHITE,
                     show_heatmap, heatmap_layer, game_over_msg, auto_play, fonts, board,
                     current_tactics)
        if game_over_msg:
            draw_game_over_overlay(screen, game_over_msg, fonts)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
