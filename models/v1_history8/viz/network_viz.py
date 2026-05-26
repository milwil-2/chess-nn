"""
Neural Network Visualizer — a separate window showing every layer firing in real time.

Reads the state written by app.py from /tmp/chess_nn_state.npz and redraws
whenever the file changes. Run this alongside app.py:

    python viz/app.py &
    python viz/network_viz.py

Layout (1500 × 860):
  ┌─────────────────────────────────────────────────────────┐
  │  INPUT PLANES (18 small 8×8 grids, one per feature)    │
  ├─────────────────────────────────────────────────────────┤
  │  NETWORK BODY: Initial Conv → Res1 → Res2 → ... → Res5 │
  │  Each layer shown as: average heatmap + top 8 filters   │
  ├─────────────────────────────────────────────────────────┤
  │  OUTPUT: Policy (move bars) │ Value (gauge)             │
  └─────────────────────────────────────────────────────────┘

What you're seeing:
  - Input planes: binary grids showing where each piece type is
  - Conv layers: what patterns the filters detect (edges, piece clusters, etc.)
  - Residual blocks: increasingly abstract features (tactics, structure, king safety)
  - Policy: which moves the network thinks are good (before MCTS)
  - Value: who the network thinks is winning
"""

import sys
import os
import time
import numpy as np
import pygame
import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_nn.move_encoding import index_to_move

STATE_FILE = "/tmp/chess_nn_state.npz"

# Window
W, H = 1500, 860
BG = (18, 18, 24)
PANEL_BG = (28, 28, 36)
TEXT = (210, 210, 220)
DIM_TEXT = (120, 120, 140)
ACCENT = (80, 180, 120)
BORDER = (60, 60, 80)

INPUT_PLANE_LABELS = [
    "W.Pawn", "W.Knight", "W.Bishop", "W.Rook", "W.Queen", "W.King",
    "B.Pawn", "B.Knight", "B.Bishop", "B.Rook", "B.Queen", "B.King",
    "W.K-cast", "W.Q-cast", "B.K-cast", "B.Q-cast", "En passant", "To move",
]


def normalise(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def activation_color(val: float) -> tuple:
    """Map [0,1] → blue→cyan→green→yellow→red colour ramp."""
    val = max(0.0, min(1.0, val))
    if val < 0.25:
        t = val / 0.25
        return (0, int(t * 180), 255)
    elif val < 0.5:
        t = (val - 0.25) / 0.25
        return (0, 180 + int(t * 75), int(255 * (1 - t)))
    elif val < 0.75:
        t = (val - 0.5) / 0.25
        return (int(t * 255), 255, 0)
    else:
        t = (val - 0.75) / 0.25
        return (255, int(255 * (1 - t)), 0)


def draw_grid(surface, data_2d: np.ndarray, x: int, y: int, cell: int,
              border_color=BORDER, glow: bool = False):
    """Draw an 8×8 (or any) 2D numpy array as a coloured grid."""
    norm = normalise(data_2d)
    rows, cols = norm.shape
    for r in range(rows):
        for c in range(cols):
            val = float(norm[r, c])
            color = activation_color(val)
            rect = pygame.Rect(x + c * cell, y + r * cell, cell - 1, cell - 1)
            pygame.draw.rect(surface, color, rect)
    if glow:
        glow_surf = pygame.Surface((cols * cell, rows * cell), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*ACCENT, 40), (0, 0, cols * cell, rows * cell))
        surface.blit(glow_surf, (x, y))
    outline = pygame.Rect(x - 1, y - 1, cols * cell + 2, rows * cell + 2)
    pygame.draw.rect(surface, border_color, outline, 1)


def draw_label(surface, font, text: str, x: int, y: int, color=DIM_TEXT, center=False):
    surf = font.render(text, True, color)
    if center:
        x -= surf.get_width() // 2
    surface.blit(surf, (x, y))


def draw_section_header(surface, font_bold, text: str, x: int, y: int, width: int):
    pygame.draw.line(surface, BORDER, (x, y + 8), (x + width, y + 8), 1)
    label = font_bold.render(f"  {text}  ", True, ACCENT)
    surface.blit(label, (x + 10, y))


def draw_input_planes(surface, fonts, board_tensor: np.ndarray, x0: int, y0: int):
    """Draw the 18 input planes as small labelled 8×8 grids."""
    draw_section_header(surface, fonts["bold"], "INPUT PLANES  (what the network sees)", x0, y0, W - 40)

    cell = 7        # pixels per chess square in these tiny grids
    grid_w = 8 * cell
    col_gap = grid_w + 36
    row_gap = 8 * cell + 28
    cols_per_row = 9

    for i in range(18):
        col = i % cols_per_row
        row = i // cols_per_row
        px = x0 + col * col_gap
        py = y0 + 22 + row * row_gap

        plane = board_tensor[i]  # (8, 8)
        # Flip vertically so rank 8 is at top (display convention)
        plane_display = plane[::-1, :]

        # Highlight active planes with a subtle glow
        is_active = plane.max() > 0
        draw_grid(surface, plane_display, px, py, cell, glow=is_active)

        label = INPUT_PLANE_LABELS[i]
        color = TEXT if is_active else DIM_TEXT
        draw_label(surface, fonts["tiny"], label, px + grid_w // 2, py + 8 * cell + 3,
                   color=color, center=True)


def draw_layer_panel(surface, fonts, activations: np.ndarray, x0: int, y0: int,
                     label: str, panel_w: int = 200, selected: bool = False):
    """
    Draw one layer's activations:
      - Large average heatmap (8×8, all filters averaged)
      - 8 individual filters shown as tiny grids below
    """
    border = ACCENT if selected else BORDER
    pygame.draw.rect(surface, PANEL_BG, (x0, y0, panel_w, 210))
    pygame.draw.rect(surface, border, (x0, y0, panel_w, 210), 1)

    # Label at top
    draw_label(surface, fonts["small_bold"], label, x0 + panel_w // 2, y0 + 5,
               color=ACCENT if selected else TEXT, center=True)

    # Average heatmap — what this whole layer "notices" on average
    avg = activations[0].mean(axis=0)  # (8, 8) — mean across all filters
    avg_display = avg[::-1, :]
    cell = 14
    avg_x = x0 + (panel_w - 8 * cell) // 2
    draw_grid(surface, avg_display, avg_x, y0 + 22, cell)

    # Stats
    draw_label(surface, fonts["tiny"],
               f"mean={activations[0].mean():.2f}  max={activations[0].max():.2f}",
               x0 + panel_w // 2, y0 + 22 + 8 * cell + 4, center=True)

    # 8 individual filters (most-activated ones)
    n_filters = activations[0].shape[0]
    # Pick the 8 filters with highest max activation
    filter_maxes = activations[0].max(axis=(1, 2))
    top8_idx = np.argsort(filter_maxes)[-8:][::-1]

    mini_cell = 5
    mini_w = 8 * mini_cell + 2
    filters_y = y0 + 22 + 8 * cell + 20
    for fi, f_idx in enumerate(top8_idx):
        fx = x0 + 6 + fi * (mini_w + 2)
        fdata = activations[0][f_idx][::-1, :]
        draw_grid(surface, fdata, fx, filters_y, mini_cell, border_color=(50, 50, 70))


def draw_policy_panel(surface, fonts, policy_logits: np.ndarray,
                      board_tensor: np.ndarray, x0: int, y0: int, width: int):
    """Show the top 12 moves with probability bars, plus an 8×8 destination heatmap."""
    draw_section_header(surface, fonts["bold"], "POLICY HEAD  (move probabilities)", x0, y0, width)

    # Reconstruct a board from the tensor to decode moves
    # (We can't perfectly reconstruct but we can show indices + rough SAN)
    logits = policy_logits.squeeze()

    # Softmax over all 4672 outputs
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()

    top_indices = np.argsort(probs)[-12:][::-1]

    bar_x = x0 + 10
    bar_y = y0 + 22
    bar_max_w = width - 20
    bar_h = 16

    for rank, idx in enumerate(top_indices):
        prob = float(probs[idx])
        src_sq = idx // 73
        src_file = src_sq % 8
        src_rank = src_sq // 8
        src_name = f"{chr(ord('a') + src_file)}{src_rank + 1}"

        bar_w = int(prob * bar_max_w * 15)  # Scale up for visibility
        bar_w = min(bar_w, bar_max_w - 80)

        color = activation_color(prob * 15)
        pygame.draw.rect(surface, color, (bar_x + 60, bar_y, bar_w, bar_h - 2))
        pygame.draw.rect(surface, BORDER, (bar_x + 60, bar_y, bar_max_w - 80, bar_h - 2), 1)

        draw_label(surface, fonts["tiny"], f"{src_name}→?", bar_x, bar_y + 2, color=TEXT)
        draw_label(surface, fonts["tiny"], f"{prob:.3f}", bar_x + bar_max_w - 75, bar_y + 2, color=DIM_TEXT)

        bar_y += bar_h + 2

    # 8×8 destination heatmap — where does the network want to move pieces TO?
    dest_heat = np.zeros((8, 8), dtype=np.float32)
    for idx in range(4672):
        dest_sq_idx = idx % 73
        if dest_sq_idx < 56:  # Queen-style move
            src_sq = idx // 73
            src_rank_enc = src_sq // 8
            src_file_enc = src_sq % 8
            dir_idx = dest_sq_idx // 7
            dist = (dest_sq_idx % 7) + 1
            dirs = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
            dr, dc = dirs[dir_idx]
            dest_r = src_rank_enc + dr * dist
            dest_c = src_file_enc + dc * dist
            if 0 <= dest_r <= 7 and 0 <= dest_c <= 7:
                dest_heat[dest_r, dest_c] += float(probs[idx])

    heat_x = bar_x + bar_max_w - 70
    heat_y = y0 + 22
    draw_label(surface, fonts["tiny"], "Dest. heat", heat_x + 36, heat_y - 14, center=True)
    draw_grid(surface, dest_heat[::-1, :], heat_x, heat_y, 9)


def draw_value_panel(surface, fonts, value: float, x0: int, y0: int, width: int, height: int):
    """Show the value head output as a large gauge."""
    draw_section_header(surface, fonts["bold"], "VALUE HEAD  (who is winning)", x0, y0, width)

    # Gauge bar (vertical)
    bar_x = x0 + width // 2 - 15
    bar_y = y0 + 30
    bar_h = height - 50
    bar_w = 30

    pygame.draw.rect(surface, (30, 30, 30), (bar_x, bar_y, bar_w, bar_h))

    fill_ratio = (value + 1) / 2.0
    fill_h = int(fill_ratio * bar_h)
    # Bottom = black wins (dark), top = white wins (light)
    pygame.draw.rect(surface, (220, 220, 220),
                     (bar_x, bar_y + bar_h - fill_h, bar_w, fill_h))

    # Centre line
    pygame.draw.line(surface, (100, 100, 100),
                     (bar_x, bar_y + bar_h // 2), (bar_x + bar_w, bar_y + bar_h // 2), 1)

    # Labels
    draw_label(surface, fonts["tiny"], "White", bar_x + bar_w + 6, bar_y, color=TEXT)
    draw_label(surface, fonts["tiny"], "Black", bar_x + bar_w + 6, bar_y + bar_h - 12, color=TEXT)

    sign = "+" if value >= 0 else ""
    val_str = f"{sign}{value:.3f}"
    color = TEXT if abs(value) < 0.3 else ((220, 220, 255) if value > 0 else (255, 180, 180))
    val_surf = fonts["medium"].render(val_str, True, color)
    surface.blit(val_surf, (bar_x + bar_w // 2 - val_surf.get_width() // 2,
                             bar_y + bar_h + 8))

    who = "Balanced" if abs(value) < 0.1 else ("White ahead" if value > 0 else "Black ahead")
    draw_label(surface, fonts["tiny"], who, x0 + width // 2, bar_y + bar_h + 28,
               color=DIM_TEXT, center=True)


def load_state():
    """Load the latest state written by app.py. Returns None if not available."""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        data = np.load(STATE_FILE, allow_pickle=False)
        return dict(data)
    except Exception:
        return None


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Chess NN — Network Internals")
    clock = pygame.time.Clock()

    fonts = {
        "bold":       pygame.font.SysFont("helvetica", 15, bold=True),
        "small_bold": pygame.font.SysFont("helvetica", 13, bold=True),
        "medium":     pygame.font.SysFont("helvetica", 20, bold=True),
        "small":      pygame.font.SysFont("helvetica", 13),
        "tiny":       pygame.font.SysFont("helvetica", 11),
    }

    state = None
    last_mtime = 0
    POLL_INTERVAL = 0.4  # seconds between file checks
    last_poll = 0

    print(f"Waiting for app.py to write state to {STATE_FILE}...")
    print("Make a move in the chess app to see the network fire.")

    running = True
    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Poll for new state
        now = time.time()
        if now - last_poll > POLL_INTERVAL:
            last_poll = now
            try:
                mtime = os.path.getmtime(STATE_FILE)
                if mtime != last_mtime:
                    new_state = load_state()
                    if new_state is not None:
                        state = new_state
                        last_mtime = mtime
            except FileNotFoundError:
                pass

        screen.fill(BG)

        if state is None:
            msg = fonts["medium"].render(
                "Waiting for chess app... make a move to see the network fire.", True, DIM_TEXT)
            screen.blit(msg, (W // 2 - msg.get_width() // 2, H // 2))
            pygame.display.flip()
            continue

        # Extract data
        board_tensor = state["board_tensor"]          # (18, 8, 8)
        value        = float(state["value"].squeeze())

        # --- SECTION 1: Input planes ---
        SEC1_Y = 10
        draw_input_planes(screen, fonts, board_tensor, 20, SEC1_Y)

        # --- SECTION 2: Network body layers ---
        SEC2_Y = 10 + 2 * (8 * 7 + 28) + 30  # Two rows of input planes + margin

        draw_section_header(screen, fonts["bold"],
                            "NETWORK BODY  (residual blocks — blue=low activation, red=high)",
                            20, SEC2_Y, W - 40)

        layer_names = ["Init Conv"] + [f"Res Block {i+1}" for i in range(5)]
        layer_keys  = ["input_conv"] + [f"res_block_{i}" for i in range(5)]
        panel_w = (W - 40) // 6 - 8

        for li, (name, key) in enumerate(zip(layer_names, layer_keys)):
            if key not in state:
                continue
            acts = state[key]   # (1, filters, 8, 8)
            if acts.ndim == 3:
                acts = acts[np.newaxis]
            px = 20 + li * (panel_w + 8)
            draw_layer_panel(screen, fonts, acts, px, SEC2_Y + 20, name,
                             panel_w=panel_w, selected=(li == 0))

        # --- SECTION 3: Outputs ---
        SEC3_Y = SEC2_Y + 240
        out_y = SEC3_Y

        # Policy panel (left 2/3)
        policy_w = int((W - 40) * 0.72)
        if "policy_logits" in state:
            draw_policy_panel(screen, fonts, state["policy_logits"],
                              board_tensor, 20, out_y, policy_w)

        # Value panel (right 1/3)
        val_x = 20 + policy_w + 10
        val_w = W - 40 - policy_w - 10
        draw_value_panel(screen, fonts, value, val_x, out_y, val_w, H - out_y - 20)

        # Footer
        footer = fonts["tiny"].render(
            "Each grid = 8×8 board  |  Colour = activation strength  |  "
            "Brighter red = network is more 'excited' about that square", True, DIM_TEXT)
        screen.blit(footer, (20, H - 18))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
