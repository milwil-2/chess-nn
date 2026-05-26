"""
Overlays a neural network activation heatmap on the chess board.

After a forward pass we grab one residual block's output (shape: 128 filters × 8 × 8),
average across filters to get a single 8×8 grid, then colour each square by activation
strength: blue (low) → red (high). This shows what the network "pays attention to."
"""

import pygame
import numpy as np
import chess


def draw_heatmap(surface: pygame.Surface, heatmap: np.ndarray,
                 renderer, flip: bool = False, alpha: int = 140):
    """
    heatmap: (8, 8) numpy array of float values (any range — we normalise)
    """
    if heatmap is None:
        return

    # Normalise to [0, 1]
    mn, mx = heatmap.min(), heatmap.max()
    if mx - mn < 1e-6:
        return
    norm = (heatmap - mn) / (mx - mn)

    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

    for rank in range(8):
        for file in range(8):
            val = float(norm[7 - rank, file])  # numpy rank 0 = rank 1 on board

            # Blue (cold) → Red (hot) colour map
            r = int(val * 220)
            b = int((1 - val) * 220)
            g = 0

            sq = chess.square(file, rank)
            rect = renderer.square_rect(sq, flip)
            rel = pygame.Rect(rect.x, rect.y, rect.width, rect.height)
            pygame.draw.rect(overlay, (r, g, b, alpha), rel)

    surface.blit(overlay, (0, 0))
