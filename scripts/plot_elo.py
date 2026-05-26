#!/usr/bin/env python3
"""
ASCII plot of elo_history.csv. No external dependencies.

Usage:
    python scripts/plot_elo.py [path/to/elo_history.csv]

Defaults to models/v3_vast/logs/elo_history.csv.
"""

import csv
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(REPO_ROOT, "models", "v3_vast", "logs", "elo_history.csv")


def load_rows(path):
    if not os.path.exists(path):
        print(f"ERROR: {path} does not exist", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print(f"No rows in {path}", file=sys.stderr)
        sys.exit(1)
    return rows


def ascii_plot(rows, width=70, height=18):
    """Draw an ASCII line chart. X = row index, Y = Elo."""
    elos = [float(r["elo"]) for r in rows]
    lows = [float(r["ci_low"]) for r in rows]
    highs = [float(r["ci_high"]) for r in rows]

    ymin = min(lows) - 20
    ymax = max(highs) + 20
    yrange = max(1.0, ymax - ymin)

    n = len(rows)
    # Distribute x positions evenly across `width`
    if n == 1:
        xs = [width // 2]
    else:
        xs = [round(i * (width - 1) / (n - 1)) for i in range(n)]

    grid = [[" "] * width for _ in range(height)]

    def y_to_row(y):
        # Higher y → lower row index (top of grid)
        frac = (y - ymin) / yrange
        return int(round((1 - frac) * (height - 1)))

    # CI bars
    for i, (lo, hi) in enumerate(zip(lows, highs)):
        x = xs[i]
        r_lo = y_to_row(lo)
        r_hi = y_to_row(hi)
        if r_hi > r_lo:
            r_lo, r_hi = r_hi, r_lo
        for r in range(r_hi, r_lo + 1):
            if grid[r][x] == " ":
                grid[r][x] = "|"

    # Data points and connecting line
    last_x = last_r = None
    for i, elo in enumerate(elos):
        x = xs[i]
        r = y_to_row(elo)
        if last_x is not None:
            # Draw a connecting segment
            dx = x - last_x
            dr = r - last_r
            steps = max(1, max(abs(dx), abs(dr)))
            for s in range(1, steps):
                ix = last_x + round(dx * s / steps)
                ir = last_r + round(dr * s / steps)
                if 0 <= ir < height and 0 <= ix < width and grid[ir][ix] in (" ", "|"):
                    grid[ir][ix] = "."
        grid[r][x] = "*"
        last_x, last_r = x, r

    # Y-axis labels (4 ticks)
    ticks = 4
    y_labels = []
    for t in range(ticks + 1):
        y_val = ymin + (yrange * t / ticks)
        y_labels.append((y_to_row(y_val), f"{y_val:>6.0f}"))

    # Compose output
    print()
    print(f"Elo history ({n} measurement{'s' if n != 1 else ''})")
    print("=" * (width + 10))

    label_by_row = {}
    for r, lab in y_labels:
        label_by_row.setdefault(r, lab)

    for r in range(height):
        prefix = label_by_row.get(r, "      ")
        print(f"{prefix} | {''.join(grid[r])}")

    # X-axis
    print(" " * 6 + " +" + "-" * width)

    # X-axis labels: first and last timestamps
    ts0 = rows[0]["timestamp"][:10]
    ts1 = rows[-1]["timestamp"][:10]
    print(" " * 8 + ts0.ljust(width - len(ts1)) + ts1)
    print()

    # Table
    print(f"{'timestamp':<20} {'hash':<14} {'games':>6} {'tc':>10} "
          f"{'elo':>7} {'ci_lo':>7} {'ci_hi':>7} {'decisive':>9} {'plies':>6}")
    for r in rows:
        print(f"{r['timestamp']:<20} {r['checkpoint_hash']:<14} "
              f"{r['games']:>6} {r['tc']:>10} {float(r['elo']):>7.0f} "
              f"{float(r['ci_low']):>7.0f} {float(r['ci_high']):>7.0f} "
              f"{float(r['decisive_rate']):>9.2f} {float(r['mean_plies']):>6.1f}")
    print()

    if n >= 2:
        delta = elos[-1] - elos[-2]
        flag = " (REGRESSION)" if delta < -50 else " (improvement)" if delta > 50 else ""
        print(f"Latest delta vs prior: {delta:+.0f} Elo{flag}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    rows = load_rows(path)
    ascii_plot(rows)


if __name__ == "__main__":
    main()
