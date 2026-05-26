"""
Create a new model variant by copying the template.

Usage:
  python new_model.py <name>

Example:
  python new_model.py v2_wider_heads
  → Creates models/v2_wider_heads/ from template/
  → Edit models/v2_wider_heads/config.py to customize the architecture
  → Run from inside that directory:
      cd models/v2_wider_heads
      python run.py supervised
      python run.py rl
"""

import os
import shutil
import sys

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

_SKIP = {"__pycache__", "*.pyc", "checkpoints", "logs", ".venv"}


def _ignore(src, names):
    return {n for n in names if n in _SKIP or n.endswith(".pyc")}


def main():
    if len(sys.argv) != 2 or sys.argv[1].startswith("-"):
        print(__doc__)
        sys.exit(1)

    name = sys.argv[1]
    dest = os.path.join(MODELS_DIR, name)

    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        sys.exit(1)

    print(f"Copying template → models/{name} ...")
    shutil.copytree(TEMPLATE_DIR, dest, ignore=_ignore)

    for subdir in ("checkpoints", "logs", os.path.join("data", "raw"), os.path.join("data", "processed")):
        os.makedirs(os.path.join(dest, subdir), exist_ok=True)

    print(f"Created: models/{name}/")
    print(f"""
Next steps:
  1. Edit  models/{name}/config.py   — tweak architecture or hyperparameters
  2. cd models/{name}
  3. python run.py supervised         — train on Lichess data
  4. python run.py rl                 — self-play RL loop

To match against another model (from repo root):
  python match.py models/{name} models/<other>
""")


if __name__ == "__main__":
    main()
