"""
WebSocket server that watches the state file written by app.py and
broadcasts activation data to connected browsers in real time.

Run alongside the chess app:
    python viz/app.py &
    python viz/web_server.py

Then open http://localhost:8000 in your browser.
"""

import os
import sys
import asyncio
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

STATE_FILE = "/tmp/chess_nn_state.npz"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Connected WebSocket clients
clients: list[WebSocket] = []


def extract_layer_activations(state: dict, n_nodes: int = 16) -> dict:
    """
    Reduce full activation tensors to n_nodes representative values per layer.
    We pick the n_nodes filters with highest mean activation — the most "excited" ones.
    This keeps the data small enough to stream efficiently.
    """
    layers = {}

    # Input: 18 planes, each summarised by its mean activation
    if "board_tensor" in state:
        bt = state["board_tensor"]  # (18, 8, 8)
        layers["input"] = [float(bt[i].mean()) for i in range(18)]

    # Body layers: pick top-n_nodes filters by mean activation
    body_keys = ["input_conv"] + [f"res_block_{i}" for i in range(5)]
    body_names = ["conv"] + [f"res{i+1}" for i in range(5)]
    for key, name in zip(body_keys, body_names):
        if key not in state:
            continue
        acts = state[key]  # (1, 128, 8, 8)
        if acts.ndim == 4:
            acts = acts[0]  # (128, 8, 8)
        filter_means = acts.mean(axis=(1, 2))  # (128,)
        top_idx = np.argsort(filter_means)[-n_nodes:][::-1]
        # Normalise to [0, 1]
        vals = filter_means[top_idx]
        mn, mx = vals.min(), vals.max()
        if mx - mn > 1e-6:
            vals = (vals - mn) / (mx - mn)
        else:
            vals = np.zeros_like(vals)
        layers[name] = vals.tolist()

    # Policy: top 10 move probabilities
    if "policy_logits" in state:
        logits = state["policy_logits"].squeeze()
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        top10 = np.argsort(probs)[-10:][::-1]
        layers["policy"] = [float(probs[i]) for i in top10]

        # Top move labels (source square names)
        move_labels = []
        for idx in top10:
            src_sq = idx // 73
            src_file = src_sq % 8
            src_rank = src_sq // 8
            move_labels.append(f"{chr(ord('a') + src_file)}{src_rank + 1}")
        layers["policy_labels"] = move_labels

    # Value
    if "value" in state:
        layers["value"] = [float(state["value"].squeeze())]

    return layers


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return html_path.read_text()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    print(f"Client connected ({len(clients)} total)")
    try:
        # Send current state immediately on connect if available
        if os.path.exists(STATE_FILE):
            try:
                state = dict(np.load(STATE_FILE, allow_pickle=False))
                payload = extract_layer_activations(state)
                await websocket.send_text(json.dumps(payload))
            except Exception:
                pass
        # Keep connection alive, wait for client messages
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.remove(websocket)
        print(f"Client disconnected ({len(clients)} remaining)")


async def watch_state_file():
    """Background task: poll the state file and broadcast when it changes."""
    last_mtime = 0
    print(f"Watching {STATE_FILE} for changes...")
    while True:
        await asyncio.sleep(0.3)
        if not os.path.exists(STATE_FILE):
            continue
        try:
            mtime = os.path.getmtime(STATE_FILE)
            if mtime == last_mtime:
                continue
            last_mtime = mtime
            state = dict(np.load(STATE_FILE, allow_pickle=False))
            payload = extract_layer_activations(state)
            msg = json.dumps(payload)
            dead = []
            for ws in clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.remove(ws)
        except Exception as e:
            pass


@app.on_event("startup")
async def startup():
    asyncio.create_task(watch_state_file())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
