import os
import json
import torch
from config import CHECKPOINT_DIR, LOG_DIR


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
    return total


def save_checkpoint(model, optimizer, epoch: int, loss: float, filename: str = "checkpoint.pt"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(path: str, model, optimizer=None):
    checkpoint = torch.load(path, map_location=get_device(), weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (loss={checkpoint['loss']:.4f})")
    return checkpoint["epoch"], checkpoint["loss"]


def log_training_step(epoch: int, step: int, policy_loss: float, value_loss: float, total_loss: float):
    """Append a training step to the JSON log so the viz app can read it live."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "training_log.json")

    entry = {
        "epoch": epoch,
        "step": step,
        "policy_loss": round(policy_loss, 6),
        "value_loss": round(value_loss, 6),
        "total_loss": round(total_loss, 6),
    }

    # Read existing entries, append, write back
    entries = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError:
                entries = []
    entries.append(entry)
    with open(log_path, "w") as f:
        json.dump(entries, f)
