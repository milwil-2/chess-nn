"""
Neural network architecture: a CNN with residual blocks and two output heads.

Architecture summary:
  Input (105, 8, 8)
    → Initial conv layer
    → 10 residual blocks with Squeeze-and-Excitation (body)
    → Policy head  (logits over 4672 moves)
    → Value head   (Win/Draw/Loss logits — 3 classes)

Improvements over baseline:
  SE blocks:   Channel attention inside each residual block (Lc0-validated).
  WDL head:    3-class value output instead of scalar tanh. Lets the network
               reason about draws separately, critical for chess.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_RESIDUAL_BLOCKS, NUM_FILTERS, INPUT_PLANES, POLICY_OUTPUT_SIZE


def wdl_to_scalar(value_logits: torch.Tensor) -> float:
    """Convert WDL logits (B=1, 3) → scalar ∈ [-1, 1] for MCTS/evaluate.
    win_prob - loss_prob gives the expected score from the current player's view.
    """
    probs = torch.softmax(value_logits.squeeze(0), dim=0)
    return float(probs[0].item() - probs[2].item())


class SqueezeExcitation(nn.Module):
    """
    Channel recalibration module (Hu et al., 2018), as used in Lc0.

    Learns which feature maps matter for a given position by squeezing
    spatial info into a channel descriptor, then gating each channel.
    reduction=4 means hidden size = C//4 = 32 for C=128 filters.
    """
    def __init__(self, num_channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(num_channels // reduction, 1)
        self.se_fc1 = nn.Linear(num_channels, hidden, bias=True)
        self.se_fc2 = nn.Linear(hidden, num_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))                         # (B, C) — global average pool
        s = F.relu(self.se_fc1(s))
        s = torch.sigmoid(self.se_fc2(s))
        return x * s.unsqueeze(2).unsqueeze(3)         # scale each channel map


class ResidualBlock(nn.Module):
    """
    Residual block with Squeeze-and-Excitation recalibration.

    SE is applied after the second conv, before adding the skip connection,
    so it scales the residual (the "correction") rather than the full signal.
    """
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.se = SqueezeExcitation(num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x = F.relu(x + residual)
        return x


class ChessNet(nn.Module):
    """
    Chess neural network with SE residual body and WDL value head.

    Input:  (batch, INPUT_PLANES, 8, 8) board tensor
    Output: policy logits (batch, 4672), value logits (batch, 3)

    Value logits index: 0 = win, 1 = draw, 2 = loss (from current player's POV).
    Use wdl_to_scalar(v) to get a [-1, 1] scalar for MCTS.
    """
    def __init__(self):
        super().__init__()

        self.input_conv = nn.Conv2d(INPUT_PLANES, NUM_FILTERS, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(NUM_FILTERS)

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(NUM_FILTERS) for _ in range(NUM_RESIDUAL_BLOCKS)
        ])

        # Policy head: 8 filters preserve more body representation before FC
        self.policy_conv = nn.Conv2d(NUM_FILTERS, 8, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(8)
        self.policy_fc = nn.Linear(8 * 8 * 8, POLICY_OUTPUT_SIZE)

        # Value head: 4 filters, outputs WDL logits (3 classes)
        self.value_conv = nn.Conv2d(NUM_FILTERS, 4, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 3)  # W / D / L logits

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.residual_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.dropout(p)
        p = self.policy_fc(p)

        # Value head — returns raw logits; apply softmax at inference or use cross_entropy at training
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.dropout(self.value_fc1(v)))
        v = self.value_fc2(v)

        return p, v

    def get_all_activations(self, x: torch.Tensor) -> dict:
        """
        Run a forward pass and capture the output of every major layer.
        Returns a dict of name → numpy array for the network visualizer.
        """
        captured = {}
        hooks = []

        def make_hook(name):
            def fn(module, inp, out):
                captured[name] = out.detach().cpu().numpy()
            return fn

        hooks.append(self.input_conv.register_forward_hook(make_hook("input_conv")))
        for i, block in enumerate(self.residual_blocks):
            hooks.append(block.register_forward_hook(make_hook(f"res_block_{i}")))
        hooks.append(self.policy_conv.register_forward_hook(make_hook("policy_conv")))
        hooks.append(self.value_conv.register_forward_hook(make_hook("value_conv")))

        with torch.no_grad():
            policy_logits, value = self.forward(x)

        for h in hooks:
            h.remove()

        captured["policy_logits"] = policy_logits.detach().cpu().numpy()
        # value is WDL logits (B, 3); apply softmax so callers get probabilities
        captured["value"] = torch.softmax(value, dim=1).detach().cpu().numpy()
        return captured

    def get_activations(self, x: torch.Tensor, layer_index: int = 0) -> torch.Tensor:
        """
        Run a forward pass and capture the output of a specific residual block.
        Used by the visualization app to show what the network "pays attention to."

        Returns: (8, 8) heatmap averaged across all filters
        """
        activations = {}

        def hook_fn(module, input, output):
            activations['out'] = output.detach()

        handle = self.residual_blocks[layer_index].register_forward_hook(hook_fn)
        with torch.no_grad():
            self.forward(x)
        handle.remove()

        # Average across the filter dimension: (batch, 128, 8, 8) → (8, 8)
        return activations['out'].squeeze(0).mean(dim=0)
