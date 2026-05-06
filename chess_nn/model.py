"""
Neural network architecture: a CNN with residual blocks and two output heads.

Architecture summary:
  Input (18, 8, 8)
    → Initial conv layer
    → 5 residual blocks  (the "body" — extracts chess patterns)
    → Policy head        (outputs probabilities over 4672 moves)
    → Value head         (outputs a single number: who's winning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_RESIDUAL_BLOCKS, NUM_FILTERS, INPUT_PLANES, POLICY_OUTPUT_SIZE


class ResidualBlock(nn.Module):
    """
    A residual block: two conv layers with a skip connection.

    The skip connection adds the input directly to the output, which lets
    the network learn "corrections" rather than full transformations.
    This is what makes deep networks trainable — without it, gradients
    vanish before reaching early layers.
    """
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)  # Skip connection: add input back in
        return x


class ChessNet(nn.Module):
    """
    The full chess neural network.

    Input:  (batch, 18, 8, 8) board tensor
    Output: policy logits (batch, 4672) and value (batch, 1)
    """
    def __init__(self):
        super().__init__()

        # Initial conv: expand from 18 input planes to NUM_FILTERS feature maps
        self.input_conv = nn.Conv2d(INPUT_PLANES, NUM_FILTERS, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(NUM_FILTERS)

        # Body: stack of residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(NUM_FILTERS) for _ in range(NUM_RESIDUAL_BLOCKS)
        ])

        # Policy head: shrink filters → flatten → predict move probabilities
        self.policy_conv = nn.Conv2d(NUM_FILTERS, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, POLICY_OUTPUT_SIZE)

        # Value head: shrink filters → flatten → predict game outcome
        self.value_conv = nn.Conv2d(NUM_FILTERS, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Dropout applied in heads only — residual blocks use BN which already regularizes
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # Body
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.residual_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.dropout(p)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.dropout(self.value_fc1(v)))
        v = torch.tanh(self.value_fc2(v))

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
        captured["value"] = value.detach().cpu().numpy()
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
