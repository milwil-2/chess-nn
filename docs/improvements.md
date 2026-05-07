# Improvement Roadmap

Ordered by expected impact on playing strength. Each section explains what to change, why it helps, and rough implementation notes.

---

## High Impact

### 1. Add Position History to Input

**Current:** Model sees only the current board (18 planes).
**Change:** Stack last 8 positions = 8 × 12 piece planes + 4 castling + 1 en passant + 1 turn = 101 planes (or simplified: 8 × 14 + 4 meta = 116).

**Why:** Without history, the model can't detect:
- Threefold repetition (draws it can't see coming)
- Recently moved pieces (a rook that just moved is important context)
- Tempo and initiative (who's been making threats)
- Whether a position is getting better or worse over time

AlphaZero uses 8 history frames. Leela uses 8. This is the single biggest gap vs the reference implementations.

**Implementation:**
- Modify `board_encoding.py` to accept a list of board states
- Change `INPUT_PLANES` in config from 18 to ~119
- Modify `dataset.py` to walk through games accumulating history
- Keep a sliding window during self-play
- First conv layer input channels changes, everything else stays the same

**Cost:** ~2× more memory per position in dataset. Training speed barely changes (first conv is not the bottleneck).

---

### 2. Increase Residual Blocks (5 → 10 or 15)

**Current:** 5 blocks.
**Change:** 10-15 blocks.

**Why:** Each block is roughly one "step of reasoning." With 5 blocks the network can combine information across ~5 hops. Chess tactics often require seeing 4-6 move sequences. More blocks = deeper tactical vision.

Leela uses 15-20 blocks for its standard nets. AlphaZero uses 19 or 39.

**Implementation:**
- Change `NUM_RESIDUAL_BLOCKS = 10` in config.py
- That's it — model builds itself from config

**Cost:** ~2× training time, ~2× inference time (directly impacts MCTS speed). Model size goes from ~3M to ~6M params. Still fits in 8GB easily.

**Suggestion:** Go to 10 first. If training is tolerable, try 15 later.

---

### 3. More Self-Play Games Per Iteration (25 → 100+)

**Current:** 25 games per RL iteration.
**Change:** 50-100 games.

**Why:** 25 games × ~150 moves = ~3,750 training positions per iteration. That's tiny — the model overfits to these quickly. More games = more diverse positions, more stable training signal, fewer "lucky" results distorting the policy.

**Implementation:**
- Change `RL_GAMES_PER_ITER = 100` in config.py
- Or pass `--games 100` to `python run.py rl`

**Cost:** Linear time increase. 25 games at ~2min each = ~50min per iteration. 100 games = ~200min. Consider running overnight, or reducing simulations to compensate.

**Alternative:** Keep 25 games but run many more iterations. Smaller steps, more frequent evaluation. Both work.

---

## Medium Impact

### 4. Widen Policy and Value Head Bottlenecks

**Current:** Policy uses 2 filters (128 values → 4672 moves). Value uses 1 filter (64 values → win prediction).
**Change:** Policy: 2 → 8 filters. Value: 1 → 4 filters.

**Why:** The body (residual blocks) builds a rich 128-filter representation. Then the heads squeeze it through a tiny bottleneck before making their prediction. This throws away information. A pawn structure nuance that matters for evaluation gets lost if it can't fit through 64 values.

Think of it like summarizing a book in one sentence vs one paragraph. More room = more nuance preserved.

**Implementation in `model.py`:**
```python
# Policy head
self.policy_conv = nn.Conv2d(NUM_FILTERS, 8, kernel_size=1, bias=False)  # was 2
self.policy_bn = nn.BatchNorm2d(8)
self.policy_fc = nn.Linear(8 * 8 * 8, POLICY_OUTPUT_SIZE)  # was 2*8*8

# Value head
self.value_conv = nn.Conv2d(NUM_FILTERS, 4, kernel_size=1, bias=False)  # was 1
self.value_bn = nn.BatchNorm2d(4)
self.value_fc1 = nn.Linear(4 * 8 * 8, 256)  # was 1*8*8
```

**Cost:** Negligible. Adds maybe 100K parameters. Training speed unchanged.

---

### 5. Reduce Dropout (0.3 → 0.1)

**Current:** 30% dropout in both heads.
**Change:** 10-15%.

**Why:** Dropout prevents overfitting by randomly killing neurons during training. But with ~1.6M training positions and only 3M parameters, you're probably not overfitting much. High dropout can cause *underfitting* — the network never gets to use its full capacity.

**How to check:** Compare training loss vs validation loss. If they're close together (val only slightly higher), you're not overfitting and dropout is too aggressive. If val is much higher than train, keep dropout high.

**Implementation:**
```python
self.dropout = nn.Dropout(p=0.1)  # was 0.3
```

**Cost:** None. Might actually train faster (converges quicker without neurons being killed).

---

### 6. Increase RL History Window (5 → 10-15)

**Current:** Trains on the 5 most recent self-play files.
**Change:** 10-15 files.

**Why:** With only 5 files of ~3,750 positions each = ~18K positions. That's very small for training a 3M parameter model. More history = more data = less overfitting to recent games. Old data was from weaker model, but it's still valid chess — the positions existed, the values are correct.

**Implementation:**
```python
RL_HISTORY_FILES = 12  # was 5
```

**Cost:** Slightly more RAM, slightly longer training per iteration. Worth it.

---

### 7. Add Legal Move Masking in RL Training

**Current:** `train_on_selfplay` does log_softmax over all 4672 moves. The supervised training masks illegal moves to -inf. RL training doesn't.

**Why:** Without masking, the network spends capacity learning "don't play illegal moves" from the zero entries in the MCTS distribution. It works eventually, but it's wasteful — that's something you can just hardcode.

**Implementation in `train_rl.py`:**
```python
# Need to store legal_masks alongside boards/policies/values in self-play data
# Then in train_on_selfplay:
masked_logits = policy_logits.masked_fill(~legal_masks, float("-inf"))
log_probs = F.log_softmax(masked_logits, dim=1)
```

**Cost:** Need to modify self-play data format to include legal masks. Moderate code change but straightforward.

---

## Lower Impact (Polish)

### 8. Squeeze-and-Excitation (SE) Blocks

Add a "channel attention" mechanism to each residual block. After the two convolutions, squeeze the spatial dimension (global average pool), pass through two small linear layers, get per-channel scaling factors, multiply.

This lets the network say "for THIS position, pay more attention to the king-safety filters and less to the pawn-structure filters." Dynamic importance weighting.

Leela Chess Zero uses these and they give consistent +30 Elo.

**Implementation:** Add ~10 lines per residual block. Adds ~5% compute.

---

### 9. Mixed Precision Training (float16)

MPS supports half-precision. Forward pass in float16, gradients accumulated in float32.

**Implementation:**
```python
from torch.cuda.amp import autocast, GradScaler  # works on MPS too
with autocast():
    policy, value = model(boards)
    loss = ...
scaler.scale(loss).backward()
scaler.step(optimizer)
```

**Cost:** Free speed (~1.5× faster). Slight numerical noise, usually irrelevant.

---

### 10. Tune Resignation Threshold

**Current:** Resign after 5 consecutive moves with value confidence > 0.95.

**Risk:** Early in RL training, value head is unreliable. It might think it's losing when it's not, causing premature resignations → bad training signal (game labeled as loss when it wasn't actually lost).

**Fix:** Disable resignation for the first 3-5 RL iterations. Or raise threshold to 0.99. Or require 10 consecutive moves instead of 5.

---

### 11. Data Augmentation via Board Symmetry

Chess has one axis of symmetry: horizontal flip (mirror across the d/e file boundary). If there's no castling rights and no asymmetric pawn structure, a mirrored position is equally valid.

This effectively doubles your training data for positions where the symmetry applies. Complex to implement correctly (must also mirror the move encoding), but free data is valuable.

---

### 12. Progressive Widening in MCTS

Instead of expanding all legal moves at once, only expand the top-K moves by prior probability. Add more children as visit count grows. Saves compute in positions with 30+ legal moves where most are garbage.

---

### 13. Pondering / Reuse of Search Tree

Currently each move starts MCTS from scratch. Instead, after making a move, keep the subtree rooted at that move's child node. Free simulations from last turn carry over.

**Implementation:** After choosing a move, set `root = root.children[chosen_move]`. Start next search from there instead of empty root.

---

## Suggested Order of Implementation

1. Widen head bottlenecks (#4) — 5 minutes, immediate benefit
2. Reduce dropout (#5) — 1 line change, test if val loss improves
3. Increase RL history window (#6) — 1 line change
4. Add legal masking to RL (#7) — moderate effort, cleaner training
5. More residual blocks (#2) — 1 line change, retrain
6. Position history (#1) — biggest effort, biggest gain
7. SE blocks (#8) — moderate effort, good gains
8. Everything else as desired
