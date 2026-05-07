# Neural Network Concepts — Intuition Guide

Everything here explained through the lens of your chess model. No abstract theory — just what's actually happening in your code and why.

---

## The Big Picture

Your model takes a chess position (a grid of pieces) and outputs two things:
1. **Policy:** "Here's how likely each legal move is to be good" (4672 probabilities)
2. **Value:** "White is winning by this much" (single number from -1 to +1)

It does this through a pipeline: encode the board as numbers → pass through layers that detect patterns → squeeze those patterns into move predictions and position evaluation.

---

## Tensors

A tensor is just a multi-dimensional array of numbers. Your board encoding is a tensor of shape (18, 8, 8):
- 18 "layers" stacked on top of each other
- Each layer is an 8×8 grid (the chess board)
- Each cell is 0 or 1

Think of it like 18 transparent overhead projector sheets. Sheet 0 shows where white pawns are (1s on their squares, 0s elsewhere). Sheet 5 shows where the white king is. Sheet 17 shows whose turn it is (all 1s or all 0s).

The network only understands numbers. This encoding translates the game into a format where spatial relationships are preserved — pieces next to each other on the board are next to each other in the tensor. That's crucial for convolutions.

---

## Convolutional Layers (Conv2d)

### What they do

A convolutional filter is a small pattern detector. Your filters are 3×3 — they look at a 3×3 patch of the board and output a single number saying "how much does this patch match my pattern?"

The filter slides across all 64 squares, producing an 8×8 output: one number per square saying "how strongly my pattern appears here."

### Intuition

Imagine a filter that learned to detect "white pawn with empty square in front." It would have positive weights where it expects a pawn and zero/negative where it expects the next square empty. Sliding it across the board, it lights up on every pawn that can push forward.

### Why 128 filters?

One filter detects one pattern. Chess has thousands of relevant patterns (open files, pawn chains, piece coordination, king safety). 128 filters per layer means 128 different pattern detectors running in parallel.

After layer 1: 128 "basic" patterns detected (piece presence, local threats).
After layer 2: 128 "combined" patterns (filter in layer 2 reads the outputs of all 128 filters from layer 1, so it can detect patterns-of-patterns).

### Why 3×3?

A 3×3 filter sees a king and its immediate neighbors — enough to detect local relationships. By stacking multiple layers, the "effective receptive field" grows. After 5 layers of 3×3 filters, each output cell has been influenced by an 11×11 region — larger than the whole 8×8 board. So deep stacking of small filters gives global awareness.

Using a single large filter (say 8×8) would theoretically work, but:
- Way more parameters (64× vs 9 per filter)
- Can't reuse local patterns in different locations
- Harder to optimize

### The 1×1 Convolution (in the heads)

A 1×1 conv doesn't look at spatial neighbors — it just mixes channels at each location. Think of it as: "at each square, combine information from all 128 feature maps into a summary." Used to shrink 128 channels down to 2 (policy) or 1 (value) before the final prediction.

---

## Residual Blocks (Skip Connections)

### The problem they solve

In a plain deep network, information must pass through every layer sequentially. If layer 3 corrupts the signal slightly, layers 4-5 work with corrupted input. Gradients (the learning signal) must also travel backwards through every layer — they get weaker at each step ("vanishing gradients"). Result: early layers barely learn.

### How skip connections fix it

```
Input ─────────────────────────┐
  │                            │
  ├─→ Conv → BN → ReLU        │ (the "residual" path)
  │                            │
  ├─→ Conv → BN               │
  │                            │
  └─→ ADD ←───────────────────┘ (skip connection: add input directly)
       │
       ReLU
       │
     Output
```

The skip connection adds the input directly to the output. This means:
- The block only needs to learn "what to change" (the residual), not "what the answer is"
- If the block learns nothing useful, it can output all zeros and the signal passes through unchanged — it can never make things worse
- Gradients flow directly through the skip connection, bypassing the conv layers — early layers still get strong learning signal

### Intuition

Without residual: each layer must describe the full picture.
With residual: each layer only describes "what's different from what I received." Like a chain of editors, each one making corrections to a document rather than rewriting it.

### Why does depth matter?

Each residual block combines information from a slightly wider spatial area with the existing representation. Block 1 might learn "this square is attacked." Block 2 might combine that with "and the attacker is pinned" (two separate facts merged into a complex concept). Block 5 might encode "there's a windmill tactic available starting from this square."

More blocks = deeper tactical combinations the network can represent.

---

## Batch Normalization (BatchNorm / BN)

### What it does

After each convolution, the outputs might be wildly different scales — some filters output values near 1000, others near 0.001. BatchNorm normalizes each filter's output to have mean=0 and std=1 across the current batch of examples.

### Why it helps

- **Stable training:** Without normalization, small changes in early layers cause cascading large changes in later layers. BN dampens this (the "internal covariate shift" problem).
- **Implicit regularization:** Because normalization depends on other examples in the batch, there's noise that acts like mild regularization (similar effect to dropout but less aggressive).
- **Allows higher learning rates:** Normalized activations mean gradients are more predictable, so you can take bigger steps without exploding.

### Intuition

Imagine each layer is a group of people shouting. Without BN, some are whispering and some are screaming — the next layer can't hear the quiet ones. BN gives everyone a microphone set to the same volume. The network can then learn to pay attention to content rather than volume.

### bias=False in conv layers

When you use BatchNorm right after a convolution, the conv layer doesn't need a bias term. Why? BN subtracts the mean anyway — any bias you add gets immediately subtracted. So `bias=False` removes a redundant parameter.

---

## Dropout

### What it does

During training, randomly sets a fraction of neuron outputs to zero. Your model uses 30% — so on each forward pass, 30% of neurons in the heads are dead.

### Why it helps

Prevents "co-adaptation" — when two neurons learn to work together so tightly that neither works alone. With dropout, every neuron must be independently useful because its partners might be dead on any given pass.

At test time (inference), dropout is turned off and all neurons are active. Their outputs are scaled down by (1 - dropout_rate) to compensate for the fact that more neurons are now contributing.

### Intuition

Like training a basketball team where random players sit out each practice. Everyone must learn to play every position. The team becomes more robust — no single player is a bottleneck.

### When it hurts

If your model is too small for the data (underfitting), dropout makes things worse — you're handicapping an already struggling network. The sign of this: training loss is high and val loss is barely different from training loss. In that case, reduce dropout.

### Why only in the heads, not the residual blocks?

The residual blocks already have BatchNorm which provides mild regularization. Adding dropout on top of BN in residual blocks tends to hurt — the two mechanisms interfere. Dropout in the heads is standard because the fully-connected layers there have the most parameters relative to their input size (overfitting risk is highest).

---

## Activation Functions (ReLU)

### What it does

`ReLU(x) = max(0, x)` — if the value is negative, output zero. If positive, pass through unchanged.

### Why networks need activations

Without activation functions, stacking layers is pointless. A linear function of a linear function is still just a linear function: `f(x) = Ax + b`, `g(f(x)) = A'(Ax+b) + b' = A''x + b''`. No matter how many layers you stack, the result is equivalent to one layer.

ReLU introduces non-linearity. Now the network can represent curves, thresholds, and combinations that no single linear function could.

### Why ReLU specifically?

- Simple: cheap to compute, trivial gradient (1 if positive, 0 if negative)
- Sparse: negative inputs get zeroed out, making representations sparse (many zeros). Sparse representations are easier to interpret and often generalize better.
- No vanishing gradient (for positive values): gradient is exactly 1, so signal flows perfectly through positive neurons.

### The "dying ReLU" problem

If a neuron's input is always negative (maybe due to a bad weight initialization), its output is always 0, its gradient is always 0, it never updates. It's dead forever. BatchNorm mostly prevents this by keeping activations centered near zero (so roughly half are positive). This is another reason BN and ReLU work well together.

---

## The Policy Head — Move Prediction

### How 4672 moves works

Every possible chess move is encoded as: (source square) × (move type) = 64 × 73 = 4672.

The 73 move types:
- 56 "queen-style" moves: 8 directions × 7 distances
- 8 knight moves: 8 L-shapes
- 9 underpromotions: 3 pieces × 3 directions

Queen promotions reuse the regular queen-move slot (distance=1 in the promotion direction).

### Why not just output one number?

A classification over all moves lets the network express uncertainty. "I think Nf3 is 40% likely to be best, but e4 is 35% and d4 is 25%" is much more useful than just picking one. MCTS uses these probabilities to decide which moves to explore first.

### Softmax

Raw network outputs ("logits") are arbitrary numbers. Softmax converts them to probabilities:
```
prob(move_i) = exp(logit_i) / sum(exp(all_logits))
```
All outputs become positive and sum to 1. Higher logit = higher probability, exponentially.

### Legal move masking

Before softmax, illegal moves are set to -infinity. `exp(-inf) = 0`, so they get exactly zero probability. The network never wastes capacity learning "don't play illegal moves" — you hardcode that constraint.

---

## The Value Head — Position Evaluation

### Why tanh?

The final output uses `tanh` which squishes to [-1, +1]:
- +1 = current player winning completely
- 0 = dead even / drawn
- -1 = current player losing completely

This bounded output matches the game outcomes we train against (+1 win, 0 draw, -1 loss).

### Why separate from policy?

Policy and value need different things. Policy needs to distinguish between 30+ legal moves — it needs fine-grained, move-specific information. Value just needs one holistic judgment of the position.

Sharing all layers but splitting at the end (the "two-headed" architecture) means:
- The shared body learns representations useful for BOTH tasks
- The body gets twice as much gradient signal (from both loss functions)
- But neither head can specialize the body solely for its own purpose — this tension is actually beneficial (regularization)

---

## Loss Functions

### Policy loss (Cross-entropy / KL divergence)

**Supervised:** Target is "the human played move X." Loss = -log(probability the network assigned to move X). If network gave 90% to the correct move, loss is low (-log(0.9) = 0.1). If it gave 1%, loss is high (-log(0.01) = 4.6).

**RL:** Target is the full MCTS visit distribution (a probability vector). Loss = KL divergence between network's distribution and MCTS distribution. Measures "how different are these two probability distributions?"

Why is the MCTS distribution a better target than single moves? Because MCTS aggregates 200 simulations of looking ahead. It represents "what would a much stronger player choose?" Training the network to match MCTS = distilling the tree search's knowledge into a single forward pass.

### Value loss (Mean Squared Error)

Target is the actual game outcome (+1, 0, -1). Loss = (prediction - target)^2. If you predicted +0.3 but the game was a loss (-1), loss = (0.3 - (-1))^2 = 1.69. High loss = learn to predict differently.

### Why combined loss?

`total_loss = policy_loss + 0.5 * value_loss`

Both tasks share the same body. The 0.5 weight on value means policy gets more influence on what patterns the body learns. Makes sense — the policy task is harder (predict among 4672 options vs predict one number) and directly determines move quality.

---

## Label Smoothing

### What it does

Instead of training target = [0, 0, 1, 0, 0] (100% on the correct move), use [0.02, 0.02, 0.92, 0.02, 0.02] (spread 10% across all legal moves).

### Why

- **Prevents overconfidence:** Without smoothing, the network is incentivized to push correct-move logits toward infinity. This makes the network brittle — certain about everything, even positions where multiple moves are good.
- **Better calibration:** In chess, there are often 2-3 moves that are roughly equal. Hard targets pretend there's always exactly one right answer.
- **Regularization effect:** Forces the network to keep logits moderate, preventing it from memorizing specific positions.

---

## Dirichlet Noise

### What it is

Dirichlet distribution generates random probability vectors (numbers that are positive and sum to 1). The parameter alpha controls how "concentrated" or "spread out" the result is:
- alpha = 0.03: output is very sparse (one option gets almost all probability)
- alpha = 0.3: output is moderately spread (2-4 options get most probability)
- alpha = 100: output is nearly uniform (all options equal)

### Where it's used

**In training targets (supervised):** Mix 25% Dirichlet noise into the policy target. The network sees that sometimes "wrong" moves get probability. This mimics the reality that humans sometimes play suboptimal but reasonable moves.

**In MCTS root (self-play):** Mix 35% Dirichlet noise into the root node's prior probabilities. Without this, every self-play game from the same opening would follow the exact same line (the network always suggests the same moves). Noise creates variety → diverse training data → better generalization.

### Alpha = 0.3 for chess

Chess has ~30 legal moves on average. Alpha = 0.3 means Dirichlet typically puts most mass on 2-4 moves. This matches chess reality — usually there are a few good moves and many bad ones. For Go (250 legal moves), AlphaZero uses alpha = 0.03 (even sparser).

---

## MCTS (Monte Carlo Tree Search)

### The core insight

The neural network gives you a "gut feeling" about each position (policy = which moves look good, value = who's winning). But gut feelings are often wrong. MCTS uses those gut feelings as a starting point, then verifies them by actually looking ahead.

### The four steps, intuitively

1. **Select:** Start at current position. Walk down the tree, always picking the child that balances "looks promising" (high prior/value) with "hasn't been explored much" (low visit count). This is the UCB formula.

2. **Expand:** When you reach an unexplored position, ask the neural network "what do you think?" It gives you a policy (priors for children) and a value (position evaluation).

3. **Backup:** Take that value and send it back up to every node you visited. Each ancestor updates its average value. Now they have one more data point about whether going down this path leads to good outcomes.

4. **Repeat:** Do this 200 times. Nodes that lead to good outcomes get visited more. Nodes that lead to bad outcomes get abandoned.

After 200 simulations, pick the most-visited move from the root. Visit count is more reliable than average value (it's more stable — a node with 100 visits and Q=0.6 is more trustworthy than one with 3 visits and Q=0.8).

### UCB formula: Q + C * P * sqrt(parent_visits) / (1 + visits)

- **Q:** Average value seen from this node. High Q = this move leads to good positions.
- **C * P * sqrt(parent_visits) / (1 + visits):** Exploration bonus. High when:
  - P is high (network thinks this move is good)
  - visits is low (we haven't checked this move much)
  - parent_visits is high (the more we explore elsewhere, the more urgent it becomes to check this one)

This naturally balances exploitation (go where Q is high) with exploration (try things we haven't verified yet).

### Why is MCTS output better than raw network output?

The network sees a position and makes one guess in ~1ms. MCTS uses that guess as a starting point, then runs 200 simulated games from that position, each one looking 5-20 moves ahead. The visit distribution reflects the aggregate wisdom of those 200 simulations.

Training the network to imitate MCTS = the network learns to predict "what I would conclude after deep thought" from a single glance. Over many iterations, the network gets better → MCTS gets better (because it uses the network) → training targets get better → network gets better. This is the AlphaZero flywheel.

---

## The Learning Rate

### What it is

How big a step to take when updating weights based on the gradient. Gradient says "go this direction." Learning rate says "go this far."

### Too high

Weights overshoot the minimum. Loss oscillates or diverges. Like trying to land a helicopter by slamming the stick left and right.

### Too low

Training takes forever. Gets stuck in bad local minima because it can't jump over small hills in the loss landscape.

### Your schedule

- **Warmup (0 → 0.001 over 1000 steps):** At the start, the network is random. Gradients are large and noisy. Small LR lets the network find a reasonable region before taking big steps.
- **Cosine annealing (0.001 → ~0):** Smoothly decay LR over training. Big steps early to explore, tiny steps late to settle precisely into the minimum.
- **RL uses 1e-4 (10× smaller than supervised):** The network already knows chess from supervised training. You're fine-tuning, not learning from scratch. Big LR would destroy existing knowledge.

---

## Optimizer (Adam)

### What it does beyond basic gradient descent

Plain gradient descent: `weight -= lr * gradient`

Adam adds two things:
1. **Momentum:** Keeps a running average of past gradients. If gradients have been pointing the same direction for many steps, go faster (confident direction). If they oscillate, slow down.
2. **Per-parameter learning rate:** Tracks how large gradients have been for each parameter individually. Parameters with consistently small gradients get effectively larger steps. Parameters with huge gradients get smaller steps. Levels the playing field.

### Why it's standard

Converges faster than plain SGD on most problems. Less sensitive to hyperparameter choice. The default betas (0.9, 0.999) work well almost everywhere.

### Weight decay (1e-4)

Penalty for large weights: each update, multiply all weights by (1 - weight_decay). Gently pushes all weights toward zero. Prevents any single connection from becoming dominant. Like a complexity tax — the network must justify every weight.

---

## Gradient Clipping

### The problem

Sometimes a batch produces an unusually large gradient (rare position, noisy label, unlucky combination). Without clipping, one bad batch can catapult weights into a bad region, undoing thousands of steps of progress.

### The fix

If the total gradient magnitude exceeds 1.0, scale it down proportionally so its magnitude is exactly 1.0. Direction preserved, magnitude capped. Like a speed limiter — you can steer anywhere, but you can't floor it.

---

## Overfitting vs Underfitting

### Overfitting

Model memorizes training data instead of learning general patterns. Signs: training loss is very low, validation loss is much higher. Like a student who memorized practice test answers but can't solve new problems.

**Remedies:** More data, dropout, weight decay, smaller model, early stopping, data augmentation.

### Underfitting

Model is too simple or too constrained to capture the patterns. Signs: both training and validation loss are high, or they're similar but neither is good.

**Remedies:** Bigger model, less regularization (less dropout, less weight decay), train longer, lower learning rate (might be overshooting).

### The balance in your model

With ~1.6M positions and ~3M parameters, you're in a reasonable ratio. The 0.3 dropout might be pushing toward underfitting. Check your training logs — if val loss tracks closely with training loss (gap < 20%), dropout is too aggressive.

---

## Temperature (in MCTS)

### What it controls

Temperature scales the visit counts before converting to probabilities:

```
prob(move) = visits^(1/temperature) / sum(all_visits^(1/temperature))
```

- **temperature = 1:** Probabilities proportional to visits. Move visited 100 times is 10× more likely than one visited 10 times.
- **temperature → 0:** Nearly deterministic. Most-visited move gets ~100%. Like turning up the contrast.
- **temperature > 1:** More uniform. Smooths out differences. Rarely used.

### Why the schedule (1.0 early, 0.1 late)?

Early game: play more randomly to generate diverse training positions. Visit openings that aren't the single "best" one. This variety is crucial — without it, the network only ever sees the same 3 openings.

Late game: play the best move. Endgames have clear right answers. Random play leads to drawn endgames being lost → bad training signals.

---

## Self-Play: The AlphaZero Loop

### Why self-play works at all

Seems circular: model trains on its own games. If it plays badly, it learns from bad games and stays bad, right?

No, because of MCTS. Even a weak model + 200 MCTS simulations produces better moves than the raw model alone. So the training target (MCTS visit distribution) is always slightly better than what the network can currently do in one shot. The network chases a moving target that's always one step ahead.

### The flywheel

1. Network is weak. MCTS compensates by searching deeply.
2. Train network to imitate MCTS output.
3. Network gets slightly better.
4. MCTS using better network produces even better moves.
5. Train network on those better moves.
6. Repeat.

The key insight: MCTS amplifies any improvement in the network into even better training data. It's a positive feedback loop that bootstraps from random initialization to superhuman play — given enough compute.

### Why not just train on human games forever?

Human games cap your model at human level (specifically, the level of players in your dataset). Self-play has no ceiling. Also, human games have biases — certain openings overrepresented, styles that are popular but suboptimal.

Self-play generates positions that are specifically informative for the current model — positions where it's uncertain, positions just beyond its current horizon. The curriculum adapts to the learner.

---

## The Two Heads: Shared Representation

### Why one network with two outputs?

Alternative: two separate networks, one for policy, one for value.

Shared body is better because:
- **Information synergy:** Understanding "who's winning" helps predict good moves. Understanding "what are the good moves" helps evaluate positions. The body learns features useful for both.
- **Efficiency:** One forward pass gives both outputs. In MCTS, you need both policy (to set priors) and value (to evaluate) at every expansion.
- **Regularization:** The two loss functions act as mutual regularizers. The policy task prevents the body from collapsing to "just predict who wins" and vice versa.

### The tension

Sometimes what the value head needs conflicts with what the policy head needs. A subtle positional feature might be crucial for evaluation but irrelevant for move selection. With a shared body, the network must compromise.

This is why the heads have their own layers after the body — the policy head can further process the shared representation specifically for move prediction, and the value head can do the same for evaluation. The shared body captures common ground; the separate heads specialize.
