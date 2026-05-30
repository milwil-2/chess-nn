// Improvements — grouped cards covering data/supervision, architecture,
// search/MCTS, and inference-time helpers.
import type { ReactNode } from "react";

interface Item {
  name: ReactNode;
  why: string;
}
interface Group {
  idx: string;
  title: string;
  items: Item[];
}

const GROUPS: Group[] = [
  {
    idx: "01",
    title: "Data & supervision",
    items: [
      {
        name: (
          <>
            Rating floor → <code>MIN_RATING 1800</code>
          </>
        ),
        why: "Raised from 1500 (v2) so the network learns from fewer blunders. Was 2000 in v1 local, dropped to 1500 in v2, settled at 1800 in v3.",
      },
      {
        name: <>Stockfish auxiliary loss</>,
        why: "20% of positions are annotated with Stockfish's best move (depth 12) and added as an auxiliary policy target with SF_LOSS_WEIGHT = 0.3.",
      },
      {
        name: <>Tactical 3x oversampling</>,
        why: "Positions containing a hanging piece or fork (detected via tactics.py) are sampled 3x during training.",
      },
    ],
  },
  {
    idx: "02",
    title: "Architecture",
    items: [
      {
        name: <>Squeeze-and-Excitation blocks</>,
        why: "Cheap channel-wise attention (reduction 4) applied inside each residual block before the skip-add.",
      },
      {
        name: <>WDL value head</>,
        why: "3-class win/draw/loss softmax instead of a single scalar, better calibrated for draw-heavy chess. Collapsed to P(win) - P(loss) for search.",
      },
      {
        name: <>8-frame board history</>,
        why: "8 stacked frames let the network see piece trajectories and repetition that a single-frame encoding can't show.",
      },
    ],
  },
  {
    idx: "03",
    title: "Search / MCTS",
    items: [
      {
        name: <>Subtree reuse</>,
        why: "The MCTS subtree under the chosen move is carried into the next search instead of rebuilt from scratch.",
      },
      {
        name: <>Shaped Dirichlet noise (KataGo)</>,
        why: "Root exploration noise is restricted to plausible moves via DIRICHLET_SHAPE_FLOOR, so self-play doesn't add low-prior moves (e.g. early king pushes) to the training targets.",
      },
      {
        name: (
          <>
            Root policy temperature <code>T = 1.5</code>
          </>
        ),
        why: "Policy logits are flattened by T before the root softmax (swept value: 1.50), so search considers tactical alternatives that the raw policy was over-confident against.",
      },
    ],
  },
  {
    idx: "04",
    title: "Inference helpers (never in self-play)",
    items: [
      {
        name: <>Opening book + Syzygy tablebases</>,
        why: "Weighted Polyglot (codekiddy) opening lookups and 3-4-5-man Syzygy probing with distance-to-zero tie-breaking. Engine-side only.",
      },
      {
        name: <>Transposition cache</>,
        why: "Persisted MCTS visit counts, segmented per-checkpoint and auto-saved, mixed into priors to warm-start positions seen in earlier games.",
      },
      {
        name: <>Blunder filter</>,
        why: "At the root, prune moves that hang material unless the move itself is a capture or check. Never used during self-play so the learned policy stays uncontaminated.",
      },
    ],
  },
];

export default function Improvements() {
  return (
    <section id="improvements" className="section">
      <div className="wrap">
        <div className="section-head">
          <span className="kicker">Improvements</span>
          <h2 className="section-title">Improvements</h2>
          <p className="section-sub">
            The hand-coded inference helpers (opening book, Syzygy tablebases, transposition cache,
            blunder filter) are used only at play time and never during self-play, so the learned
            policy is not contaminated by hand-coded heuristics.
          </p>
        </div>

        <div className="imp-groups">
          {GROUPS.map((g) => (
            <div className="imp-group" key={g.idx}>
              <h4>
                <span className="idx">{g.idx}</span>
                {g.title}
              </h4>
              {g.items.map((it, i) => (
                <div className="imp-item" key={i}>
                  <div className="name">{it.name}</div>
                  <div className="why">{it.why}</div>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
