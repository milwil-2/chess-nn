// Improvements / "what I learned" — grouped cards, each with a one-line
// "why it matters".
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
        why: "Bad moves in the training set become moves the network thinks are valid. A higher floor means fewer blunders to imitate.",
      },
      {
        name: <>Stockfish auxiliary loss</>,
        why: "20% of positions get Stockfish's best move (depth 12) as an extra policy target — engine-quality ground truth pulling the policy straight.",
      },
      {
        name: <>Tactical 3× oversampling</>,
        why: "Positions with a hanging piece or fork are sampled three times as often, so the net actually practises the sharp moments instead of glossing over them.",
      },
    ],
  },
  {
    idx: "02",
    title: "Architecture",
    items: [
      {
        name: <>Squeeze-and-Excitation blocks</>,
        why: "Cheap channel-wise attention that consistently helps — the network learns which feature maps matter for a given position.",
      },
      {
        name: <>WDL value head</>,
        why: "Three-class win/draw/loss instead of a single scalar — far better calibrated for draw-heavy chess.",
      },
      {
        name: <>8-frame board history</>,
        why: "Eight stacked frames expose piece trajectories and repetition that a single snapshot can't show.",
      },
    ],
  },
  {
    idx: "03",
    title: "Search / MCTS",
    items: [
      {
        name: <>Subtree reuse</>,
        why: "The tree under the played move carries into the next search instead of being thrown away — effectively free simulations.",
      },
      {
        name: <>Shaped Dirichlet noise</>,
        why: "Root exploration noise is restricted to plausible moves, so self-play stops poisoning the training data with early-king-walk junk.",
      },
      {
        name: (
          <>
            Root policy temperature <code>T = 1.5</code>
          </>
        ),
        why: "Flattening over-confident priors at the root lets search actually consider the tactical alternatives it would otherwise ignore.",
      },
    ],
  },
  {
    idx: "04",
    title: "Inference helpers (never in self-play)",
    items: [
      {
        name: <>Opening book + Syzygy tablebases</>,
        why: "Principled, varied openings and perfect 3-4-5-man endgame play — bolted on at play time only.",
      },
      {
        name: <>Transposition cache</>,
        why: "Persisted MCTS visit counts warm-start positions the engine has seen before, across games.",
      },
      {
        name: <>Blunder filter</>,
        why: "At the root, prune moves that hang material unless they're a capture or check. Kept out of training so the policy isn't contaminated by hand-coded rules.",
      },
    ],
  },
];

export default function Improvements() {
  return (
    <section id="improvements" className="section">
      <div className="wrap">
        <div className="section-head">
          <span className="kicker">What I learned</span>
          <h2 className="section-title">The fixes that moved the needle</h2>
          <p className="section-sub">
            A deliberate rule runs through all of this: the hand-coded helpers (book, tablebases,
            cache, blunder filter) are used only at play time and never during self-play, so the
            learned policy stays honest.
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
