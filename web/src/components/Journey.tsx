// Version history — v1 to v3 (plus a pre-history prototype). Each variant
// targets the recurring symptom: the network played visibly weak moves
// (early king walks, hung pieces).
interface Stage {
  ver: string;
  where: string;
  star?: boolean;
  title: string;
  body: string;
  tags: string[];
}

const STAGES: Stage[] = [
  {
    ver: "v1_history8",
    where: "local · M3 Mac",
    title: "First 8-frame-history model",
    body: "A 10-block SE-residual tower with a WDL value head, fed 8 frames of board history so it could see piece trajectories. Trained on filtered Lichess games (rating ≥ 2000) on a laptop GPU.",
    tags: ["8-frame history", "SE blocks", "WDL head", "MIN_RATING 2000"],
  },
  {
    ver: "v1_history8_vast",
    where: "cloud GPU",
    title: "Ported to cloud GPU",
    body: "Same architecture, moved onto a rented Vast.ai GPU so training was no longer bottlenecked by the Mac. Larger batches and a linearly scaled learning rate.",
    tags: ["Vast.ai GPU", "larger batch", "LR scaling"],
  },
  {
    ver: "v2_vast",
    where: "cloud GPU",
    title: "Expanded metadata, larger batches",
    body: "Metadata grew from 6 to 9 planes (added side-to-move and two repetition flags, for 105 input planes total). Large-batch training and a rebalanced value loss. Stronger than v1, but still played occasional clearly bad moves.",
    tags: ["105 planes", "batch 2048", "value-loss rebalance"],
  },
  {
    ver: "v3_vast",
    where: "cloud GPU",
    star: true,
    title: "Current best — Stockfish supervision + tactical oversampling",
    body: "Adds a Stockfish auxiliary policy loss (depth-12 best move on 20% of positions), 3x oversampling of tactical positions (hanging piece or fork), MIN_RATING raised to 1800, and a set of inference-time search helpers used only at play time.",
    tags: ["Stockfish supervision", "tactical 3x oversample", "search helpers", "MIN_RATING 1800"],
  },
];

export default function Journey() {
  return (
    <section id="journey" className="section">
      <div className="wrap">
        <div className="section-head">
          <span className="kicker">Versions</span>
          <h2 className="section-title">Versions</h2>
          <p className="section-sub">
            Four trained variants (plus a pre-history prototype). Each <code>models/&lt;variant&gt;/</code>
            directory is a self-contained snapshot of the template codebase at creation time, so
            older variants stay runnable as fixed baselines. The progression was driven by a
            recurring symptom — the model played visibly weak moves (early king walks, hung pieces).
          </p>
        </div>

        <div className="timeline">
          {STAGES.map((s) => (
            <div className="tl-item" key={s.ver}>
              <div className="tl-meta">
                <div className={`ver${s.star ? " star" : ""}`}>
                  {s.star ? "★ " : ""}
                  {s.ver}
                </div>
                <div className="where">{s.where}</div>
              </div>
              <div className="tl-body">
                <h4>{s.title}</h4>
                <p>{s.body}</p>
                <div className="tl-tags">
                  {s.tags.map((t) => (
                    <span className="tl-tag" key={t}>
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
