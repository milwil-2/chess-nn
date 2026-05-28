// Journey section — the v1 → v3 evolution, framed around the recurring symptom
// (visibly weak moves: early king walks, hung pieces) that drove each change.
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
    title: "First model with 8-frame history",
    body: "The first real network: a 10-block SE-residual tower with a WDL value head, fed eight frames of board history so it could see piece trajectories. Trained on filtered Lichess games (rating ≥ 2000) right on a laptop GPU.",
    tags: ["8-frame history", "SE blocks", "WDL head", "MIN_RATING 2000"],
  },
  {
    ver: "v1_history8_vast",
    where: "cloud GPU",
    title: "Ported to the cloud",
    body: "Same architecture, moved onto a rented Vast.ai GPU so training wasn't bottlenecked by the Mac. Bigger batches and a scaled learning rate — faster iteration, same brain.",
    tags: ["Vast.ai GPU", "larger batch", "LR scaling"],
  },
  {
    ver: "v2_vast",
    where: "cloud GPU",
    title: "More context, bigger batches",
    body: "Metadata grew from 6 to 9 planes (added side-to-move and two repetition flags → 105 input planes). Large-batch training and a rebalanced value loss. Stronger, but still played the odd embarrassing move.",
    tags: ["105 planes", "batch 2048", "value-loss rebalance"],
  },
  {
    ver: "v3_vast",
    where: "cloud GPU",
    star: true,
    title: "Current best — supervision + tactics",
    body: "The push to kill the weak moves: a Stockfish auxiliary loss grounds the policy on engine-quality targets, tactical positions are oversampled 3×, the rating floor is 1800, and a suite of inference-time search helpers backs the engine at play time.",
    tags: ["Stockfish supervision", "tactical 3× oversample", "search helpers", "MIN_RATING 1800"],
  },
];

export default function Journey() {
  return (
    <section id="journey" className="section">
      <div className="wrap">
        <div className="section-head">
          <span className="kicker">The journey</span>
          <h2 className="section-title">Four versions, one stubborn symptom</h2>
          <p className="section-sub">
            Every version targeted the same problem: the network kept making moves a club player
            would never make — walking the king out early, leaving pieces hanging. Each iteration
            chipped away at it from a different angle.
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
