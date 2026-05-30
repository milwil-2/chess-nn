// Roadmap — items pulled from template/docs/improvements.md.
interface Card {
  title: string;
  body: string;
  src?: string;
}

const CARDS: Card[] = [
  {
    title: "Opponent-policy head",
    body: "A small auxiliary head that predicts the opponent's reply, forcing the body to model threats — the signal that is missing when the network hangs a piece.",
    src: "KataGo · +40–90 Elo in Go",
  },
  {
    title: "Forced playouts + target pruning",
    body: "Guarantee a minimum visit count per child, then prune moves whose visits came only from the forcing rule. Cleaner training targets with less Dirichlet noise.",
    src: "KataGo §3 · arXiv:1902.10565",
  },
  {
    title: "Playout-cap randomization",
    body: "Give only a fraction of self-play moves a full search and the rest a cheap one. Roughly 3x faster self-play with negligible target-quality loss.",
    src: "KataGo §4",
  },
  {
    title: "Syzygy supervision in self-play",
    body: "Replace value labels with tablebase ground truth for ≤6-man positions so over-generalized endgame patterns stop leaking back into the opening.",
    src: "Seer engine approach",
  },
  {
    title: "Transformer body",
    body: "Swap the conv tower for a Leela BT4-style encoder over the 64 squares (with smolgen). A large rewrite, only worth doing once the MCTS-side improvements are exhausted.",
    src: "Leela Chess Zero · 2024",
  },
];

export default function Roadmap() {
  return (
    <section id="roadmap" className="section">
      <div className="wrap">
        <div className="section-head">
          <span className="kicker">Roadmap</span>
          <h2 className="section-title">Roadmap</h2>
          <p className="section-sub">
            Planned changes, ordered roughly by expected impact on strength. Weighted toward
            removing the weak-move symptom — opponent-policy head, forced playouts, playout-cap
            randomization, tablebase value supervision, and eventually a transformer body.
          </p>
        </div>

        <div className="road-grid">
          {CARDS.map((c, i) => (
            <div className="road-card" key={c.title}>
              <span className="num">{String(i + 1).padStart(2, "0")}</span>
              <h4>{c.title}</h4>
              <p>{c.body}</p>
              {c.src && <div className="src">{c.src}</div>}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
