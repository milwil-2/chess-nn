// Page shell: nav, hero, Play & Compare, Architecture,
// Improvements, Strength, footer.
import EngineBoard from "./components/EngineBoard";
import Architecture from "./components/Architecture";
import Improvements from "./components/Improvements";
import Strength from "./components/Strength";

export default function App() {
  return (
    <div className="shell">
      <nav className="nav">
        <div className="nav-inner">
          <div className="nav-brand">
            <span className="nav-dot" />
            chess-nn
          </div>
          <div className="nav-links">
            <a href="#play">Play</a>
            <a href="#architecture">Architecture</a>
            <a href="#improvements">Improvements</a>
            <a href="#strength">Strength</a>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="hero">
        <div className="wrap">
          <h1>
            A chess engine, trained on an{" "}
            <span className="accent">8GB MacBook Air</span>.
          </h1>
          <p className="lead">
            A convolutional policy/value network driven by Monte-Carlo Tree Search, written in
            PyTorch. Play Stockfish at a selectable Elo in the browser; on every move the
            network's top picks (raw policy, no search) are shown next to Stockfish's analysis.
          </p>

          <div className="hero-stats">
            <div className="stat">
              <div className="v">5.6M</div>
              <div className="l">parameters</div>
            </div>
            <div className="stat">
              <div className="v">105</div>
              <div className="l">input planes</div>
            </div>
            <div className="stat">
              <div className="v">10×128</div>
              <div className="l">SE-residual blocks</div>
            </div>
            <div className="stat">
              <div className="v">4672</div>
              <div className="l">move policy</div>
            </div>
          </div>

          <div className="hero-cta">
            <a className="btn primary" href="#play">
              Play and compare
            </a>
            <a className="btn ghost" href="#architecture">
              How it works
            </a>
          </div>
        </div>
      </section>

      {/* Play & Compare */}
      <section id="play" className="section">
        <div className="wrap">
          <div className="section-head">
            <span className="kicker">
              <span className="sec-idx">01</span>Demonstration
            </span>
            <h2 className="section-title">Play and compare</h2>
            <p className="section-sub">
              Drag a piece to move. Stockfish replies at the chosen Elo. After every move, the
              network's top pick (green) and Stockfish's best move (blue) are drawn on the board,
              with the full breakdown on the right.
            </p>
          </div>
          <EngineBoard />
        </div>
      </section>

      <Architecture />
      <Improvements />
      <Strength />

      <footer className="footer">
        <div className="wrap footer-inner">
          <div className="muted">
            chess-nn, a chess engine in PyTorch.
          </div>
          <div className="mono">
            net = raw policy · Stockfish 16 NNUE · runs client-side
          </div>
        </div>
      </footer>
    </div>
  );
}
