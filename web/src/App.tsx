// Page shell: nav → hero → Play & Compare → Architecture → Journey →
// Improvements → Strength → Roadmap → footer.
import EngineBoard from "./components/EngineBoard";
import Architecture from "./components/Architecture";
import Journey from "./components/Journey";
import Improvements from "./components/Improvements";
import Strength from "./components/Strength";
import Roadmap from "./components/Roadmap";

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
            <a href="#journey">Journey</a>
            <a href="#improvements">Improvements</a>
            <a href="#strength">Strength</a>
            <a href="#roadmap">Roadmap</a>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="hero hero-grid">
        <div className="wrap">
          <span className="kicker">An AlphaZero-style engine, built from scratch</span>
          <h1>
            A homemade chess brain you can <span className="accent">play against</span> — and
            second-guess.
          </h1>
          <p className="lead">
            A self-trained policy/value network and Monte-Carlo Tree Search, written from scratch in
            PyTorch. Load a position, play Stockfish at any strength, and watch my network's move
            picks sit side-by-side with Stockfish's — no search, just the raw neural net thinking
            out loud. Training has been limited by the compute I had access to, so it's still a
            modest player — the point here is the build, not the rating.
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
              Play &amp; compare ↓
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
            <span className="kicker">Play &amp; compare</span>
            <h2 className="section-title">You vs Stockfish — with a backseat driver</h2>
            <p className="section-sub">
              Drag a piece to move. Stockfish replies at your chosen Elo. After every move, my
              network's top pick (green) and Stockfish's best (blue) are drawn right on the board,
              with the full breakdown on the right.
            </p>
          </div>
          <EngineBoard />
        </div>
      </section>

      <Architecture />
      <Journey />
      <Improvements />
      <Strength />
      <Roadmap />

      <footer className="footer">
        <div className="wrap footer-inner">
          <div className="muted">
            chess-nn — a from-scratch AlphaZero-style chess engine in PyTorch.
          </div>
          <div className="mono">
            net = raw policy · Stockfish 16 NNUE · runs fully client-side
          </div>
        </div>
      </footer>
    </div>
  );
}
