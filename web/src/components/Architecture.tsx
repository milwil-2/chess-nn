// Architecture section — diagram of the network: input planes, residual
// tower, and the policy + value heads.
export default function Architecture() {
  // 105 input planes: 96 piece planes (8 frames × 12) + 9 meta. The mosaic
  // is mostly "piece" with a 4-cell "meta" tail.
  const planeCells = Array.from({ length: 36 }, (_, i) => (i < 32 ? "piece" : "meta"));

  return (
    <section id="architecture" className="section">
      <div className="wrap">
        <div className="section-head">
          <span className="kicker">Architecture</span>
          <h2 className="section-title">Network</h2>
          <p className="section-sub">
            A residual CNN with two heads: a stack of board planes goes through a 10-block residual
            tower with Squeeze-and-Excitation channel attention, then splits into a 4672-move
            policy head and a 3-class win/draw/loss value head. Roughly 5.6M parameters, trained
            from scratch in PyTorch.
          </p>
        </div>

        <div className="arch-flow">
          <div className="arch-stage">
            <span className="tag">Input</span>
            <h4>105 planes</h4>
            <p>
              8 history frames × 12 piece planes (96) + 9 meta planes: castling, en passant,
              side-to-move, 50-move clock, repetition. The board is mirrored when Black is to move,
              so the side-to-move's pieces always sit in a fixed set of channels.
            </p>
            <div className="plane-stack">
              {planeCells.map((kind, i) => (
                <span key={i} className={`plane-cell ${kind}`} />
              ))}
            </div>
          </div>

          <div className="arch-arrow">→</div>

          <div className="arch-stage">
            <span className="tag">Body</span>
            <h4>10 residual blocks · 128 filters</h4>
            <p>
              An initial 3×3 conv, then ten residual blocks at 128 channels, each with a
              Squeeze-and-Excitation module (reduction 4) for cheap channel-wise attention applied
              before the skip-add.
            </p>
            <div className="resblock-row">
              {Array.from({ length: 10 }).map((_, i) => (
                <span key={i} className="resblock" />
              ))}
            </div>
          </div>

          <div className="arch-arrow">→</div>

          <div className="arch-stage">
            <span className="tag">Heads</span>
            <h4>policy + value</h4>
            <div className="heads">
              <div className="head-box policy">
                <div className="ht">Policy → 4672 moves</div>
                <div className="hd">
                  AlphaZero 73-plane × 64-square move encoding. Legal-masked + softmaxed at
                  inference.
                </div>
              </div>
              <div className="head-box value">
                <div className="ht">Value → WDL (3-class)</div>
                <div className="hd">
                  Win / draw / loss from the side-to-move POV; collapses to P(win) − P(loss) for
                  search.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
