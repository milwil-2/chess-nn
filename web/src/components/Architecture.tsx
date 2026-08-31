// Architecture: one factual paragraph on the network, plus a mono spec line.
export default function Architecture() {
  return (
    <section id="architecture" className="section">
      <div className="wrap">
        <div className="section-head">
          <span className="kicker">
            <span className="sec-idx">02</span>Architecture
          </span>
          <h2 className="section-title">The network</h2>
        </div>

        <p className="arch-prose">
          The model is a residual convolutional network of about 5.6M parameters. It reads the
          board as 105 planes: 8 history frames of 12 piece planes each, plus 9 metadata planes
          for castling rights, en passant, side to move, the 50-move clock, and repetition. An
          initial 3×3 convolution feeds ten residual blocks of 128 filters, each with
          Squeeze-and-Excitation channel attention, and the tower splits into two heads: a policy
          head over 4672 encoded moves (legal-masked and softmaxed at inference) and a 3-class
          win/draw/loss value head. It is trained in PyTorch on rating-filtered Lichess games
          with an auxiliary Stockfish policy target. The browser demo above runs the raw ONNX
          policy with no search.
        </p>

        <div className="arch-spec">
          105×8×8 → conv 3×3 → 10 × [res 128 + SE] → policy 4672 · value WDL
        </div>
      </div>
    </section>
  );
}
