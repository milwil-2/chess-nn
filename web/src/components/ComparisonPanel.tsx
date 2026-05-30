// Side-by-side panel:
//  left  = chess-nn raw policy (top-3 + WDL bar), labeled "no search"
//  right = Stockfish at full strength (best move + eval)
// The banner above the cards quantifies how much weaker the network's pick is
// when the two disagree; it's also given more visual weight while autoplay is
// driving the pace.
import type { NetResult, SfAnalysis } from "../engine/types";

/** Comparison of the network's top move against Stockfish's best, for the
 *  current position. cpLoss is non-negative and only set when both sides have
 *  centipawn scores. mateNote replaces the number when mate is involved. */
export interface Deviation {
  agree: boolean;
  netUci: string;
  netSan: string;
  sfUci: string;
  sfSan: string;
  cpLoss: number | null;
  mateNote: string | null;
}

interface ComparisonPanelProps {
  netResult: NetResult | null;
  netLoading: boolean;
  netReady: boolean; // false while the net engine is still a stub / not loaded
  sf: SfAnalysis | null;
  sfLoading: boolean;
  /** true when it's the human's turn / a position to think about exists. */
  active: boolean;
  deviation: Deviation | null;
  autoplay: boolean;
  autoplayDelayMs: number;
  driftLog: { cpLoss: number }[];
}

function formatEval(sf: SfAnalysis): { big: string; unit: string } {
  if (sf.mate !== null) {
    return { big: `#${sf.mate > 0 ? "" : "-"}${Math.abs(sf.mate)}`, unit: "mate" };
  }
  if (sf.scoreCp !== null) {
    const pawns = sf.scoreCp / 100;
    const sign = pawns > 0 ? "+" : "";
    return { big: `${sign}${pawns.toFixed(2)}`, unit: "pawns (stm)" };
  }
  return { big: "—", unit: "" };
}

function SkeletonRows({ n }: { n: number }) {
  return (
    <>
      {Array.from({ length: n }).map((_, i) => (
        <div key={i} className="skeleton skel-line" style={{ width: `${90 - i * 12}%` }} />
      ))}
    </>
  );
}

function formatCp(cp: number): string {
  // Anything inside ~5 cp is noise at MultiPV/movetime; call it equal.
  if (cp < 5) return "≈ same eval";
  if (cp < 100) return `−${(cp / 100).toFixed(2)} pawns`;
  return `−${cp} cp`;
}

function DeviationBanner({
  deviation,
  autoplay,
  autoplayDelayMs,
}: {
  deviation: Deviation;
  autoplay: boolean;
  autoplayDelayMs: number;
}) {
  if (deviation.agree) {
    return (
      <div className={`agree-banner${autoplay ? " autoplay" : ""}`}>
        <span>◆</span> Network and Stockfish picked the same move:{" "}
        <strong>{deviation.netSan}</strong>
      </div>
    );
  }

  const lossText = deviation.mateNote
    ? deviation.mateNote
    : deviation.cpLoss !== null
      ? formatCp(deviation.cpLoss)
      : "—";
  return (
    <div className={`agree-banner disagree-banner${autoplay ? " autoplay" : ""}`}>
      <span>◇</span> Network plays <strong>{deviation.netSan}</strong>, Stockfish prefers{" "}
      <strong>{deviation.sfSan}</strong>
      <span className="cp-loss"> · {lossText}</span>
      {autoplay && (
        <span
          className="countdown"
          style={{ animationDuration: `${autoplayDelayMs}ms` }}
          aria-hidden
        />
      )}
    </div>
  );
}

export default function ComparisonPanel({
  netResult,
  netLoading,
  netReady,
  sf,
  sfLoading,
  active,
  deviation,
  autoplay,
  autoplayDelayMs,
  driftLog,
}: ComparisonPanelProps) {
  const netTop = netResult?.topMoves.slice(0, 3) ?? [];
  const maxProb = netTop.length ? Math.max(...netTop.map((m) => m.prob ?? 0)) : 1;

  const wdl = netResult?.wdl ?? [0, 0, 0];
  const wdlSum = wdl[0] + wdl[1] + wdl[2] || 1;
  const wPct = (wdl[0] / wdlSum) * 100;
  const dPct = (wdl[1] / wdlSum) * 100;
  const lPct = (wdl[2] / wdlSum) * 100;

  const driftSum = driftLog.reduce((s, d) => s + d.cpLoss, 0);
  const driftCount = driftLog.length;

  return (
    <div className="compare">
      {active && deviation ? (
        <DeviationBanner
          deviation={deviation}
          autoplay={autoplay}
          autoplayDelayMs={autoplayDelayMs}
        />
      ) : null}

      <div className="compare-grid">
        <div className="cmp-card net">
          <div className="cmp-title">
            <span className="swatch net" /> chess-nn
          </div>
          <div className="cmp-sub">raw policy, no search</div>

          {!netReady ? (
            <div className="empty-hint">
              Network not loaded in this build.
              <br />
              <span className="muted-note">
                The ONNX policy/value network wires in here once its bundle ships.
              </span>
            </div>
          ) : !active ? (
            <div className="empty-hint">Make a move to see the network's policy.</div>
          ) : netLoading && !netResult ? (
            <SkeletonRows n={3} />
          ) : netTop.length === 0 ? (
            <div className="empty-hint">No policy output for this position.</div>
          ) : (
            <>
              <div className="movebar">
                {netTop.map((m) => {
                  const prob = m.prob ?? 0;
                  return (
                    <div className="row" key={m.uci}>
                      <span className="san">{m.san}</span>
                      <span className="track">
                        <span
                          className="fill"
                          style={{ width: `${maxProb ? (prob / maxProb) * 100 : 0}%` }}
                        />
                      </span>
                      <span className="pct">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
              <div className="wdl">
                <span className="field-label">value head — win / draw / loss (stm)</span>
                <div className="wdl-bar">
                  <span className="wdl-seg win" style={{ width: `${wPct}%` }}>
                    {wPct > 14 ? `${Math.round(wPct)}` : ""}
                  </span>
                  <span className="wdl-seg draw" style={{ width: `${dPct}%` }}>
                    {dPct > 14 ? `${Math.round(dPct)}` : ""}
                  </span>
                  <span className="wdl-seg loss" style={{ width: `${lPct}%` }}>
                    {lPct > 14 ? `${Math.round(lPct)}` : ""}
                  </span>
                </div>
              </div>
              {driftCount > 0 && (
                <div className="drift-line">
                  Net drift: <strong>−{(driftSum / 100).toFixed(2)} pawns</strong> across{" "}
                  {driftCount} {driftCount === 1 ? "move" : "moves"}
                </div>
              )}
            </>
          )}
        </div>

        <div className="cmp-card sf">
          <div className="cmp-title">
            <span className="swatch sf" /> Stockfish
          </div>
          <div className="cmp-sub">full strength reference</div>

          {!active ? (
            <div className="empty-hint">Make a move to request analysis.</div>
          ) : sfLoading && !sf ? (
            <SkeletonRows n={2} />
          ) : !sf ? (
            <div className="empty-hint">No analysis yet.</div>
          ) : (
            <>
              <span className="field-label">evaluation</span>
              <div className="cmp-eval">
                {formatEval(sf).big} <span className="unit">{formatEval(sf).unit}</span>
              </div>
              <span className="field-label" style={{ marginTop: 12, display: "block" }}>
                best move · depth {sf.depth}
              </span>
              <div className="cmp-best">{sf.best.san || sf.best.uci || "—"}</div>
              {sf.pv.length > 1 && (
                <div className="cmp-pv">
                  pv&nbsp;&nbsp;{sf.pv.slice(0, 8).join("  ")}
                </div>
              )}
            </>
          )}
        </div>
      </div>

      <p className="muted-note">
        The network shows its <strong>raw move probabilities</strong> from a single forward pass,
        with no tree search. Stockfish runs a full search to fixed depth as a calibrated reference.
      </p>
    </div>
  );
}
