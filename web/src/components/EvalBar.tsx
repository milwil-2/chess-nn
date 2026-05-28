// Vertical eval bar driven by Stockfish centipawns (always White-relative),
// with an optional second tick for the net's WDL scalar.
interface EvalBarProps {
  /** Centipawns from WHITE's POV (positive = white better). null when mate/unknown. */
  whiteCp: number | null;
  /** Signed mate-in-N from WHITE's POV. null when no mate. */
  whiteMate: number | null;
  /** Net P(win)-P(loss) from WHITE's POV, [-1,1]. null hides the tick. */
  netScalarWhite?: number | null;
}

/** Logistic squash of centipawns -> [0,1] white-advantage fraction. */
function cpToFraction(cp: number): number {
  // ~ matches lichess eval bar feel; 4 pawns ≈ 0.9.
  return 1 / (1 + Math.pow(10, -cp / 400));
}

export default function EvalBar({ whiteCp, whiteMate, netScalarWhite }: EvalBarProps) {
  let frac = 0.5;
  let topLabel = "";
  let botLabel = "";

  if (whiteMate !== null) {
    frac = whiteMate > 0 ? 0.98 : 0.02;
    const m = `M${Math.abs(whiteMate)}`;
    if (whiteMate > 0) botLabel = m;
    else topLabel = m;
  } else if (whiteCp !== null) {
    frac = cpToFraction(whiteCp);
    const pawns = (whiteCp / 100).toFixed(1);
    if (whiteCp >= 0) botLabel = `+${pawns}`;
    else topLabel = pawns;
  }

  const whiteHeightPct = frac * 100;

  // Net tick: map [-1,1] white scalar onto bar bottom-offset percentage.
  const netTick =
    netScalarWhite === null || netScalarWhite === undefined
      ? null
      : ((netScalarWhite + 1) / 2) * 100;

  return (
    <div className="evalbar" title="Stockfish evaluation (White's perspective)">
      <div style={{ flex: 1 }} />
      <div className="white-fill" style={{ height: `${whiteHeightPct}%` }} />
      {topLabel && <span className="label top">{topLabel}</span>}
      {botLabel && <span className="label bot">{botLabel}</span>}
      {netTick !== null && (
        <span
          className="net-tick"
          style={{ bottom: `${netTick}%` }}
          title="chess-nn net value (P(win)−P(loss))"
        />
      )}
    </div>
  );
}
