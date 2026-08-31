// Strength: single-day gauntlet snapshot with CI error bars. Not a
// training curve.
import eloData from "../data/elo.json";

interface Row {
  level: number | null;
  tc: string;
  elo: number;
  ciLow: number;
  ciHigh: number;
  games: number;
}

const AXIS_MIN = 0;
const AXIS_MAX = 2100;

function pct(elo: number) {
  return ((Math.max(AXIS_MIN, Math.min(AXIS_MAX, elo)) - AXIS_MIN) / (AXIS_MAX - AXIS_MIN)) * 100;
}

function StrengthRow({ row, headline }: { row: Row; headline?: boolean }) {
  const lo = pct(row.ciLow);
  const hi = pct(row.ciHigh);
  const mid = pct(row.elo);
  const label = headline
    ? `Full engine`
    : row.level === null
      ? `Aggregate`
      : `vs SF level ${row.level}`;
  return (
    <div className="str-row">
      <div className="str-label">
        {label}
        <div className="sub">
          {row.tc} · {row.games} games
        </div>
      </div>
      <div className="str-track">
        <span className="str-ci" style={{ left: `${lo}%`, width: `${Math.max(0, hi - lo)}%` }} />
        <span className="str-ci-cap" style={{ left: `${lo}%` }} />
        <span className="str-ci-cap" style={{ left: `${hi}%` }} />
        <span className={`str-point${headline ? " headline" : ""}`} style={{ left: `${mid}%` }} />
      </div>
      <div className="str-val">{Math.round(row.elo)}</div>
    </div>
  );
}

export default function Strength() {
  const headline = eloData.headline as Row[];
  const rows = eloData.rows as Row[];

  return (
    <section id="strength" className="section">
      <div className="wrap">
        <div className="section-head">
          <span className="kicker">
            <span className="sec-idx">04</span>Strength
          </span>
          <h2 className="section-title">Gauntlet results</h2>
          <p className="section-sub">
            Current v3_vast strength is roughly 700 Elo on the local Stockfish gauntlet: weak,
            which is what the next training cycle aims to fix. This is a{" "}
            <strong>single gauntlet snapshot</strong> against Stockfish on one day with wide
            confidence intervals, not a training curve. The estimate lands around 700–870 Elo for
            the full engine. Training has been bounded by available compute (limited GPU time and
            dataset size on rented hardware), which is the main cap on strength so far.
          </p>
        </div>

        <div className="strength-chart">
          <span className="field-label">v3_vast vs Stockfish · 2026-05-26 · checkpoint daf568…</span>
          <div className="strength-rows" style={{ marginTop: 14 }}>
            {headline.map((r) => (
              <StrengthRow key={`h-${r.tc}`} row={r} headline />
            ))}
            {rows.map((r) => (
              <StrengthRow key={`l-${r.level}-${r.tc}`} row={r} />
            ))}
          </div>
          <div className="str-axis">
            <span />
            <span className="ticks">
              <span>0</span>
              <span>700</span>
              <span>1400</span>
              <span>2100</span>
            </span>
            <span style={{ textAlign: "right" }}>Elo</span>
          </div>

          <p className="caption">
            Blue points are the full engine (MCTS + inference helpers); green points are per-level
            sweeps. The whiskers are 95% confidence intervals, wide because each data point is
            only a handful of games. The network used in the browser above is{" "}
            <strong>raw policy with no search</strong>, so it plays noticeably weaker than these
            gauntlet numbers.
          </p>
        </div>
      </div>
    </section>
  );
}
