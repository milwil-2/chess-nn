// Strength section — honest rendering of the single gauntlet snapshot, with
// CI error bars. NOT framed as an upward progression.
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
          <span className="kicker">Strength</span>
          <h2 className="section-title">Honestly: it's weak — for now</h2>
          <p className="section-sub">
            This is a <strong>single gauntlet snapshot</strong> of v3_vast against Stockfish on one
            day, with wide confidence intervals — not a training curve trending upward. The
            estimate lands somewhere around 700–870 Elo for the full engine. Training has been
            bounded by available compute — limited GPU time and dataset size on rented hardware —
            which is the main thing capping strength so far.
          </p>
        </div>

        <div className="panel panel-pad">
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
            sweeps. The whiskers are the 95% confidence intervals — they're enormous because each
            data point is only a handful of games. The network you can play above is{" "}
            <strong>raw policy with no search</strong>, so it plays noticeably weaker than these
            gauntlet numbers.
          </p>
        </div>
      </div>
    </section>
  );
}
