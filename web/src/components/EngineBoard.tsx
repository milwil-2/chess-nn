// Game state is (startFen, movesUci[]) so the net rebuilds its 8-frame
// history encoding from the real move list; chess.js is derived for legality
// and SAN. Loading a raw FEN resets to (that fen, []).
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Chessboard } from "react-chessboard";
import type { Arrow, PieceDropHandlerArgs } from "react-chessboard";
import { Chess, validateFen } from "chess.js";
import { createStockfish } from "../engine/stockfish";
import { createNetEngine } from "../engine/net";
import type { NetResult, SfAnalysis } from "../engine/types";
import type { Deviation } from "./ComparisonPanel";
import ComparisonPanel from "./ComparisonPanel";
import EvalBar from "./EvalBar";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const ELO_PRESETS = [1350, 1500, 1800, 2200, 2800];
// Single-threaded WASM SF reaches its depth at very different speeds across
// positions, so the analyst is bounded by movetime — keeps the panel snappy
// and stops autoplay from stalling when SF chains into a complex middlegame.
// 1500ms reaches roughly depth 12-15 on opening positions with MultiPV=3.
const ANALYSIS_MOVETIME_MS = 1500;
const AUTOPLAY_DELAY_MS = 1800;
const NET_ACCENT = "#7ecfa0"; // var(--accent)
const SF_ACCENT = "#6aa0ff"; // var(--accent-2)

type Color = "white" | "black";

function gameFrom(startFen: string, moves: string[]): Chess | null {
  try {
    const g = new Chess(startFen);
    for (const uci of moves) {
      g.move({
        from: uci.slice(0, 2),
        to: uci.slice(2, 4),
        promotion: uci.length > 4 ? uci[4] : undefined,
      });
    }
    return g;
  } catch {
    return null;
  }
}

function toWhiteCp(scoreCp: number | null, whiteToMove: boolean): number | null {
  if (scoreCp === null) return null;
  return whiteToMove ? scoreCp : -scoreCp;
}
function toWhiteMate(mate: number | null, whiteToMove: boolean): number | null {
  if (mate === null) return null;
  return whiteToMove ? mate : -mate;
}

export default function EngineBoard() {
  const opponentRef = useRef(createStockfish());
  const analystRef = useRef(createStockfish());
  const netRef = useRef(createNetEngine());

  const [startFen, setStartFen] = useState(START_FEN);
  const [moves, setMoves] = useState<string[]>([]);
  const [humanColor, setHumanColor] = useState<Color>("white");
  const [orientation, setOrientation] = useState<Color>("white");
  const [elo, setElo] = useState<number>(1500);
  const [showHints, setShowHints] = useState(true);

  const [fenInput, setFenInput] = useState(START_FEN);
  const [fenError, setFenError] = useState<string | null>(null);

  const [engineBooting, setEngineBooting] = useState(false);
  const [opponentThinking, setOpponentThinking] = useState(false);
  const [sf, setSf] = useState<SfAnalysis | null>(null);
  const [sfLoading, setSfLoading] = useState(false);
  const [netResult, setNetResult] = useState<NetResult | null>(null);
  const [netLoading, setNetLoading] = useState(false);
  const [netReady, setNetReady] = useState(false);
  const [deviation, setDeviation] = useState<Deviation | null>(null);
  const [autoplay, setAutoplay] = useState(false);
  const [driftLog, setDriftLog] = useState<{ cpLoss: number }[]>([]);

  const game = useMemo(() => gameFrom(startFen, moves) ?? new Chess(startFen), [startFen, moves]);
  const fen = game.fen();
  const whiteToMove = game.turn() === "w";
  const isGameOver = game.isGameOver();
  const humanTurn = !isGameOver && game.turn() === (humanColor === "white" ? "w" : "b");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await netRef.current.ready();
        const probe = await netRef.current.evaluate(START_FEN, []);
        if (!cancelled) setNetReady(probe.topMoves.length > 0);
      } catch {
        if (!cancelled) setNetReady(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const opp = opponentRef.current;
    const analyst = analystRef.current;
    return () => {
      opp.dispose();
      analyst.dispose();
    };
  }, []);

  useEffect(() => {
    void analystRef.current.setElo(null);
  }, []);

  useEffect(() => {
    void opponentRef.current.setElo(elo);
  }, [elo]);

  // Guard against stale async results overwriting a newer position.
  const analysisToken = useRef(0);
  useEffect(() => {
    const token = ++analysisToken.current;
    const currentFen = fen;
    const startSnapshot = startFen;
    const movesSnapshot = moves;

    if (isGameOver) {
      setSf(null);
      setNetResult(null);
      setSfLoading(false);
      setNetLoading(false);
      return;
    }

    if (netReady) {
      setNetLoading(true);
      setNetResult(null);
      netRef.current
        .evaluate(startSnapshot, movesSnapshot)
        .then((res) => {
          if (analysisToken.current === token) {
            setNetResult(res);
            setNetLoading(false);
          }
        })
        .catch(() => {
          if (analysisToken.current === token) setNetLoading(false);
        });
    } else {
      setNetResult(null);
    }

    setSfLoading(true);
    setSf(null);
    setEngineBooting(true);
    analystRef.current
      .analyze(currentFen, { movetimeMs: ANALYSIS_MOVETIME_MS })
      .then((res) => {
        if (analysisToken.current === token) {
          setSf(res);
          setSfLoading(false);
          setEngineBooting(false);
        }
      })
      .catch(() => {
        if (analysisToken.current === token) {
          setSfLoading(false);
          setEngineBooting(false);
        }
      });
  }, [fen, startFen, moves, isGameOver, netReady]);

  // Deviation: once sf and netResult have landed for the current position,
  // look up the net's top move in Stockfish's MultiPV alternatives. If it's in
  // the top-N lines we get a real cp loss; if not, the net's pick is "outside
  // Stockfish's top N" and we render that instead of a number. All synchronous
  // — no second SF call (the `searchmoves` variant hangs in this WASM build).
  useEffect(() => {
    if (!sf || !netResult || netResult.topMoves.length === 0 || isGameOver) {
      setDeviation(null);
      return;
    }
    const netUci = netResult.topMoves[0].uci;
    const sfUci = sf.best.uci;
    const netSan = netResult.topMoves[0].san;
    const sfSan = sf.best.san;
    if (netUci === sfUci) {
      setDeviation({ agree: true, netUci, netSan, sfUci, sfSan, cpLoss: 0, mateNote: null });
      return;
    }
    const altMatch = sf.alternatives.find((a) => a.uci === netUci);
    let cpLoss: number | null = null;
    let mateNote: string | null = null;
    if (sf.mate !== null) {
      const n = Math.abs(sf.mate);
      mateNote = sf.mate > 0 ? `net misses mate-in-${n}` : `net is already in mate-in-${n}`;
    } else if (!altMatch) {
      mateNote = `outside Stockfish's top ${sf.alternatives.length} lines`;
    } else if (altMatch.mate !== null && altMatch.mate < 0) {
      mateNote = `net's move walks into mate-in-${Math.abs(altMatch.mate)}`;
    } else if (sf.scoreCp !== null && altMatch.scoreCp !== null) {
      cpLoss = Math.max(0, sf.scoreCp - altMatch.scoreCp);
    }
    setDeviation({ agree: false, netUci, netSan, sfUci, sfSan, cpLoss, mateNote });
  }, [sf, netResult, isGameOver]);

  const replyToken = useRef(0);
  useEffect(() => {
    if (isGameOver || humanTurn) return;
    const token = ++replyToken.current;
    const movesAtRequest = moves;
    const fenAtRequest = fen;
    setOpponentThinking(true);
    setEngineBooting(true);
    opponentRef.current
      .bestMove(fenAtRequest, { movetimeMs: 700 })
      .then((mv) => {
        if (replyToken.current !== token) return;
        setOpponentThinking(false);
        setEngineBooting(false);
        if (!mv.uci || mv.uci === "(none)") return;
        // Validate legality against the live position before committing.
        const check = gameFrom(startFen, movesAtRequest);
        if (!check) return;
        try {
          check.move({
            from: mv.uci.slice(0, 2),
            to: mv.uci.slice(2, 4),
            promotion: mv.uci.length > 4 ? mv.uci[4] : undefined,
          });
        } catch {
          return;
        }
        setMoves((prev) =>
          prev === movesAtRequest ? [...prev, mv.uci] : prev,
        );
      })
      .catch(() => {
        if (replyToken.current === token) {
          setOpponentThinking(false);
          setEngineBooting(false);
        }
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fen, humanTurn, isGameOver]);

  // Autoplay loop: when on and it's the human-side's turn, wait for analysis +
  // deviation to settle, pause AUTOPLAY_DELAY_MS so the visitor can read the
  // banner, then commit the net's top move. Cancelling on any dep change
  // (toggle off, position change, autoplay flip) clears the pending timer.
  useEffect(() => {
    if (!autoplay) return;
    if (isGameOver || !humanTurn || opponentThinking) return;
    if (!netResult || netResult.topMoves.length === 0) return;
    if (!sf || sfLoading || netLoading) return;

    const netUci = netResult.topMoves[0].uci;
    const movesAtRequest = moves;
    const driftEntry = { cpLoss: deviation?.cpLoss ?? 0 };

    const timer = setTimeout(() => {
      const check = gameFrom(startFen, movesAtRequest);
      if (!check) {
        setAutoplay(false);
        return;
      }
      try {
        check.move({
          from: netUci.slice(0, 2),
          to: netUci.slice(2, 4),
          promotion: netUci.length > 4 ? netUci[4] : undefined,
        });
      } catch {
        setAutoplay(false);
        return;
      }
      setMoves((prev) => (prev === movesAtRequest ? [...prev, netUci] : prev));
      setDriftLog((prev) => [...prev, driftEntry]);
    }, AUTOPLAY_DELAY_MS);

    return () => clearTimeout(timer);
  }, [
    autoplay,
    humanTurn,
    isGameOver,
    opponentThinking,
    netResult,
    sf,
    sfLoading,
    netLoading,
    deviation,
    moves,
    startFen,
  ]);

  const onPieceDrop = useCallback(
    ({ sourceSquare, targetSquare }: PieceDropHandlerArgs): boolean => {
      if (!targetSquare) return false;
      if (!humanTurn || opponentThinking) return false;
      const probe = gameFrom(startFen, moves);
      if (!probe) return false;
      // Auto-queen promotions in the UI.
      const piece = probe.get(sourceSquare as never);
      const isPromotion =
        piece?.type === "p" &&
        ((piece.color === "w" && targetSquare[1] === "8") ||
          (piece.color === "b" && targetSquare[1] === "1"));
      try {
        const mv = probe.move({
          from: sourceSquare,
          to: targetSquare,
          promotion: isPromotion ? "q" : undefined,
        });
        const uci = mv.from + mv.to + (mv.promotion ?? "");
        setMoves((prev) => [...prev, uci]);
        return true;
      } catch {
        return false;
      }
    },
    [humanTurn, opponentThinking, startFen, moves],
  );

  const newGame = useCallback(() => {
    analysisToken.current++;
    replyToken.current++;
    setStartFen(START_FEN);
    setFenInput(START_FEN);
    setFenError(null);
    setMoves([]);
    setSf(null);
    setNetResult(null);
    setDeviation(null);
    setDriftLog([]);
  }, []);

  const loadFen = useCallback(() => {
    const trimmed = fenInput.trim();
    const result = validateFen(trimmed);
    if (!result.ok) {
      setFenError(result.error ?? "Invalid FEN");
      return;
    }
    analysisToken.current++;
    replyToken.current++;
    setFenError(null);
    setStartFen(trimmed);
    setMoves([]);
    setSf(null);
    setNetResult(null);
    setDeviation(null);
    setDriftLog([]);
    const stm: Color = trimmed.split(/\s+/)[1] === "b" ? "black" : "white";
    setHumanColor(stm);
    setOrientation(stm);
  }, [fenInput]);

  const undo = useCallback(() => {
    if (moves.length === 0) return;
    analysisToken.current++;
    replyToken.current++;
    setMoves((prev) => {
      const next = prev.slice(0, -1);
      return next;
    });
  }, [moves.length]);

  const flip = useCallback(() => {
    setOrientation((o) => (o === "white" ? "black" : "white"));
  }, []);

  const chooseColor = useCallback((c: Color) => {
    setHumanColor(c);
    setOrientation(c);
  }, []);

  const arrows: Arrow[] = useMemo(() => {
    if (!showHints || isGameOver) return [];
    const out: Arrow[] = [];
    const netTop = netResult?.topMoves[0]?.uci;
    const sfBest = sf?.best.uci;
    if (netTop && netTop.length >= 4) {
      out.push({
        startSquare: netTop.slice(0, 2),
        endSquare: netTop.slice(2, 4),
        color: NET_ACCENT,
      });
    }
    if (sfBest && sfBest.length >= 4 && sfBest !== netTop) {
      out.push({
        startSquare: sfBest.slice(0, 2),
        endSquare: sfBest.slice(2, 4),
        color: SF_ACCENT,
      });
    }
    return out;
  }, [showHints, isGameOver, netResult, sf]);

  let status = "";
  let gameOverFlag = false;
  if (game.isCheckmate()) {
    status = `Checkmate — ${whiteToMove ? "Black" : "White"} wins`;
    gameOverFlag = true;
  } else if (game.isStalemate()) {
    status = "Stalemate — draw";
    gameOverFlag = true;
  } else if (game.isInsufficientMaterial()) {
    status = "Draw — insufficient material";
    gameOverFlag = true;
  } else if (game.isThreefoldRepetition()) {
    status = "Draw — threefold repetition";
    gameOverFlag = true;
  } else if (game.isDraw()) {
    status = "Draw — 50-move rule";
    gameOverFlag = true;
  } else if (opponentThinking) {
    status = "Stockfish is thinking…";
  } else if (engineBooting && !sf) {
    status = "Booting engine…";
  } else {
    status = `${whiteToMove ? "White" : "Black"} to move${game.inCheck() ? " — check" : ""}`;
  }

  const whiteCp = sf ? toWhiteCp(sf.scoreCp, whiteToMove) : null;
  const whiteMate = sf ? toWhiteMate(sf.mate, whiteToMove) : null;
  const netScalarWhite =
    netResult && netReady ? (whiteToMove ? netResult.scalar : -netResult.scalar) : null;

  const boardOptions = {
    position: fen,
    boardOrientation: orientation,
    onPieceDrop,
    arrows,
    allowDragging: humanTurn && !opponentThinking && !autoplay,
    id: "engine-board",
    darkSquareStyle: { backgroundColor: "#2a2a3c" },
    lightSquareStyle: { backgroundColor: "#3d3d52" },
    boardStyle: { borderRadius: "8px" },
    animationDurationInMs: 180,
  };

  return (
    <div className="play">
      <div className="board-col">
        <EvalBar whiteCp={whiteCp} whiteMate={whiteMate} netScalarWhite={netScalarWhite} />
        <div className="board-stage">
          <div className="board-frame">
            <Chessboard options={boardOptions} />
          </div>
          <div className={`board-status${gameOverFlag ? " gameover" : ""}`}>
            <span
              className="turn-dot"
              style={{ background: whiteToMove ? "#e9ecf5" : "#0a0a0f" }}
            />
            {opponentThinking || (engineBooting && !sf) ? <span className="spin" /> : null}
            {status}
          </div>

          <div className="controls">
            <div className="controls-row">
              <div className="grow">
                <label className="field-label" htmlFor="fen">
                  position (FEN)
                </label>
                <input
                  id="fen"
                  className={`input${fenError ? " invalid" : ""}`}
                  value={fenInput}
                  spellCheck={false}
                  onChange={(e) => setFenInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && loadFen()}
                  placeholder="paste a FEN…"
                />
              </div>
              <button className="btn" onClick={loadFen}>
                Load
              </button>
            </div>
            {fenError && (
              <span className="muted-note" style={{ color: "var(--warn)" }}>
                {fenError}
              </span>
            )}

            <div>
              <span className="field-label">opponent strength — Stockfish</span>
              <div className="controls-row">
                <div className="elo-presets">
                  {ELO_PRESETS.map((p) => (
                    <button
                      key={p}
                      className={`chip${elo === p ? " active" : ""}`}
                      onClick={() => setElo(p)}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
              <div className="controls-row" style={{ marginTop: 8 }}>
                <input
                  className="range"
                  type="range"
                  min={800}
                  max={3000}
                  step={50}
                  value={elo}
                  onChange={(e) => setElo(Number(e.target.value))}
                />
                <span className="mono" style={{ fontSize: 13, width: 70, textAlign: "right" }}>
                  {elo} Elo
                </span>
              </div>
            </div>

            <div className="controls-row">
              <div>
                <span className="field-label">play as</span>
                <div className="seg">
                  <button
                    className={humanColor === "white" ? "active" : ""}
                    onClick={() => chooseColor("white")}
                  >
                    White
                  </button>
                  <button
                    className={humanColor === "black" ? "active" : ""}
                    onClick={() => chooseColor("black")}
                  >
                    Black
                  </button>
                </div>
              </div>
              <div style={{ flex: 1 }} />
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={showHints}
                  onChange={(e) => setShowHints(e.target.checked)}
                />
                show hint arrows
              </label>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={autoplay}
                  onChange={(e) => setAutoplay(e.target.checked)}
                />
                autoplay (chess-nn plays)
              </label>
            </div>

            <div className="controls-row">
              <button className="btn primary" onClick={newGame}>
                New game
              </button>
              <button className="btn" onClick={flip}>
                Flip board
              </button>
              <button className="btn ghost" onClick={undo} disabled={moves.length === 0}>
                Undo
              </button>
            </div>
          </div>
        </div>
      </div>

      <div>
        <ComparisonPanel
          netResult={netResult}
          netLoading={netLoading}
          netReady={netReady}
          sf={sf}
          sfLoading={sfLoading}
          active={!isGameOver}
          deviation={deviation}
          autoplay={autoplay}
          autoplayDelayMs={AUTOPLAY_DELAY_MS}
          driftLog={driftLog}
        />
      </div>
    </div>
  );
}
