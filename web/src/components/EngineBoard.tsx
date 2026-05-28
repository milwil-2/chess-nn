// =====================================================================
// EngineBoard — the centerpiece.
//
// Game state is held as (startFen, movesUci[]) so the net gets the real move
// history to rebuild its 8-frame board encoding. A chess.js instance is
// derived from those for legality / SAN / game-over. Loading a raw FEN resets
// (startFen = that fen, moves = []).
//
// Flow: human drags a legal move -> push to movesUci -> if it's now the
// opponent's turn, Stockfish@Elo replies via bestMove() -> push. After every
// settled position we query net.evaluate(startFen, moves) AND analyst.analyze()
// and render live. The net's top move (green) and Stockfish's best (blue) are
// drawn as board arrows.
// =====================================================================
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Chessboard } from "react-chessboard";
import type { Arrow, PieceDropHandlerArgs } from "react-chessboard";
import { Chess, validateFen } from "chess.js";
import { createStockfish } from "../engine/stockfish";
import { createNetEngine } from "../engine/net";
import type { NetResult, SfAnalysis } from "../engine/types";
import ComparisonPanel from "./ComparisonPanel";
import EvalBar from "./EvalBar";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const ELO_PRESETS = [1350, 1500, 1800, 2200, 2800];
const ANALYSIS_DEPTH = 15;
const NET_ACCENT = "#7ecfa0"; // var(--accent)
const SF_ACCENT = "#6aa0ff"; // var(--accent-2)

type Color = "white" | "black";

/** Rebuild a Chess from a start FEN + uci move list. Returns null on any illegal step. */
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

/** Flip a side-to-move-relative centipawn score to White's POV. */
function toWhiteCp(scoreCp: number | null, whiteToMove: boolean): number | null {
  if (scoreCp === null) return null;
  return whiteToMove ? scoreCp : -scoreCp;
}
function toWhiteMate(mate: number | null, whiteToMove: boolean): number | null {
  if (mate === null) return null;
  return whiteToMove ? mate : -mate;
}

export default function EngineBoard() {
  // Persistent engines (created once). Net is the parallel agent's stub for now.
  const opponentRef = useRef(createStockfish());
  const analystRef = useRef(createStockfish());
  const netRef = useRef(createNetEngine());

  // Game state.
  const [startFen, setStartFen] = useState(START_FEN);
  const [moves, setMoves] = useState<string[]>([]);
  const [humanColor, setHumanColor] = useState<Color>("white");
  const [orientation, setOrientation] = useState<Color>("white");
  const [elo, setElo] = useState<number>(1500);
  const [showHints, setShowHints] = useState(true);

  // FEN input.
  const [fenInput, setFenInput] = useState(START_FEN);
  const [fenError, setFenError] = useState<string | null>(null);

  // Engine state / results.
  const [engineBooting, setEngineBooting] = useState(false);
  const [opponentThinking, setOpponentThinking] = useState(false);
  const [sf, setSf] = useState<SfAnalysis | null>(null);
  const [sfLoading, setSfLoading] = useState(false);
  const [netResult, setNetResult] = useState<NetResult | null>(null);
  const [netLoading, setNetLoading] = useState(false);
  const [netReady, setNetReady] = useState(false);

  // Derived game.
  const game = useMemo(() => gameFrom(startFen, moves) ?? new Chess(startFen), [startFen, moves]);
  const fen = game.fen();
  const whiteToMove = game.turn() === "w";
  const isGameOver = game.isGameOver();
  const humanTurn = !isGameOver && game.turn() === (humanColor === "white" ? "w" : "b");

  // Probe whether the net engine actually produces output (real impl vs stub).
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

  // Cleanup engines on unmount.
  useEffect(() => {
    const opp = opponentRef.current;
    const analyst = analystRef.current;
    return () => {
      opp.dispose();
      analyst.dispose();
    };
  }, []);

  // Keep the analyst at full strength.
  useEffect(() => {
    void analystRef.current.setElo(null);
  }, []);

  // Push the chosen opponent Elo down to the opponent engine.
  useEffect(() => {
    void opponentRef.current.setElo(elo);
  }, [elo]);

  // ---- Analysis side-effect: after every settled position, query both engines.
  // Guard with a token so a stale async result can't overwrite a newer position.
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

    // Net (raw policy on the real history).
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

    // Stockfish analyst.
    setSfLoading(true);
    setSf(null);
    setEngineBooting(true);
    analystRef.current
      .analyze(currentFen, { depth: ANALYSIS_DEPTH })
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

  // ---- Opponent auto-reply when it's the engine's turn.
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
        // Validate the move is legal in the live position before committing.
        const check = gameFrom(startFen, movesAtRequest);
        if (!check) return;
        try {
          check.move({
            from: mv.uci.slice(0, 2),
            to: mv.uci.slice(2, 4),
            promotion: mv.uci.length > 4 ? mv.uci[4] : undefined,
          });
        } catch {
          return; // illegal — drop it
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

  // ---- Human move via drag.
  const onPieceDrop = useCallback(
    ({ sourceSquare, targetSquare }: PieceDropHandlerArgs): boolean => {
      if (!targetSquare) return false;
      if (!humanTurn || opponentThinking) return false;
      const probe = gameFrom(startFen, moves);
      if (!probe) return false;
      // Auto-queen promotions for the UI (most common choice).
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
        return false; // snapback
      }
    },
    [humanTurn, opponentThinking, startFen, moves],
  );

  // ---- Controls.
  const newGame = useCallback(() => {
    analysisToken.current++;
    replyToken.current++;
    setStartFen(START_FEN);
    setFenInput(START_FEN);
    setFenError(null);
    setMoves([]);
    setSf(null);
    setNetResult(null);
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
    // Orient toward the side to move for convenience.
    const stm: Color = trimmed.split(/\s+/)[1] === "b" ? "black" : "white";
    setHumanColor(stm);
    setOrientation(stm);
  }, [fenInput]);

  const undo = useCallback(() => {
    if (moves.length === 0) return;
    analysisToken.current++;
    replyToken.current++;
    // Undo back to the human's previous turn (pop opponent reply + human move).
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

  // ---- Arrows (net top move + SF best move).
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

  // ---- Board status line.
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
    allowDragging: humanTurn && !opponentThinking,
    id: "engine-board",
    darkSquareStyle: { backgroundColor: "#2a2a3c" },
    lightSquareStyle: { backgroundColor: "#3d3d52" },
    boardStyle: { borderRadius: "8px" },
    animationDurationInMs: 180,
  };

  return (
    <div className="play">
      {/* Board column */}
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

          {/* Controls */}
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

      {/* Comparison column */}
      <div>
        <ComparisonPanel
          netResult={netResult}
          netLoading={netLoading}
          netReady={netReady}
          sf={sf}
          sfLoading={sfLoading}
          active={!isGameOver}
        />
      </div>
    </div>
  );
}
