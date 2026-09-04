"""Play a build in the browser, then have Stockfish tell you where it went wrong.

    uv run python -m tools.play_gui              # then open http://127.0.0.1:8800
    uv run python -m tools.play_gui --open       # and open the browser for you

Why this exists: every other tool in the repo answers "which build is stronger".
None of them answer "what does it actually *do*". A human sitting across the
board finds a class of bug that self-play never surfaces, because self-play only
plays moves both engines understand -- it never walks into the sideline where the
eval is quietly insane, and it never punts a won ending into a threefold because
neither side knew to avoid it.

Fidelity: the bot is not imported into this process. It is started through
`harness.sandbox.local`, the same call the referee uses, so it speaks the same
JSON protocol over the same pipes, gets the same 60s init budget, the same
watchdog, and the same 4 KB stdout cap. Game termination is the referee's own
logic, including `board.outcome(claim_draw=True)` -- so if your bot shuffles a
won rook ending into a threefold, this board claims the draw against it exactly
as the platform would, instead of politely letting the game continue.

The clock is deliberately splittable, because "play the bot" and "test the bot"
want opposite things:

    Match clock   real 120s + 0.5s, deducted by wall clock, flag enforced.
                  This is the one that catches time-management bugs.
    Fixed clock   the bot is told the same time_left_ms on every move and is
                  never flagged. Use it to see how it plays at a chosen budget
                  without the clock manager confusing the picture.
    Your clock    off by default -- take as long as you like; the bot's clock is
                  independent of yours.

Review runs Stockfish over the finished game at N+1 positions rather than 2N:
the eval after your move is the eval before the reply. Stockfish is located via
--engine, then $SPAR_ENGINE, then PATH. It is only ever a spectator here; nothing
this file touches goes anywhere near submission.zip.

Also loads a PGN, so a ladder game you lost can be pasted in and autopsied.
"""

from __future__ import annotations

import argparse
import atexit
import io
import json
import os
import queue
import random
import shutil
import signal
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import chess
import chess.engine
import chess.pgn
import chess.svg

from harness.rules import BASE_MS, INCREMENT_MS, INIT_BUDGET_S, PLY_CAP, WATCHDOG_GRACE_MS
from harness.sandbox import Agent, AgentFailure, _parse_move, _pipe

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

# Centipawn loss thresholds for the review tags. Tighter than a human-facing site
# would use: at this strength a 60cp giveaway is a real bug, not a rounding error.
TAG_THRESHOLDS: tuple[tuple[int, str], ...] = ((25, "ok"), (75, "inaccuracy"), (150, "mistake"))
BLUNDER_TAG = "blunder"
MATE_SCORE = 100_000
EVAL_CLAMP = 2_000  # differencing raw mate scores produces meaningless six-figure losses
LOG_TAIL_BYTES = 48_000
HARD_THINK_CAP_S = 600.0  # a hung bot should stall one move, not the whole server


# --------------------------------------------------------------------------- agents


def make_limit(kind: str, value: int) -> chess.engine.Limit:
    return {
        "depth": chess.engine.Limit(depth=value),
        "nodes": chess.engine.Limit(nodes=value),
        "movetime": chess.engine.Limit(time=value / 1000.0),
    }.get(kind, chess.engine.Limit(depth=12))


class GuiAgent(Agent):
    """A harness Agent that will also tell you what it has been printing.

    `Agent._drain` cannot be reused for this: it throws away stdout chunks, and
    stdout is the protocol stream. `pump` mirrors `_await_line`'s dispatch
    instead, so a diagnostic read can never eat a move.
    """

    def pump(self) -> None:
        while True:
            try:
                name, chunk = self._chunks.get_nowait()
            except queue.Empty:
                break
            if name == "stderr":
                self._tail += chunk
            elif not chunk:
                self._chunks.put((name, chunk))  # EOF is terminal; leave it for move()
                break
            else:
                self._buffer += chunk
        if len(self._tail) > LOG_TAIL_BYTES:
            self._tail = self._tail[-LOG_TAIL_BYTES:]

    def log(self) -> str:
        self.pump()
        return self._tail.decode("utf-8", "replace")

    def ask(self, fen: str, time_left_ms: int, watchdog_s: float) -> str:
        """`Agent.move`, but the watchdog is ours rather than derived from the clock.

        In fixed-clock mode `time_left_ms` is a fiction we hand the bot, so it
        cannot also be the deadline -- a bot told it has an hour would be allowed
        to hang for an hour.
        """
        if self._process is None:
            raise RuntimeError("agent moved before start")
        request = json.dumps({"fen": fen, "time_left_ms": time_left_ms}).encode()
        try:
            _pipe(self._process.stdin).write(request + b"\n")
        except BrokenPipeError:
            raise AgentFailure("crash") from None
        line = self._await_line(time.monotonic() + watchdog_s)
        if line is None:
            raise AgentFailure("flag")
        return _parse_move(line)


def discover_agents() -> list[dict[str, str]]:
    """Everything playable, submission first."""
    found: list[dict[str, str]] = []
    if (REPO / "agent.py").exists():
        found.append({"path": ".", "label": "submission (root agent.py)", "group": "ship"})
    for group, directory in (("versions", REPO / "versions"), ("baselines", REPO / "baselines")):
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if (entry / "agent.py").exists():
                found.append(
                    {
                        "path": str(entry.relative_to(REPO)),
                        "label": entry.name,
                        "group": group,
                    }
                )
    return found


def materialise() -> None:
    """versions/ and baselines/ are build products, not repo contents."""
    try:
        from tools import agents as agent_builder

        agent_builder.build(None, check=False)
    except Exception as error:  # a stale layout should not stop you playing
        print(f"could not materialise agents ({error}); using whatever is on disk", file=sys.stderr)


# --------------------------------------------------------------------------- openings


def opening_positions() -> list[str]:
    try:
        from tools.openings import OPENINGS

        return list(OPENINGS)
    except Exception:
        return []


# --------------------------------------------------------------------------- session


@dataclass
class Seat:
    kind: str = "human"  # "human" | "bot"
    path: str = ""
    label: str = "You"
    agent: GuiAgent | None = None
    failure: str | None = None


@dataclass
class PlayedMove:
    ply: int
    move_no: int
    san: str
    uci: str
    side: str
    ms: float
    clock_after: float | None
    fen: str = ""  # the position this move produced, so the board can travel back to it
    note: str | None = None


@dataclass
class Review:
    running: bool = False
    done: int = 0
    total: int = 0
    error: str | None = None
    engine: str = ""
    limit: str = ""
    rows: list[dict] = field(default_factory=list)
    acpl: dict[str, float | None] = field(default_factory=dict)


class Session:
    def __init__(self, engine_path: str | None, sf_threads: int, sf_hash: int) -> None:
        self.lock = threading.RLock()
        self.generation = 0
        self.version = 0
        self.phase = "idle"  # idle | starting | playing | over | review
        self.board = chess.Board()
        self.start_fen = chess.STARTING_FEN
        self.seats: dict[chess.Color, Seat] = {chess.WHITE: Seat(), chess.BLACK: Seat()}
        self.clock = {chess.WHITE: float(BASE_MS), chess.BLACK: float(BASE_MS)}
        self.clock_enabled = {chess.WHITE: False, chess.BLACK: False}
        self.base_ms = BASE_MS
        self.increment_ms = INCREMENT_MS
        self.bot_clock_mode = "match"  # match | fixed
        self.bot_fixed_ms = 3_000
        self.human_clock = False
        self.turn_started_at: float | None = None
        self.thinking = False
        self.paused = False
        self.moves: list[PlayedMove] = []
        self.result: str | None = None
        self.termination: str | None = None
        self.message = "Pick who plays which colour, then start the game."
        self.review = Review()
        self.live_eval = False
        self.live_kind = "depth"
        self.live_value = 12
        self.evals: dict[int, dict] = {}  # ply -> {"cp", "mate"}; ply 0 is the start position
        self._pending = 0  # evaluations in flight; the bots wait for these to reach zero
        self._engine_lock = threading.Lock()
        self._live: chess.engine.SimpleEngine | None = None
        self.engine_path = engine_path
        self.sf_threads = sf_threads
        self.sf_hash = sf_hash
        self._engine: chess.engine.SimpleEngine | None = None
        atexit.register(self.shutdown)

    # -- lifecycle ---------------------------------------------------------

    def shutdown(self) -> None:
        with self.lock:
            self._stop_agents()
            self._close_engine()
            self._close_live()

    def _stop_agents(self) -> None:
        for seat in self.seats.values():
            if seat.agent is not None:
                try:
                    seat.agent.stop()
                except Exception:
                    pass
                seat.agent = None

    def _close_engine(self) -> None:
        # SimpleEngine keeps a non-daemon thread; without this the interpreter hangs at exit.
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def _close_live(self) -> None:
        engine, self._live = self._live, None
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass

    def _live_engine(self) -> chess.engine.SimpleEngine | None:
        """One long-lived Stockfish for the bar, reused across moves."""
        if self._live is not None:
            return self._live
        binary = self.find_engine()
        if binary is None:
            return None
        self._live = chess.engine.SimpleEngine.popen_uci(binary)
        try:
            self._live.configure({"Threads": self.sf_threads, "Hash": self.sf_hash})
        except Exception:
            pass
        return self._live

    def _touch(self) -> None:
        self.version += 1

    # -- setup -------------------------------------------------------------

    def new_game(self, config: dict) -> None:
        with self.lock:
            self.generation += 1
            generation = self.generation
            self._stop_agents()
            self.review = Review()

            start = self._resolve_start(config)
            self.start_fen = start
            self.board = chess.Board(start)
            self.moves = []
            self.result = None
            self.termination = None
            self.paused = bool(config.get("paused", False))
            self.thinking = False

            self.base_ms = max(0, int(config.get("base_ms", BASE_MS)))
            self.increment_ms = max(0, int(config.get("increment_ms", INCREMENT_MS)))
            self.bot_clock_mode = "fixed" if config.get("bot_clock") == "fixed" else "match"
            self.bot_fixed_ms = max(1, int(config.get("bot_fixed_ms", 3_000)))
            self.human_clock = bool(config.get("human_clock", False))
            self.live_eval = bool(config.get("live_eval", False))
            self.live_kind = str(config.get("live_kind", "depth"))
            self.live_value = max(1, int(config.get("live_value", 12)))
            self.evals = {}

            for colour, key in ((chess.WHITE, "white"), (chess.BLACK, "black")):
                spec = config.get(key) or {}
                kind = "bot" if spec.get("kind") == "bot" else "human"
                path = str(spec.get("path") or "")
                label = "You"
                if kind == "bot":
                    label = next(
                        (a["label"] for a in discover_agents() if a["path"] == path), path or "bot"
                    )
                self.seats[colour] = Seat(kind=kind, path=path, label=label)
                self.clock[colour] = float(self.base_ms)
                self.clock_enabled[colour] = (
                    self.human_clock if kind == "human" else self.bot_clock_mode == "match"
                )

            self.phase = "starting"
            self.message = "Starting the bots."
            self.turn_started_at = None
            self._touch()

        threading.Thread(target=self._boot, args=(generation,), daemon=True).start()

    def _resolve_start(self, config: dict) -> str:
        mode = config.get("start", "standard")
        if mode == "fen":
            candidate = (config.get("fen") or "").strip()
            try:
                chess.Board(candidate)
                return candidate
            except ValueError:
                self.message = "That FEN did not parse. Started from the initial position instead."
                return chess.STARTING_FEN
        if mode == "opening":
            pool = opening_positions()
            if pool:
                return random.choice(pool)
        return chess.STARTING_FEN

    def _boot(self, generation: int) -> None:
        """Start every bot process. Import cost is real, so it happens off the request thread."""
        started: list[tuple[chess.Color, GuiAgent | None, str | None]] = []
        for colour in (chess.WHITE, chess.BLACK):
            with self.lock:
                if generation != self.generation:
                    return
                seat = self.seats[colour]
                kind, path = seat.kind, seat.path
            if kind != "bot":
                started.append((colour, None, None))
                continue
            directory = (REPO / path).resolve()
            if not (directory / "agent.py").exists():
                started.append((colour, None, f"no agent.py under {path}"))
                continue
            agent = GuiAgent([sys.executable, str(REPO / "harness" / "runner.py"), str(directory)])
            began = time.monotonic()
            try:
                agent.start(INIT_BUDGET_S)
                elapsed = time.monotonic() - began
                started.append((colour, agent, None))
                if elapsed > INIT_BUDGET_S * 0.5:
                    with self.lock:
                        self.message = (
                            f"{path} took {elapsed:.1f}s to start. "
                            f"The event allows {INIT_BUDGET_S:.0f}s."
                        )
            except AgentFailure as failure:
                started.append((colour, None, f"{path} failed to start: {failure.reason}"))

        with self.lock:
            if generation != self.generation:
                for _, agent, _ in started:
                    if agent is not None:
                        agent.stop()
                return
            broken = [error for _, _, error in started if error]
            for colour, agent, error in started:
                self.seats[colour].agent = agent
                self.seats[colour].failure = error
            if broken:
                self.phase = "over"
                self.message = broken[0]
                self._touch()
                return
            self.phase = "playing"
            self.message = ""
            self._queue_eval()
            self._touch()
        self._maybe_think()

    # -- clocks ------------------------------------------------------------

    def _elapsed_ms(self) -> float:
        if self.turn_started_at is None:
            return 0.0
        return (time.monotonic() - self.turn_started_at) * 1000.0

    def _display_clock(self, colour: chess.Color) -> float:
        value = self.clock[colour]
        if (
            self.phase == "playing"
            and self.board.turn == colour
            and self.clock_enabled[colour]
            and not self.paused
        ):
            value -= self._elapsed_ms()
        return value

    def _budget_for(self, colour: chess.Color) -> int:
        if self.bot_clock_mode == "fixed":
            return self.bot_fixed_ms
        return max(0, int(self.clock[colour]))

    # -- play --------------------------------------------------------------

    def human_move(self, uci: str) -> str | None:
        with self.lock:
            if self.phase != "playing":
                return "The game is not running."
            if self.thinking:
                return "The bot is still thinking."
            colour = self.board.turn
            if self.seats[colour].kind != "human":
                return "It is not your move."
            try:
                move = chess.Move.from_uci(uci)
            except chess.InvalidMoveError:
                return "That is not a move."
            if move not in self.board.legal_moves:
                return "That move is not legal here."
            spent = self._elapsed_ms() if self.clock_enabled[colour] else 0.0
            self.paused = False  # you moved, so you want the game to go on
            flagged = self._apply(move, colour, spent)
        if not flagged:
            self._maybe_think()
        return None

    def _apply(self, move: chess.Move, colour: chess.Color, spent_ms: float) -> bool:
        """Push a move under referee clock rules. Returns True if the mover flagged."""
        san = self.board.san(move)
        number = self.board.fullmove_number
        note = None
        if self.clock_enabled[colour]:
            self.clock[colour] -= spent_ms
            if self.clock[colour] < 0:
                self.moves.append(
                    PlayedMove(
                        ply=len(self.board.move_stack) + 1,
                        move_no=number,
                        san=san,
                        uci=move.uci(),
                        side="white" if colour == chess.WHITE else "black",
                        ms=spent_ms,
                        clock_after=self.clock[colour],
                        fen=self.board.fen(),
                        note="flag",
                    )
                )
                self._finish(
                    "black" if colour == chess.WHITE else "white",
                    "flag",
                    f"{self._name(colour)} ran out of time.",
                )
                return True
        self.board.push(move)
        if self.clock_enabled[colour]:
            self.clock[colour] += self.increment_ms
        self.moves.append(
            PlayedMove(
                ply=len(self.board.move_stack),
                move_no=number,
                san=san,
                uci=move.uci(),
                side="white" if colour == chess.WHITE else "black",
                ms=spent_ms,
                clock_after=self.clock[colour] if self.clock_enabled[colour] else None,
                fen=self.board.fen(),
                note=note,
            )
        )
        self._touch()
        self._check_terminal()
        self._queue_eval()
        return False

    def _queue_eval(self) -> None:
        """Evaluate the position now, before either side is allowed to think.

        Stockfish and the bot never run at the same time. If they did, the bar would
        take a core off the bot and, in match mode, spend real milliseconds off its
        clock -- which would make this rig lie about the one thing it exists to measure.
        Serialising them costs wall-clock time and nothing else, because the referee's
        clock only starts when the agent is asked for a move.
        """
        if not self.live_eval:
            self.turn_started_at = time.monotonic()
            return
        self._pending += 1
        threading.Thread(
            target=self._evaluate,
            args=(self.generation, len(self.board.move_stack), self.board.fen()),
            daemon=True,
        ).start()

    def _evaluate(self, generation: int, ply: int, fen: str) -> None:
        try:
            with self._engine_lock:  # one search at a time; SimpleEngine is not reentrant
                engine = self._live_engine()
                if engine is None:
                    with self.lock:
                        self.live_eval = False
                        self.message = "No Stockfish binary found, so the bar is off."
                else:
                    scored = self._score(
                        engine, make_limit(self.live_kind, self.live_value), chess.Board(fen)
                    )
                    with self.lock:
                        if generation == self.generation:
                            self.evals[ply] = {"cp": scored["cp"], "mate": scored["mate"]}
        except Exception as error:  # noqa: BLE001 -- a dead engine must not stop the game
            self._close_live()
            with self.lock:
                self.live_eval = False
                self.message = f"The evaluation bar stopped: {error}"
        finally:
            resume = False
            with self.lock:
                self._pending = max(0, self._pending - 1)
                if self._pending == 0:
                    self.turn_started_at = time.monotonic()
                    resume = True
                self._touch()
            if resume:
                self._maybe_think()

    def _check_terminal(self) -> None:
        """The referee's own test, claim_draw included. That flag loses won games."""
        finish = self.board.outcome(claim_draw=True)
        if finish is not None:
            decision = (
                "draw"
                if finish.winner is None
                else ("white" if finish.winner == chess.WHITE else "black")
            )
            termination = finish.termination.name.lower()
            note = ""
            if termination in ("threefold_repetition", "fifty_moves"):
                note = " The referee claims this automatically -- neither side has to ask."
            self._finish(decision, termination, f"{termination.replace('_', ' ')}.{note}")
            return
        if len(self.board.move_stack) >= PLY_CAP:
            balance = sum(
                value
                * (
                    len(self.board.pieces(piece, chess.WHITE))
                    - len(self.board.pieces(piece, chess.BLACK))
                )
                for piece, value in PIECE_VALUES.items()
            )
            decision = "white" if balance > 0 else "black" if balance < 0 else "draw"
            self._finish(
                decision,
                "adjudication",
                f"{PLY_CAP} plies reached. Adjudicated on material: {balance:+d}.",
            )

    def _finish(self, result: str, termination: str, message: str) -> None:
        self.phase = "over"
        self.result = result
        self.termination = termination
        self.message = message
        self.thinking = False
        self._touch()

    def _name(self, colour: chess.Color) -> str:
        seat = self.seats[colour]
        return seat.label if seat.kind == "bot" else "You"

    def _maybe_think(self) -> None:
        with self.lock:
            if self.phase != "playing" or self.thinking or self.paused or self._pending:
                return
            colour = self.board.turn
            seat = self.seats[colour]
            if seat.kind != "bot" or seat.agent is None:
                return
            self.thinking = True
            self.turn_started_at = time.monotonic()
            generation = self.generation
            self._touch()
        threading.Thread(target=self._think, args=(generation, colour), daemon=True).start()

    def _think(self, generation: int, colour: chess.Color) -> None:
        with self.lock:
            agent = self.seats[colour].agent
            fen = self.board.fen()
            budget = self._budget_for(colour)
            watchdog = min(
                HARD_THINK_CAP_S,
                (budget + WATCHDOG_GRACE_MS) / 1000.0
                if self.bot_clock_mode == "match"
                else HARD_THINK_CAP_S,
            )
        if agent is None:
            return

        began = time.monotonic()
        failure: str | None = None
        uci = ""
        try:
            uci = agent.ask(fen, budget, watchdog)
        except AgentFailure as error:
            failure = error.reason
        except Exception as error:  # noqa: BLE001 -- a broken bot must not take the server with it
            failure = f"crash ({error})"
        spent = (time.monotonic() - began) * 1000.0

        with self.lock:
            if generation != self.generation or self.phase != "playing":
                self.thinking = False
                return
            self.thinking = False
            if failure is not None:
                self._finish(
                    "black" if colour == chess.WHITE else "white",
                    failure,
                    f"{self._name(colour)} failed: {failure}. "
                    f"On the ladder this is a lost game.",
                )
                return
            try:
                move = chess.Move.from_uci(uci)
            except chess.InvalidMoveError:
                move = chess.Move.null()
            if move not in self.board.legal_moves:
                self._finish(
                    "black" if colour == chess.WHITE else "white",
                    "illegal",
                    f"{self._name(colour)} returned {uci!r}, which is not legal here.",
                )
                return
            flagged = self._apply(move, colour, spent)
        if not flagged:
            self._maybe_think()

    def nudge(self) -> None:
        """Un-pause / kick the bot when it is its turn."""
        with self.lock:
            self.paused = False
            if self.phase == "playing" and not self.thinking and not self._pending:
                self.turn_started_at = time.monotonic()
            self._touch()
        self._maybe_think()

    def pause(self) -> None:
        with self.lock:
            self.paused = True
            self._touch()

    def takeback(self) -> None:
        with self.lock:
            if self.thinking or not self.board.move_stack:
                return
            plies = 1
            if any(seat.kind == "human" for seat in self.seats.values()):
                colour = self.board.turn
                if self.seats[colour].kind == "bot":
                    plies = 1
                else:
                    plies = 2 if len(self.board.move_stack) >= 2 else 1
            for _ in range(plies):
                if not self.board.move_stack:
                    break
                self.board.pop()
                if self.moves:
                    undone = self.moves.pop()
                    side = chess.WHITE if undone.side == "white" else chess.BLACK
                    if self.clock_enabled[side]:
                        self.clock[side] += undone.ms - self.increment_ms
            self.phase = "playing" if self.phase == "over" else self.phase
            self.result = None
            self.termination = None
            self.message = ""
            # if the rewind lands on a bot, hold it -- otherwise it instantly replays
            # into the position you just took back
            self.paused = self.seats[self.board.turn].kind == "bot"
            for ply in [p for p in self.evals if p > len(self.board.move_stack)]:
                del self.evals[ply]
            self._queue_eval()
            self._touch()

    def resign(self) -> None:
        with self.lock:
            if self.phase != "playing":
                return
            human = next(
                (c for c, s in self.seats.items() if s.kind == "human"), self.board.turn
            )
            self._finish(
                "black" if human == chess.WHITE else "white", "resignation", "You resigned."
            )

    # -- pgn ---------------------------------------------------------------

    def pgn(self, with_evals: bool = True) -> str:
        with self.lock:
            game = chess.pgn.Game()
            game.headers["Event"] = "AI Chessathon sparring"
            game.headers["White"] = self._name(chess.WHITE)
            game.headers["Black"] = self._name(chess.BLACK)
            game.headers["Date"] = time.strftime("%Y.%m.%d")
            if self.start_fen != chess.STARTING_FEN:
                game.headers["FEN"] = self.start_fen
                game.headers["SetUp"] = "1"
            game.headers["Result"] = {
                "white": "1-0",
                "black": "0-1",
                "draw": "1/2-1/2",
                None: "*",
            }.get(self.result, "*")
            if self.termination:
                game.headers["Termination"] = self.termination
            game.headers["TimeControl"] = f"{self.base_ms // 1000}+{self.increment_ms / 1000:g}"

            evals = {row["ply"]: row for row in self.review.rows} if with_evals else {}
            node: chess.pgn.GameNode = game
            board = chess.Board(self.start_fen)
            for index, move in enumerate(self.board.move_stack, start=1):
                node = node.add_variation(move)
                played = self.moves[index - 1] if index - 1 < len(self.moves) else None
                comments = []
                row = evals.get(index)
                if row is not None:
                    if row.get("mate_after") is not None:
                        comments.append(f"[%eval #{row['mate_after']}]")
                    else:
                        comments.append(f"[%eval {row['eval_after'] / 100:.2f}]")
                if played is not None and played.ms:
                    comments.append(f"[%clk {played.ms / 1000:.2f}s]")
                if comments:
                    node.comment = " ".join(comments)
                board.push(move)
            return str(game)

    def load_pgn(self, text: str) -> str | None:
        try:
            game = chess.pgn.read_game(io.StringIO(text))
        except Exception:
            game = None
        if game is None:
            return "That did not parse as PGN."
        board = game.board()
        replayed: list[PlayedMove] = []
        for move in game.mainline_moves():
            san = board.san(move)
            number, side = board.fullmove_number, board.turn
            board.push(move)
            replayed.append(
                PlayedMove(
                    ply=len(board.move_stack),
                    move_no=number,
                    san=san,
                    uci=move.uci(),
                    side="white" if side == chess.WHITE else "black",
                    ms=0.0,
                    clock_after=None,
                    fen=board.fen(),
                )
            )
        if not replayed:
            return "That PGN has no moves."
        with self.lock:
            self.generation += 1
            self._stop_agents()
            self.review = Review()
            self.evals = {}
            self.start_fen = game.board().fen()
            self.board = board
            self.moves = replayed
            self.phase = "review"
            self.thinking = False
            self.paused = True
            self.result = None
            self.termination = game.headers.get("Termination") or None
            self.seats[chess.WHITE] = Seat(kind="human", label=game.headers.get("White", "White"))
            self.seats[chess.BLACK] = Seat(kind="human", label=game.headers.get("Black", "Black"))
            self.clock_enabled = {chess.WHITE: False, chess.BLACK: False}
            self.message = f"Loaded {len(replayed)} plies. Review it with Stockfish."
            self._touch()
        return None

    # -- review ------------------------------------------------------------

    def find_engine(self) -> str | None:
        for candidate in (self.engine_path, os.environ.get("SPAR_ENGINE"), "stockfish"):
            if not candidate:
                continue
            # a path that exists but cannot be executed is not an engine; accepting it
            # makes the bar report itself available and then fail on first use
            resolved = shutil.which(candidate)
            if resolved is None and Path(candidate).is_file() and os.access(candidate, os.X_OK):
                resolved = candidate
            if resolved:
                return resolved
        return None

    def start_review(self, limit_kind: str, limit_value: int) -> str | None:
        with self.lock:
            if self.review.running:
                return "A review is already running."
            if not self.board.move_stack:
                return "There is nothing to review yet."
            binary = self.find_engine()
            if binary is None:
                return (
                    "No Stockfish binary found. Pass --engine /path/to/stockfish, "
                    "set $SPAR_ENGINE, or put it on PATH."
                )
            total = len(self.board.move_stack) + 1
            self.review = Review(
                running=True,
                done=0,
                total=total,
                engine=Path(binary).name,
                limit=f"{limit_kind} {limit_value}",
            )
            generation = self.generation
            self._touch()
        threading.Thread(
            target=self._review,
            args=(generation, binary, limit_kind, limit_value),
            daemon=True,
        ).start()
        return None

    def _review(self, generation: int, binary: str, limit_kind: str, limit_value: int) -> None:
        limit = make_limit(limit_kind, limit_value)

        with self.lock:
            start_fen = self.start_fen
            stack = list(self.board.move_stack)
            played = list(self.moves)

        engine = None
        try:
            engine = chess.engine.SimpleEngine.popen_uci(binary)
            self._engine = engine
            try:
                engine.configure({"Threads": self.sf_threads, "Hash": self.sf_hash})
            except Exception:
                pass  # some builds reject one or both; the review still works

            board = chess.Board(start_fen)
            scored: list[dict] = []
            for index in range(len(stack) + 1):
                if generation != self.generation:
                    return
                scored.append(self._score(engine, limit, board))
                with self.lock:
                    self.review.done = index + 1
                    self.review.rows = self._rows(scored, stack, played)
                    self._touch()
                if index < len(stack):
                    board.push(stack[index])

            with self.lock:
                self.review.rows = self._rows(scored, stack, played)
                self.review.acpl = self._acpl(self.review.rows)
                self.review.running = False
                # the bar reads from one place, so a finished review replaces whatever
                # the shallower live pass recorded
                self.evals = {index: {"cp": item["cp"], "mate": item["mate"]}
                              for index, item in enumerate(scored)}
                self._touch()
        except Exception as error:  # noqa: BLE001
            with self.lock:
                self.review.running = False
                self.review.error = f"{type(error).__name__}: {error}"
                self._touch()
        finally:
            if engine is not None:
                try:
                    engine.quit()
                except Exception:
                    pass
            self._engine = None

    @staticmethod
    def _score(engine: chess.engine.SimpleEngine, limit, board: chess.Board) -> dict:
        """One position's verdict, from White's point of view.

        A finished position is answered here rather than by the engine. Asking an
        engine to search a checkmate is undefined -- some builds reply with no
        score line at all, which is a crash rather than an eval.
        """
        common = {
            "fen": board.fen(),
            "side": "white" if board.turn == chess.WHITE else "black",
            "move_no": board.fullmove_number,
        }
        if board.is_checkmate():
            mated_is_white = board.turn == chess.WHITE
            return {
                **common,
                "cp": -MATE_SCORE if mated_is_white else MATE_SCORE,
                "mate": 0,
                "best_uci": None,
                "best_san": None,
            }
        if not any(board.legal_moves) or board.is_insufficient_material():
            return {**common, "cp": 0, "mate": None, "best_uci": None, "best_san": None}

        info = engine.analyse(board, limit)
        score = info.get("score")
        pov = score.white() if score is not None else None
        best = info.get("pv") or []
        return {
            **common,
            "cp": pov.score(mate_score=MATE_SCORE) if pov is not None else 0,
            "mate": pov.mate() if pov is not None else None,
            "best_uci": best[0].uci() if best else None,
            "best_san": board.san(best[0]) if best else None,
        }

    @staticmethod
    def _rows(scored: list[dict], stack: list[chess.Move], played: list[PlayedMove]) -> list[dict]:
        """One row per played move. Loss comes from the next position's eval, not a second search."""
        rows: list[dict] = []
        for index in range(min(len(scored) - 1, len(stack))):
            before, after = scored[index], scored[index + 1]
            # from the analysed board, so a game starting from a black-to-move
            # curated opening numbers its moves correctly
            board_side = before["side"]
            mover_is_white = board_side == "white"

            def clamp(value: int | None) -> int:
                return max(-EVAL_CLAMP, min(EVAL_CLAMP, int(value or 0)))

            delta = clamp(before["cp"]) - clamp(after["cp"])
            loss = max(0, delta if mover_is_white else -delta)
            uci = stack[index].uci()
            if before["best_uci"] == uci:
                tag = "best"
            else:
                tag = BLUNDER_TAG
                for ceiling, name in TAG_THRESHOLDS:
                    if loss < ceiling:
                        tag = name
                        break
            rows.append(
                {
                    "ply": index + 1,
                    "move_no": before["move_no"],
                    "side": board_side,
                    "san": played[index].san if index < len(played) else uci,
                    "uci": uci,
                    "eval_before": before["cp"],
                    "eval_after": after["cp"],
                    "mate_before": before["mate"],
                    "mate_after": after["mate"],
                    "loss": loss,
                    "tag": tag,
                    "best_san": before["best_san"],
                    "best_uci": before["best_uci"],
                    "fen_before": before["fen"],
                    "ms": played[index].ms if index < len(played) else 0.0,
                }
            )
        return rows

    @staticmethod
    def _acpl(rows: list[dict]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for side in ("white", "black"):
            losses = [row["loss"] for row in rows if row["side"] == side]
            out[side] = round(sum(losses) / len(losses), 1) if losses else None
        return out

    # -- state -------------------------------------------------------------

    def snapshot(self) -> dict:
        with self.lock:
            legal: dict[str, list[str]] = {}
            promotions: list[str] = []
            if self.phase == "playing":
                for move in self.board.legal_moves:
                    source = chess.square_name(move.from_square)
                    target = chess.square_name(move.to_square)
                    legal.setdefault(source, [])
                    if target not in legal[source]:
                        legal[source].append(target)
                    if move.promotion:
                        promotions.append(source + target)

            last = self.board.move_stack[-1] if self.board.move_stack else None
            check_square = None
            if self.board.is_check():
                king = self.board.king(self.board.turn)
                check_square = chess.square_name(king) if king is not None else None

            return {
                "version": self.version,
                "phase": self.phase,
                "fen": self.board.fen(),
                "start_fen": self.start_fen,
                "turn": "white" if self.board.turn == chess.WHITE else "black",
                "legal": legal,
                "promotions": sorted(set(promotions)),
                "last_move": [
                    chess.square_name(last.from_square),
                    chess.square_name(last.to_square),
                ]
                if last
                else None,
                "check_square": check_square,
                "thinking": self.thinking,
                "paused": self.paused,
                "result": self.result,
                "termination": self.termination,
                "message": self.message,
                "ply_cap": PLY_CAP,
                "plies": len(self.board.move_stack),
                "increment_ms": self.increment_ms,
                "bot_clock_mode": self.bot_clock_mode,
                "bot_fixed_ms": self.bot_fixed_ms,
                "seats": {
                    "white": self._seat_state(chess.WHITE),
                    "black": self._seat_state(chess.BLACK),
                },
                "moves": [
                    {
                        "ply": m.ply,
                        "move_no": m.move_no,
                        "san": m.san,
                        "uci": m.uci,
                        "side": m.side,
                        "ms": round(m.ms),
                        "clock_after": None if m.clock_after is None else round(m.clock_after),
                        "fen": m.fen,
                        "note": m.note,
                    }
                    for m in self.moves
                ],
                "evals": self.evals,
                "live_eval": self.live_eval,
                "live_limit": f"{self.live_kind} {self.live_value}",
                "evaluating": self._pending > 0,
                "review": {
                    "running": self.review.running,
                    "done": self.review.done,
                    "total": self.review.total,
                    "error": self.review.error,
                    "engine": self.review.engine,
                    "limit": self.review.limit,
                    "rows": self.review.rows,
                    "acpl": self.review.acpl,
                },
                "engine_available": self.find_engine() is not None,
            }

    def _seat_state(self, colour: chess.Color) -> dict:
        seat = self.seats[colour]
        return {
            "kind": seat.kind,
            "path": seat.path,
            "label": seat.label,
            "clock_ms": round(self._display_clock(colour)),
            "clock_enabled": self.clock_enabled[colour],
            "clock_running": (
                self.phase == "playing"
                and self.board.turn == colour
                and self.clock_enabled[colour]
                and not self.paused
            ),
            "log": seat.agent.log() if seat.agent is not None else "",
            "failure": seat.failure,
        }


# --------------------------------------------------------------------------- http


POST_ROUTES = frozenset(
    {
        "/api/new",
        "/api/move",
        "/api/go",
        "/api/pause",
        "/api/takeback",
        "/api/resign",
        "/api/loadpgn",
        "/api/review",
    }
)


class Handler(BaseHTTPRequestHandler):
    session: Session
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:  # the console is for the bots, not the web server
        return

    def _send(self, payload: dict | str, status: int = 200, kind: str = "application/json") -> None:
        body = (payload if isinstance(payload, str) else json.dumps(payload)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{kind}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route in POST_ROUTES:
            self._send({"error": f"{route} needs POST, not GET"}, status=405)
        elif route == "/":
            self._send(PAGE.replace("__PIECES__", piece_sprites()), kind="text/html")
        elif route == "/api/state":
            self._send(self.session.snapshot())
        elif route == "/api/agents":
            self._send({"agents": discover_agents(), "openings": len(opening_positions())})
        elif route == "/api/pgn":
            self._send(self.session.pgn(), kind="text/plain")
        else:
            self._send({"error": "no such route"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        body = self._body()
        if route == "/api/new":
            self.session.new_game(body)
        elif route == "/api/move":
            error = self.session.human_move(str(body.get("uci", "")))
            if error:
                self._send({"error": error}, status=400)
                return
        elif route == "/api/go":
            self.session.nudge()
        elif route == "/api/pause":
            self.session.pause()
        elif route == "/api/takeback":
            self.session.takeback()
        elif route == "/api/resign":
            self.session.resign()
        elif route == "/api/loadpgn":
            error = self.session.load_pgn(str(body.get("pgn", "")))
            if error:
                self._send({"error": error}, status=400)
                return
        elif route == "/api/review":
            error = self.session.start_review(
                str(body.get("kind", "depth")), int(body.get("value", 16))
            )
            if error:
                self._send({"error": error}, status=400)
                return
        else:
            self._send({"error": "no such route"}, status=404)
            return
        self._send(self.session.snapshot())


def piece_sprites() -> str:
    """Reuse python-chess's bundled Cburnett set rather than shipping images."""
    parts = []
    for symbol, svg in chess.svg.PIECES.items():
        colour = "w" if symbol.isupper() else "b"
        parts.append(f'<symbol id="pc-{colour}{symbol.lower()}" viewBox="0 0 45 45">{svg}</symbol>')
    return "".join(parts)


# --------------------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sparring board</title>
<style>
:root{
  --ink:#141d25; --panel:#1a252f; --raise:#213039; --line:#2b3d49;
  --text:#d5e1e9; --dim:#7b91a1; --faint:#54697a;
  --live:#5ec5b6; --warm:#d9a441; --bad:#d9644a; --good:#8fbf7a;
  --light:#adbcc7; --dark:#5d7385; --mark:#c9b458; --pick:#5ec5b6;
  --ui:ui-sans-serif,Inter,"Segoe UI",system-ui,sans-serif;
  --data:ui-monospace,"JetBrains Mono","SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{
  background:var(--ink);color:var(--text);font-family:var(--ui);
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased;
  display:grid;grid-template-columns:270px minmax(0,1fr) 380px;gap:1px;
  background-color:var(--line);overflow:hidden;
}
.col{background:var(--ink);overflow-y:auto;scrollbar-gutter:stable;padding:18px;min-width:0}
.col::-webkit-scrollbar{width:9px}
.col::-webkit-scrollbar-thumb{background:var(--line);border-radius:5px}
/* The board is sized from the viewport height minus the fixed chrome around it, not
   from whatever space is left over. It also never shrinks: as a flex item it would
   otherwise be compressed vertically the moment the status line grew a second row,
   which changed the board's size at the exact moment a game started. */
#centre{--bw:max(280px,min(calc(100vh - 210px),100%));background:var(--ink);display:flex;
  flex-direction:column;align-items:center;gap:10px;padding:18px 24px;
  overflow-y:auto;scrollbar-gutter:stable;min-width:0}
#centre > *{flex:none}
@media (max-width:1180px){
  body{grid-template-columns:250px minmax(0,1fr);grid-template-rows:1fr auto;height:auto;overflow:auto}
  #right{grid-column:1/-1;max-height:60vh}
}
@media (max-width:760px){
  body{grid-template-columns:1fr}
  #centre{--bw:100%;order:-1}
  #right{grid-column:auto}
}

h1{font-size:15px;font-weight:600;margin:0 0 2px;letter-spacing:-.01em}
h2{font-size:12px;font-weight:600;color:var(--dim);margin:20px 0 8px;letter-spacing:0}
h2:first-of-type{margin-top:16px}
p.note{color:var(--faint);font-size:12px;margin:4px 0 0}
a{color:var(--live)}

label.row{display:flex;align-items:center;gap:8px;padding:5px 0;cursor:pointer;font-size:13px}
label.row input{accent-color:var(--live);margin:0}
select,input[type=text],input[type=number],textarea{
  width:100%;background:var(--panel);color:var(--text);border:1px solid var(--line);
  border-radius:3px;padding:6px 8px;font:inherit;font-size:13px
}
input[type=number]{font-family:var(--data)}
select:focus,input:focus,textarea:focus,button:focus-visible{outline:2px solid var(--live);outline-offset:1px}
textarea{font-family:var(--data);font-size:11px;resize:vertical;min-height:70px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.field{margin:8px 0}
.field > span{display:block;color:var(--dim);font-size:12px;margin-bottom:4px}

button{
  background:var(--raise);color:var(--text);border:1px solid var(--line);border-radius:3px;
  padding:7px 12px;font:inherit;font-size:13px;cursor:pointer
}
button:hover:not(:disabled){border-color:var(--dim)}
button:disabled{opacity:.4;cursor:default}
button.primary{background:var(--live);border-color:var(--live);color:#0d1a1a;font-weight:600}
button.primary:hover{filter:brightness(1.08)}
.actions{display:flex;flex-wrap:wrap;gap:6px}
.actions button{flex:1 1 auto}

/* board */
#boardrow{display:flex;gap:8px;width:var(--bw)}
#boardwrap{position:relative;flex:none;align-self:start;
  width:calc(100% - 32px);aspect-ratio:1}   /* 24px bar + 8px gap */
#evalbar[hidden] + #boardwrap{width:100%}   /* no bar, no gap, so take it all back */
#evalbar{position:relative;flex:none;align-self:stretch;width:24px;background:#10161b;
  border:1px solid var(--line);border-radius:2px;overflow:hidden}
#evalbar[hidden]{display:none}
#evalbar i{position:absolute;left:0;right:0;bottom:0;height:50%;background:#e7edf1;
  transition:height .25s ease}
#evalbar.flip i{bottom:auto;top:0}
#evalbar .v{position:absolute;left:0;right:0;bottom:3px;text-align:center;
  font:600 9px/1.4 var(--data);color:#10161b}
#evalbar.flip .v{bottom:auto;top:3px}
#evalbar.on-dark .v{color:var(--dim)}
#evalbar.pending{opacity:.55}
@media (prefers-reduced-motion:reduce){#evalbar i{transition:none}}

#transport{display:flex;align-items:center;gap:6px;width:var(--bw)}
#transport button{padding:5px 10px;font-size:12px}
#transport #tPos{flex:1;text-align:center;font-family:var(--data);
  font-variant-numeric:tabular-nums;font-size:12px;color:var(--dim)}
#transport.browsing #tPos{color:var(--warm)}
#board{position:absolute;inset:0;display:grid;
  grid-template-columns:repeat(8,1fr);grid-template-rows:repeat(8,1fr);
  border:1px solid var(--line);user-select:none}
.sq{position:relative;display:flex;align-items:center;justify-content:center}
.sq.light{background:var(--light)} .sq.dark{background:var(--dark)}
.sq.last::after{content:"";position:absolute;inset:0;background:var(--mark);opacity:.32}
.sq.sel::after{content:"";position:absolute;inset:0;background:var(--pick);opacity:.4}
.sq.check{background:radial-gradient(circle,var(--bad) 0%,var(--bad) 32%,transparent 72%)}
.sq svg.pc{position:relative;width:100%;height:100%;pointer-events:none;z-index:2;
  filter:drop-shadow(0 1px 1px rgba(0,0,0,.35))}
.sq .dot{position:absolute;width:28%;height:28%;border-radius:50%;background:#0c1418;opacity:.34;z-index:1}
.sq.occupied .dot{width:100%;height:100%;border-radius:0;background:none;opacity:1;
  box-shadow:inset 0 0 0 5px rgba(12,20,24,.34)}
.sq .coord{position:absolute;font:600 9px/1 var(--data);color:#0c1418;opacity:.55;z-index:3}
.sq.dark .coord{color:#e3ebf0;opacity:.6}
.sq .coord.f{bottom:2px;right:3px} .sq .coord.r{top:2px;left:3px}
#arrows{position:absolute;inset:0;pointer-events:none;z-index:4}

/* clocks */
.clockline{display:flex;align-items:baseline;gap:10px;width:var(--bw);min-height:26px}
.clockline .who{font-size:13px;color:var(--dim);flex:1;min-width:0;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.clockline .who b{color:var(--text);font-weight:600}
.clockline .t{font-family:var(--data);font-size:20px;font-variant-numeric:tabular-nums;
  letter-spacing:-.02em;color:var(--faint)}
.clockline.on .t{color:var(--text)}
.clockline.low .t{color:var(--bad)}
.clockline .beat{width:7px;height:7px;border-radius:50%;background:var(--live);opacity:0}
.clockline.on .beat{opacity:1;animation:beat 1.1s ease-in-out infinite}
@keyframes beat{0%,100%{opacity:.25}50%{opacity:1}}
@media (prefers-reduced-motion:reduce){.clockline .beat{animation:none;opacity:1}}

#status{width:var(--bw);min-height:44px;font-size:13px;color:var(--dim)}
#status b{color:var(--text);font-weight:600}
#status.alert b{color:var(--warm)}

/* tape */
.tabs{display:flex;gap:2px;border-bottom:1px solid var(--line);margin-bottom:12px}
.tabs button{background:none;border:none;border-bottom:2px solid transparent;border-radius:0;
  padding:6px 10px;color:var(--dim)}
.tabs button.on{color:var(--text);border-bottom-color:var(--live)}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:3px 6px;text-align:left}
th{color:var(--faint);font-weight:500;font-size:11px;border-bottom:1px solid var(--line)}
tr.mv{cursor:pointer}
tr.mv:hover{background:var(--panel)}
tr.mv.on{background:var(--raise)}
.n{font-family:var(--data);font-variant-numeric:tabular-nums;text-align:right;color:var(--dim)}
.san{font-weight:600}
.tag{font-family:var(--data);font-size:11px}
.tag.blunder{color:var(--bad)} .tag.mistake{color:var(--warm)}
.tag.inaccuracy{color:var(--dim)} .tag.best{color:var(--good)} .tag.ok{color:var(--faint)}
#log{font-family:var(--data);font-size:11px;line-height:1.45;white-space:pre-wrap;
  word-break:break-word;color:var(--dim);background:var(--panel);border:1px solid var(--line);
  border-radius:3px;padding:10px;max-height:calc(100vh - 190px);overflow:auto}
#spark{width:100%;height:64px;display:block;background:var(--panel);
  border:1px solid var(--line);border-radius:3px}
.bar{height:3px;background:var(--line);border-radius:2px;overflow:hidden;margin:8px 0}
.bar i{display:block;height:100%;background:var(--live);transition:width .2s}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--faint);margin-top:6px}

/* promotion */
#promo{position:absolute;inset:0;background:rgba(10,16,20,.82);display:none;
  align-items:center;justify-content:center;gap:6px;z-index:5}
#promo.on{display:flex}
#promo button{width:64px;height:64px;padding:6px;background:var(--panel)}
#promo svg{width:100%;height:100%}
.empty{color:var(--faint);font-size:13px;padding:12px 0}
</style>
</head>
<body>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>__PIECES__</defs></svg>

<div class="col" id="left">
  <h1>Sparring board</h1>
  <p class="note">Your bot runs as a real agent process, through the harness runner.</p>

  <h2>White</h2>
  <select id="whitePick"></select>
  <h2>Black</h2>
  <select id="blackPick"></select>

  <h2>Bot clock</h2>
  <label class="row"><input type="radio" name="botclock" value="match" checked>
    Match clock, flag enforced</label>
  <label class="row"><input type="radio" name="botclock" value="fixed">
    Fixed budget every move</label>
  <div class="pair">
    <div class="field"><span>Base seconds</span><input type="number" id="baseS" value="120" min="1" step="1"></div>
    <div class="field"><span>Increment ms</span><input type="number" id="incMs" value="500" min="0" step="50"></div>
  </div>
  <div class="field" id="fixedField" style="display:none">
    <span>Tell the bot it has (ms)</span><input type="number" id="fixedMs" value="3000" min="50" step="500">
  </div>
  <label class="row"><input type="checkbox" id="humanClock"> Run my clock too</label>

  <h2>Engine evaluation</h2>
  <label class="row"><input type="checkbox" id="liveEval"> Show the evaluation bar</label>
  <div class="field" id="liveField" style="display:none">
    <span>Depth per position</span>
    <input type="number" id="liveDepth" value="12" min="1" max="30">
  </div>
  <p class="note" id="liveNote">Stockfish evaluates each new position before either side
    is allowed to think, so it never takes CPU from the bot or spends its clock.</p>

  <h2>Starting position</h2>
  <select id="startPick">
    <option value="standard">Initial position</option>
    <option value="opening">Random curated opening</option>
    <option value="fen">From a FEN</option>
  </select>
  <div class="field" id="fenField" style="display:none">
    <input type="text" id="fen" placeholder="paste a FEN">
  </div>

  <div class="actions" style="margin-top:16px">
    <button class="primary" id="start">Start game</button>
  </div>

  <h2>Load a played game</h2>
  <textarea id="pgnIn" placeholder="paste PGN from a ladder game"></textarea>
  <div class="actions" style="margin-top:6px"><button id="loadPgn">Load PGN</button></div>
</div>

<div id="centre">
  <div class="clockline" id="topClock"><span class="beat"></span><span class="who"></span><span class="t">—</span></div>
  <div id="boardrow">
    <div id="evalbar" hidden><i></i><span class="v"></span></div>
    <div id="boardwrap">
      <div id="board"></div>
      <svg id="arrows" viewBox="0 0 8 8"></svg>
      <div id="promo"></div>
    </div>
  </div>
  <div class="clockline" id="botClock"><span class="beat"></span><span class="who"></span><span class="t">—</span></div>
  <div id="transport">
    <button id="tStart">Start</button>
    <button id="tBack">Back</button>
    <span id="tPos">—</span>
    <button id="tFwd">Forward</button>
    <button id="tNow">Present</button>
  </div>
  <div id="status"></div>
  <div class="actions" style="width:var(--bw)">
    <button id="flip">Flip board</button>
    <button id="back">Take back</button>
    <button id="go">Bot moves now</button>
    <button id="pause">Pause</button>
    <button id="resign">Resign</button>
    <button id="savePgn">Save PGN</button>
  </div>
</div>

<div class="col" id="right">
  <div class="tabs">
    <button data-tab="moves" class="on">Moves</button>
    <button data-tab="review">Review</button>
    <button data-tab="log">Bot output</button>
  </div>
  <div id="paneMoves"></div>
  <div id="paneReview" style="display:none">
    <p class="note" id="noEngine" style="display:none">
      No Stockfish found. Start this tool with <code>--engine /path/to/stockfish</code>,
      or set <code>SPAR_ENGINE</code>.</p>
    <div class="pair">
      <div class="field"><span>Search limit</span><select id="rKind">
        <option value="depth">Depth</option>
        <option value="nodes">Nodes</option>
        <option value="movetime">Milliseconds</option>
      </select></div>
      <div class="field"><span>Value</span><input type="number" id="rVal" value="16" min="1"></div>
    </div>
    <div class="actions"><button class="primary" id="runReview">Review with Stockfish</button></div>
    <div id="reviewOut"></div>
  </div>
  <div id="paneLog" style="display:none">
    <select id="logPick"><option value="white">White</option><option value="black">Black</option></select>
    <div id="log" style="margin-top:8px"></div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const FILES = "abcdefgh";
let S = null, sel = null, orient = "white", tab = "moves";
let lastFen = null, agents = [];
/* histPly: how many plies of the game the board is showing. null means the present.
   Anything else is read only -- the game cannot be played from the past. */
let histPly = null, pausedByBrowsing = false;

/* ---------- helpers ---------- */
/* Every /api/ route that changes something is POST-only. A helper that quietly
   downgraded a bodyless call to GET is how "Resign" reached do_GET and came back
   "no such route". Reads use getJSON. */
async function api(path, body){
  const res = await fetch(path, {
    method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body || {})
  });
  const data = await res.json().catch(()=>({}));
  if(!res.ok && data.error) flash(data.error);
  return data;
}
async function getJSON(path){
  const res = await fetch(path);
  return res.json().catch(()=>({}));
}
function flash(text){ const s = $("#status"); s.classList.add("alert"); s.innerHTML = "<b>"+esc(text)+"</b>"; }
function esc(t){ return String(t).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function clockText(ms){
  if(ms === null || ms === undefined) return "—";
  const neg = ms < 0; ms = Math.abs(ms);
  const total = ms/1000, m = Math.floor(total/60), s = total - m*60;
  const body = total < 60 ? s.toFixed(1) : m + ":" + String(Math.floor(s)).padStart(2,"0");
  return (neg ? "-" : "") + body;
}
function evalText(row, which){
  const mate = row["mate_"+which], cp = row["eval_"+which];
  if(mate === 0) return "#";                       // the game ended here
  if(mate !== null && mate !== undefined) return (mate > 0 ? "#" : "#-") + Math.abs(mate);
  return (cp >= 0 ? "+" : "") + (cp/100).toFixed(2);
}

/* ---------- board ---------- */
function parseFen(fen){
  const rows = fen.split(" ")[0].split("/"), out = {};
  rows.forEach((row, i) => {
    let file = 0;
    for(const ch of row){
      if(ch >= "1" && ch <= "8") file += +ch;
      else { out[FILES[file] + (8-i)] = ch; file++; }
    }
  });
  return out;
}
function plyCount(){ return S && S.moves ? S.moves.length : 0; }
function viewPly(){ return histPly === null ? plyCount() : histPly; }
function browsing(){ return histPly !== null; }
function shownFen(){
  if(!browsing()) return S.fen;
  if(histPly === 0) return S.start_fen;
  return (S.moves[histPly-1] || {}).fen || S.fen;
}
/* the review row for the move played *from* the position on screen */
function nextRow(){ return (S.review.rows || []).find(r => r.ply === viewPly() + 1) || null; }
function drawBoard(){
  const board = $("#board"), pieces = parseFen(shownFen());
  const ranks = orient === "white" ? [8,7,6,5,4,3,2,1] : [1,2,3,4,5,6,7,8];
  const files = orient === "white" ? [0,1,2,3,4,5,6,7] : [7,6,5,4,3,2,1,0];
  const targets = (!browsing() && sel && S.legal[sel]) ? S.legal[sel] : [];
  const played = viewPly() > 0 ? S.moves[viewPly()-1] : null;
  const last = browsing() ? (played ? [played.uci.slice(0,2), played.uci.slice(2,4)] : null)
                          : S.last_move;
  board.innerHTML = "";
  for(const rank of ranks) for(const fi of files){
    const name = FILES[fi] + rank;
    const cell = document.createElement("div");
    cell.className = "sq " + ((fi + rank) % 2 ? "light" : "dark");
    if(last && last.includes(name)) cell.classList.add("last");
    if(name === sel) cell.classList.add("sel");
    if(name === S.check_square && !browsing()) cell.classList.add("check");
    cell.dataset.sq = name;
    const piece = pieces[name];
    if(piece){
      cell.classList.add("occupied");
      const svg = document.createElementNS("http://www.w3.org/2000/svg","svg");
      svg.setAttribute("class","pc"); svg.setAttribute("viewBox","0 0 45 45");
      const use = document.createElementNS("http://www.w3.org/2000/svg","use");
      use.setAttribute("href", "#pc-" + (piece === piece.toUpperCase() ? "w" : "b") + piece.toLowerCase());
      svg.appendChild(use); cell.appendChild(svg);
    }
    if(targets.includes(name)){ const dot = document.createElement("div"); dot.className = "dot"; cell.appendChild(dot); }
    if(rank === (orient === "white" ? 1 : 8)){
      const c = document.createElement("span"); c.className = "coord f"; c.textContent = FILES[fi]; cell.appendChild(c);
    }
    if(fi === (orient === "white" ? 0 : 7)){
      const c = document.createElement("span"); c.className = "coord r"; c.textContent = rank; cell.appendChild(c);
    }
    cell.addEventListener("click", () => click(name));
    board.appendChild(cell);
  }
  drawArrow(browsing() ? nextRow() : null);
  checkGeometry();
}

/* The board's geometry has been wrong twice, and neither time did it announce itself.
   Measure it once and say so, rather than leaving it to be noticed. */
let geometryChecked = false;
function checkGeometry(){
  if(geometryChecked) return;
  const cells = $("#board").children;
  if(cells.length !== 64) return;
  const first = cells[0].getBoundingClientRect();
  if(first.height < 1) return;                 // not laid out yet; try again next draw
  geometryChecked = true;
  let worst = 0;
  for(const cell of cells){
    const r = cell.getBoundingClientRect();
    worst = Math.max(worst, Math.abs(r.height - first.height),
                            Math.abs(r.width - first.width),
                            Math.abs(r.height - r.width));
  }
  if(worst > 2){
    console.warn("board squares differ by up to", worst.toFixed(1), "px");
    flash(`The board rendered unevenly: squares differ by up to ${worst.toFixed(0)}px.`);
  }
}
function centre(name){
  const fi = FILES.indexOf(name[0]), rank = +name[1];
  const x = orient === "white" ? fi : 7-fi, y = orient === "white" ? 8-rank : rank-1;
  return [x + .5, y + .5];
}
function drawArrow(row){
  const svg = $("#arrows");
  svg.innerHTML = "";
  if(!row || !row.best_uci || row.best_uci === row.uci) return;
  const [x1,y1] = centre(row.best_uci.slice(0,2)), [x2,y2] = centre(row.best_uci.slice(2,4));
  const dx = x2-x1, dy = y2-y1, len = Math.hypot(dx,dy);
  const ux = dx/len, uy = dy/len, tipX = x2 - ux*.18, tipY = y2 - uy*.18;
  svg.innerHTML =
    `<line x1="${x1}" y1="${y1}" x2="${tipX-ux*.16}" y2="${tipY-uy*.16}"
       stroke="var(--live)" stroke-width=".13" stroke-linecap="round" opacity=".85"/>
     <polygon points="${tipX},${tipY} ${tipX-ux*.3-uy*.16},${tipY-uy*.3+ux*.16} ${tipX-ux*.3+uy*.16},${tipY-uy*.3-ux*.16}"
       fill="var(--live)" opacity=".85"/>`;
}
function click(name){
  if(!S || browsing()) return;          // the past is read only
  if(S.phase !== "playing" || S.thinking) return;
  if(S.seats[S.turn].kind !== "human") return;
  if(sel && S.legal[sel] && S.legal[sel].includes(name)){
    if(S.promotions.includes(sel + name)) return askPromo(sel, name);
    const from = sel; sel = null; drawBoard();
    api("/api/move", {uci: from + name}).then(apply);
    return;
  }
  sel = S.legal[name] ? name : null;
  drawBoard();
}
function askPromo(from, to){
  const box = $("#promo");
  const colour = S.turn === "white" ? "w" : "b";
  box.innerHTML = ["q","r","b","n"].map(p =>
    `<button data-p="${p}"><svg viewBox="0 0 45 45"><use href="#pc-${colour}${p}"/></svg></button>`).join("");
  box.classList.add("on");
  box.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    box.classList.remove("on"); sel = null;
    api("/api/move", {uci: from + to + b.dataset.p}).then(apply);
  }));
}

/* ---------- panes ---------- */
function renderMoves(){
  const pane = $("#paneMoves");
  if(!S.moves.length){ pane.innerHTML = '<p class="empty">No moves yet.</p>'; return; }
  const evals = S.evals || {}, showEval = Object.keys(evals).length > 0;
  let html = `<table><tr><th class="n">#</th><th>move</th><th class="n">time</th>${
    showEval ? '<th class="n">eval</th>' : ""}</tr>`;
  S.moves.forEach((m, i) => {
    const ply = i + 1, e = evals[String(ply)];
    html += `<tr class="mv ${viewPly() === ply ? "on" : ""}" data-ply="${ply}">
      <td class="n">${m.move_no}${m.side === "white" ? "." : "…"}</td>
      <td><span class="san">${esc(m.san)}</span>${
        m.note ? ` <span class="tag blunder">${esc(m.note)}</span>` : ""}</td>
      <td class="n">${m.ms ? (m.ms/1000).toFixed(2) : ""}</td>
      ${showEval ? `<td class="n">${e ? shortEval(e) : ""}</td>` : ""}</tr>`;
  });
  pane.innerHTML = html + "</table>";
  pane.querySelectorAll("tr.mv").forEach(tr =>
    tr.addEventListener("click", () => goTo(+tr.dataset.ply)));
}
function renderReview(){
  const pane = $("#reviewOut"), r = S.review;
  $("#noEngine").style.display = S.engine_available ? "none" : "block";
  const run = $("#runReview");
  run.disabled = !S.engine_available || r.running || !S.moves.length;
  run.textContent = r.running ? "Reviewing" : "Review with Stockfish";

  let html = "";
  if(r.running || (r.total && r.done)){
    const pct = r.total ? Math.round(100*r.done/r.total) : 0;
    html += `<div class="bar"><i style="width:${pct}%"></i></div>
      <p class="note">${r.running ? `Analysing position ${r.done} of ${r.total}` :
        `${esc(r.engine)} at ${esc(r.limit)}, ${r.total} positions`}</p>`;
  }
  if(r.error) html += `<p class="note" style="color:var(--bad)">${esc(r.error)}</p>`;

  if(r.rows.length){
    html += `<svg id="spark" viewBox="0 0 ${Math.max(r.rows.length,2)} 100" preserveAspectRatio="none"></svg>`;
    const a = r.acpl || {};
    if(a.white !== undefined && a.white !== null)
      html += `<p class="note">Average centipawn loss — White ${a.white}, Black ${a.black}</p>`;
    html += "<table><tr><th class='n'>#</th><th>move</th><th class='n'>eval</th><th class='n'>lost</th><th>best</th></tr>";
    r.rows.forEach((row, i) => {
      const dots = row.side === "white" ? row.move_no + "." : row.move_no + "…";
      html += `<tr class="mv ${viewPly() === row.ply - 1 ? "on" : ""}" data-ply="${row.ply - 1}">
        <td class="n">${dots}</td>
        <td><span class="san">${esc(row.san)}</span> <span class="tag ${row.tag}">${row.tag === "ok" ? "" : row.tag}</span></td>
        <td class="n">${evalText(row,"after")}</td>
        <td class="n" style="color:${row.loss >= 150 ? "var(--bad)" : row.loss >= 75 ? "var(--warm)" : "var(--faint)"}">${row.loss || ""}</td>
        <td>${row.best_uci === row.uci ? "" : esc(row.best_san || "")}</td></tr>`;
    });
    html += "</table>";
    html += `<div class="legend"><span>lost = centipawns given away</span>
      <span class="tag inaccuracy">&lt;75</span><span class="tag mistake">&lt;150</span><span class="tag blunder">150+</span></div>`;
  }
  pane.innerHTML = html;
  pane.querySelectorAll("tr.mv").forEach(tr =>
    tr.addEventListener("click", () => goTo(+tr.dataset.ply)));
  if(r.rows.length) drawSpark(r.rows);
}
function renderBoardFor(){ lastFen = shownFen(); drawBoard(); }

/* Stepping into the past holds a running game so the present does not move while you
   look at it; returning releases it, but only if the hold was ours to release. */
function goTo(target){
  const max = plyCount();
  const clamped = Math.max(0, Math.min(max, target));
  histPly = clamped === max ? null : clamped;
  sel = null;
  if(browsing() && S.phase === "playing" && !S.paused && !pausedByBrowsing){
    pausedByBrowsing = true;
    api("/api/pause").then(apply);
  } else if(!browsing() && pausedByBrowsing){
    pausedByBrowsing = false;
    api("/api/go").then(apply);
  }
  renderBoardFor(); renderTransport(); renderEval(); renderStatus(); renderButtons();
  if(tab === "moves") renderMoves();
  if(tab === "review") renderReview();
}
function renderTransport(){
  const max = plyCount(), at = viewPly();
  $("#tPos").textContent = max ? `${at} / ${max}` : "—";
  $("#transport").classList.toggle("browsing", browsing());
  $("#tStart").disabled = $("#tBack").disabled = at <= 0;
  $("#tFwd").disabled = $("#tNow").disabled = !browsing();
}
function shortEval(e){
  if(e.mate === 0) return "#";
  if(e.mate !== null && e.mate !== undefined) return (e.mate > 0 ? "#" : "#-") + Math.abs(e.mate);
  const v = e.cp / 100;
  return (v >= 0 ? "+" : "") + (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1));
}
function evalShare(e){
  if(e.mate === 0) return e.cp > 0 ? 100 : 0;
  if(e.mate !== null && e.mate !== undefined) return e.mate > 0 ? 100 : 0;
  return 50 + 50 * Math.tanh(e.cp / 400);
}
function renderEval(){
  const bar = $("#evalbar"), evals = S.evals || {};
  const show = S.live_eval || Object.keys(evals).length > 0;
  bar.hidden = !show;
  if(!show) return;
  bar.classList.toggle("flip", orient === "black");
  bar.classList.toggle("pending", !!S.evaluating && !browsing());
  const e = evals[String(viewPly())];
  if(!e){ bar.querySelector(".v").textContent = "·"; return; }
  const share = Math.max(0, Math.min(100, evalShare(e)));
  bar.querySelector("i").style.height = share + "%";
  bar.querySelector(".v").textContent = shortEval(e);
  bar.classList.toggle("on-dark", share < 16);
}
function drawSpark(rows){
  const svg = $("#spark"); if(!svg) return;
  const clamp = v => Math.max(-600, Math.min(600, v));
  const pts = rows.map((r,i) => [i, 50 - clamp(r.eval_after)/12]);
  const line = pts.map(([x,y],i) => (i ? "L" : "M") + x + " " + y.toFixed(1)).join(" ");
  const area = line + ` L ${rows.length-1} 50 L 0 50 Z`;
  const mark = browsing() ? `<line x1="${viewPly()}" y1="0" x2="${viewPly()}" y2="100"
      stroke="var(--live)" stroke-width=".6" opacity=".9"/>` : "";
  svg.innerHTML = `<line x1="0" y1="50" x2="${rows.length}" y2="50" stroke="var(--line)" stroke-width=".8"/>
    <path d="${area}" fill="var(--text)" opacity=".10"/>
    <path d="${line}" fill="none" stroke="var(--text)" stroke-width=".9" vector-effect="non-scaling-stroke"/>${mark}`;
}
function renderLog(){
  const which = $("#logPick").value;
  const text = S.seats[which].log || "";
  const box = $("#log"), atEnd = box.scrollTop + box.clientHeight >= box.scrollHeight - 20;
  box.textContent = text || "This bot has printed nothing. Anything it prints lands here.";
  if(atEnd) box.scrollTop = box.scrollHeight;
}

/* ---------- shell ---------- */
function renderClocks(){
  const top = orient === "white" ? "black" : "white", bottom = orient === "white" ? "white" : "black";
  for(const [id, side] of [["#topClock", top], ["#botClock", bottom]]){
    const seat = S.seats[side], el = $(id);
    const ms = seat.clock_enabled ? seat.clock_ms : null;
    el.querySelector(".who").innerHTML =
      `<b>${esc(seat.label)}</b> ${side === S.turn && S.thinking ? "· thinking" : ""}`;
    el.querySelector(".t").textContent = seat.clock_enabled ? clockText(ms) : "no clock";
    el.classList.toggle("on", !!seat.clock_running);
    el.classList.toggle("low", seat.clock_enabled && ms !== null && ms < 15000);
  }
}
function renderStatus(){
  const s = $("#status"); s.classList.remove("alert");
  let text = "";
  if(browsing()){
    s.classList.add("alert");
    s.innerHTML = `<b>Position ${viewPly()} of ${plyCount()}, read only.</b> ` +
      (pausedByBrowsing ? "The game is held while you look. " : "") +
      "Press Present to resume play.";
    return;
  }
  if(S.phase === "idle") text = S.message;
  else if(S.phase === "starting") text = "<b>Starting the bots.</b> Import time counts against the 60s budget.";
  else if(S.phase === "review") text = esc(S.message);
  else if(S.phase === "over"){
    const who = {white:"White wins", black:"Black wins", draw:"Drawn"}[S.result] || "Game over";
    text = `<b>${who}</b> — ${esc(S.message || S.termination || "")}`;
    s.classList.add("alert");
  } else {
    const seat = S.seats[S.turn];
    text = S.paused ? "<b>Paused.</b> Press “Bot moves now” to continue."
      : seat.kind === "human" ? "Your move." : `<b>${esc(seat.label)}</b> is thinking.`;
    if(S.plies >= S.ply_cap - 20) text += ` Ply ${S.plies} of ${S.ply_cap} before material adjudication.`;
  }
  s.innerHTML = text;
}
function apply(state){
  if(!state || !state.phase) return;
  S = state;
  if(shownFen() !== lastFen || sel !== null) renderBoardFor();
  renderClocks(); renderStatus(); renderTransport(); renderEval();
  if(tab === "moves") renderMoves();
  if(tab === "review") renderReview();
  if(tab === "log") renderLog();
  renderButtons();
}
function renderButtons(){
  const playing = S.phase === "playing", past = browsing();
  $("#back").disabled = !S.moves.length || S.thinking || S.phase === "review" || past;
  $("#go").disabled = !playing || S.thinking || past;
  $("#pause").disabled = !playing || past;
  $("#resign").disabled = !playing || past;
}
async function poll(){
  try{
    const state = await (await fetch("/api/state")).json();
    const changed = !S || state.version !== S.version;
    S = state;
    if(changed){ apply(state); } else { renderClocks(); if(tab === "log") renderLog(); }
  }catch(e){}
  setTimeout(poll, S && (S.thinking || S.review.running) ? 200 : 500);
}

/* ---------- wiring ---------- */
function seatOptions(select, def){
  select.innerHTML = '<option value="human">You</option>' +
    agents.map(a => `<option value="${esc(a.path)}">${esc(a.label)}</option>`).join("");
  select.value = def;
}
function seatSpec(select){
  return select.value === "human" ? {kind:"human"} : {kind:"bot", path:select.value};
}
async function boot(){
  const meta = await getJSON("/api/agents");
  agents = meta.agents || [];
  seatOptions($("#whitePick"), "human");
  seatOptions($("#blackPick"), agents.length ? agents[0].path : "human");
  if(!meta.openings) $("#startPick").querySelector('option[value=opening]').disabled = true;
  const first = await getJSON("/api/state");
  if(!first.engine_available){
    $("#liveEval").disabled = true;
    $("#liveNote").textContent =
      "No Stockfish binary found, so the bar is unavailable. Start this tool with " +
      "--engine /path/to/stockfish, or set SPAR_ENGINE.";
  }

  $("#start").addEventListener("click", () => {
    const white = seatSpec($("#whitePick"));
    orient = white.kind === "human" ? "white" : (seatSpec($("#blackPick")).kind === "human" ? "black" : "white");
    histPly = null; pausedByBrowsing = false; lastFen = null; sel = null;
    api("/api/new", {
      white, black: seatSpec($("#blackPick")),
      base_ms: Math.round(+$("#baseS").value * 1000),
      increment_ms: +$("#incMs").value,
      bot_clock: document.querySelector("input[name=botclock]:checked").value,
      bot_fixed_ms: +$("#fixedMs").value,
      human_clock: $("#humanClock").checked,
      start: $("#startPick").value,
      fen: $("#fen").value,
      live_eval: $("#liveEval").checked,
      live_kind: "depth",
      live_value: +$("#liveDepth").value
    }).then(apply);
  });
  document.querySelectorAll("input[name=botclock]").forEach(r => r.addEventListener("change", () => {
    $("#fixedField").style.display = r.value === "fixed" && r.checked ? "block" : "none";
  }));
  $("#startPick").addEventListener("change", e => {
    $("#fenField").style.display = e.target.value === "fen" ? "block" : "none";
  });
  $("#liveEval").addEventListener("change", e => {
    $("#liveField").style.display = e.target.checked ? "block" : "none";
  });
  $("#tStart").addEventListener("click", () => goTo(0));
  $("#tBack").addEventListener("click", () => goTo(viewPly() - 1));
  $("#tFwd").addEventListener("click", () => goTo(viewPly() + 1));
  $("#tNow").addEventListener("click", () => goTo(plyCount()));
  $("#flip").addEventListener("click", () => {
    orient = orient === "white" ? "black" : "white"; drawBoard(); renderClocks(); renderEval();
  });
  $("#back").addEventListener("click", () => { histPly = null; lastFen = null; api("/api/takeback").then(apply); });
  $("#go").addEventListener("click", () => api("/api/go").then(apply));
  $("#pause").addEventListener("click", () => api("/api/pause").then(apply));
  $("#resign").addEventListener("click", () => api("/api/resign").then(apply));
  $("#loadPgn").addEventListener("click", () => {
    histPly = null; pausedByBrowsing = false; lastFen = null;
    api("/api/loadpgn", {pgn: $("#pgnIn").value}).then(apply);
  });
  $("#savePgn").addEventListener("click", async () => {
    const text = await (await fetch("/api/pgn")).text();
    const url = URL.createObjectURL(new Blob([text], {type:"text/plain"}));
    const a = document.createElement("a");
    a.href = url; a.download = "sparring-" + Date.now() + ".pgn"; a.click();
    URL.revokeObjectURL(url);
  });
  $("#runReview").addEventListener("click", () => {
    api("/api/review", {kind: $("#rKind").value, value: +$("#rVal").value}).then(apply);
  });
  $("#rKind").addEventListener("change", e => {
    $("#rVal").value = {depth:16, nodes:200000, movetime:300}[e.target.value];
  });
  $("#logPick").addEventListener("change", renderLog);
  document.querySelectorAll(".tabs button").forEach(b => b.addEventListener("click", () => {
    tab = b.dataset.tab;
    document.querySelectorAll(".tabs button").forEach(x => x.classList.toggle("on", x === b));
    $("#paneMoves").style.display = tab === "moves" ? "block" : "none";
    $("#paneReview").style.display = tab === "review" ? "block" : "none";
    $("#paneLog").style.display = tab === "log" ? "block" : "none";
    apply(S);
  }));
  document.addEventListener("keydown", e => {
    if(!S || e.target.matches("input,textarea,select")) return;
    if(e.key === "f") $("#flip").click();
    if(e.key === "ArrowLeft"){ e.preventDefault(); goTo(viewPly() - 1); }
    if(e.key === "ArrowRight"){ e.preventDefault(); goTo(viewPly() + 1); }
    if(e.key === "Home"){ e.preventDefault(); goTo(0); }
    if(e.key === "End" || e.key === "Escape"){ e.preventDefault(); goTo(plyCount()); }
  });
  poll();
}
boot();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description="Play a build in the browser.")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--engine", help="path to Stockfish; else $SPAR_ENGINE, else PATH")
    parser.add_argument("--sf-threads", type=int, default=2)
    parser.add_argument("--sf-hash", type=int, default=256)
    parser.add_argument("--open", action="store_true", help="open a browser window")
    parser.add_argument("--no-build", action="store_true", help="skip materialising versions/")
    arguments = parser.parse_args()

    if not arguments.no_build:
        materialise()

    session = Session(arguments.engine, arguments.sf_threads, arguments.sf_hash)
    Handler.session = session
    server = ThreadingHTTPServer((arguments.host, arguments.port), Handler)
    server.daemon_threads = True

    url = f"http://{arguments.host}:{arguments.port}"
    found = session.find_engine()
    print(f"sparring board on {url}")
    print(f"  agents:    {len(discover_agents())} playable")
    print(f"  stockfish: {found or 'not found -- review is disabled until you pass --engine'}")
    if arguments.open:
        threading.Thread(target=lambda: (time.sleep(0.4), webbrowser.open(url)), daemon=True).start()

    # Ctrl-C already unwinds through the finally below; SIGTERM would not, and would
    # leave the agent subprocesses running.
    def terminate(*_args) -> None:
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, terminate)
    except ValueError:
        pass  # not the main thread; the atexit hook is the fallback

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        session.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
