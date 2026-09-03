"""A Stockfish sparring partner that speaks the AI Chessathon agent contract.

Project copy is named bot_stockfish_spar.py. It goes in the repo at
baselines/stockfish/agent.py -- the harness imports agents by that name, and each
agent needs its own directory. Do not put it at the repo root; see below.

    THIS NEVER SHIPS. It lives under baselines/ on purpose: harness/package.py
    globs every *.py at the repo ROOT into submission.zip, so a copy of this file
    at the root would put an engine wrapper inside your upload. There is an
    import-time guard below, but package.py zips without importing, so also run
    `unzip -l submission.zip | grep -iE 'spar|stockfish|engine'` before uploading. Shipping or
    calling Stockfish, Lc0, Maia or any existing engine from the zip is an
    instant disqualification, and it is checked after the fact. Keep the binary
    outside the repo (or gitignored) and never pass this to `make zip`.

What it is for: an absolute yardstick. tools/bench.py measures you against your
own previous version, which tells you whether a change helped. This tells you
where you actually sit. Stockfish at a fixed node count is reproducible across
machines and does not care how fast your laptop is that afternoon, so the node
count at which you score 50% is a single number you can track across versions.

Setup
-----
Put a stockfish binary somewhere and point at it, or leave it on PATH:

    export SPAR_ENGINE=/usr/local/bin/stockfish     # optional if on PATH

Choosing a level
----------------
One environment variable, a comma-separated list of key:value pairs. Pick a work
limit, optionally cap the strength, optionally add noise.

Work limits, one of:

    nodes:10000     fixed nodes per move   (the workhorse: machine independent)
    depth:6         fixed depth per move
    movetime:200    fixed milliseconds per move
    clock           Stockfish manages the real 120s + 0.5s clock itself

Strength caps, combinable with any work limit:

    elo:1600        UCI_LimitStrength. The floor is 1320 on Stockfish 16/17.
    skill:5         Skill Level, 0 to 20.

Either strength cap on its own defaults the work limit to movetime:100, which
keeps a benchmark run to a sensible length.

Other modifiers:

    noise:40        pick randomly among moves within 40cp of best (MultiPV)
    threads:1       default 1; keep it at 1 or results stop being reproducible
    hash:16         megabytes, default 16
    seed:7          seed for the noise picker
    overhead:40     milliseconds held back from the clock

Examples:

    SPAR_LEVEL="nodes:2000"
    SPAR_LEVEL="elo:1400,movetime:150"
    SPAR_LEVEL="skill:3,nodes:50000"
    SPAR_LEVEL="nodes:5000,noise:50,seed:7"

Using it
--------
    SPAR_LEVEL="nodes:2000" uv run python -m harness.play --black baselines/stockfish
    SPAR_LEVEL="nodes:2000" uv run python -m tools.bench --opponent baselines/stockfish --games 40
    uv run python -m tools.ladder --games 30        # sweeps a whole range at once

Notes on making the comparison honest
-------------------------------------
  * Node limits, not time limits. A time-limited engine measures your CPU, not
    your agent, and the number stops meaning anything on a different machine.
  * Stockfish is not a substitute for self-play regression testing. It is far
    stronger per node than anything you will write in Python and it fails in
    completely different ways, so a change can help against Stockfish and hurt
    against the field. Use tools/bench.py against your last version for A/B.
  * Both sides are deterministic without noise, so the same opening produces the
    same game every time. Add `noise:` when you want a varied sample.
  * This tracks the game history the same way agent.py has to, so Stockfish sees
    repetitions and the fifty-move clock. Without that it shuffles in won
    positions, gets draws claimed against it, and flatters your results.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import chess
import chess.engine

# --------------------------------------------------------------------------
# 0. Refuse to run from anywhere that gets packaged
# --------------------------------------------------------------------------

# harness/package.py zips every *.py at the repo root. If this file is sitting
# next to harness/, it is at the root, and the next `make zip` puts an engine
# wrapper inside the submission. That is a disqualification and it is checked
# after the fact, so fail here rather than let it happen quietly.
if (Path(__file__).resolve().parent / "harness").is_dir():
    raise RuntimeError(
        "The Stockfish sparring agent is at the repo root, where "
        "harness/package.py will zip it into submission.zip. Move it to "
        "baselines/stockfish/agent.py. Shipping an engine wrapper is a "
        "disqualification."
    )

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_LEVEL: Final = "nodes:10000"

# Places a binary tends to live, tried in order after PATH.
_CANDIDATE_PATHS: Final = (
    "/usr/games/stockfish",
    "/usr/local/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
    "/usr/bin/stockfish",
)

_LIMITS: Final = frozenset({"nodes", "depth", "movetime", "clock"})
_CAPS: Final = frozenset({"elo", "skill"})
_MODIFIERS: Final = frozenset({"noise", "threads", "hash", "seed", "overhead"})

# Safety net so the sparring partner never flags and quietly hands you a win. The
# referee deducts wall time around the whole request and the watchdog does not
# forgive, so every mode gets capped well inside the clock.
_TIME_FRACTION: Final = 1.0 / 12.0
_MIN_CEILING_MS: Final = 20.0

# A strength cap with no work limit still needs something to bound the search.
_CAPPED_DEFAULT_MOVETIME_MS: Final = 100

_INCREMENT_S: Final = 0.5


@dataclass(frozen=True)
class Level:
    """A parsed SPAR_LEVEL spec."""

    limit: str = "nodes"
    value: int = 10_000
    elo: int | None = None
    skill: int | None = None
    noise_cp: int = 0
    threads: int = 1
    hash_mb: int = 16
    seed: int = 0
    overhead_ms: int = 40

    @property
    def label(self) -> str:
        parts = [self.limit if self.limit == "clock" else f"{self.limit}:{self.value}"]
        if self.elo is not None:
            parts.append(f"elo:{self.elo}")
        if self.skill is not None:
            parts.append(f"skill:{self.skill}")
        if self.noise_cp:
            parts.append(f"noise:{self.noise_cp}")
        return ",".join(parts)


def parse_level(spec: str) -> Level:
    """Parse "nodes:10000,noise:40" into a Level. Raises on anything unknown."""
    modifiers: dict[str, int] = {}
    caps: dict[str, int] = {}
    limit = ""
    value = 0

    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, _, raw = chunk.partition(":")
        key = key.strip().lower()
        if key in _LIMITS:
            if limit:
                raise ValueError(f"two work limits in {spec!r}: {limit} and {key}")
            limit = key
            value = int(raw) if raw else 0
        elif key in _CAPS:
            caps[key] = int(raw)
        elif key in _MODIFIERS:
            modifiers[key] = int(raw)
        else:
            known = sorted(_LIMITS | _CAPS | _MODIFIERS)
            raise ValueError(f"unknown key {key!r} in {spec!r}; known keys are {known}")

    if not limit:
        if caps:
            limit, value = "movetime", _CAPPED_DEFAULT_MOVETIME_MS
        else:
            raise ValueError(f"no work limit in {spec!r}; one of {sorted(_LIMITS)} is required")

    fallback = Level()
    return Level(
        limit=limit,
        value=value,
        elo=caps.get("elo"),
        skill=caps.get("skill"),
        noise_cp=modifiers.get("noise", fallback.noise_cp),
        threads=modifiers.get("threads", fallback.threads),
        hash_mb=modifiers.get("hash", fallback.hash_mb),
        seed=modifiers.get("seed", fallback.seed),
        overhead_ms=modifiers.get("overhead", fallback.overhead_ms),
    )


def find_engine() -> str:
    """Locate a stockfish binary, preferring SPAR_ENGINE, then PATH."""
    override = os.environ.get("SPAR_ENGINE")
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(f"SPAR_ENGINE={override!r} is not a file")
        return override
    found = shutil.which("stockfish")
    if found:
        return found
    for candidate in _CANDIDATE_PATHS:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "no stockfish binary found. Install one and put it on PATH, or set "
        "SPAR_ENGINE=/path/to/stockfish. Keep the binary out of the repo so it "
        "can never end up in submission.zip."
    )


# --------------------------------------------------------------------------
# Engine lifecycle
# --------------------------------------------------------------------------

LEVEL: Final = parse_level(os.environ.get("SPAR_LEVEL") or DEFAULT_LEVEL)

_engine: chess.engine.SimpleEngine | None = None
_random = random.Random(LEVEL.seed)
_multipv_ok = True


def _configure(engine: chess.engine.SimpleEngine, level: Level) -> None:
    settings: dict[str, object] = {}
    if "Threads" in engine.options:
        settings["Threads"] = max(1, level.threads)
    if "Hash" in engine.options:
        settings["Hash"] = max(1, level.hash_mb)
    if "Move Overhead" in engine.options:
        settings["Move Overhead"] = max(0, level.overhead_ms)

    if level.elo is not None:
        option = engine.options.get("UCI_Elo")
        if option is None or "UCI_LimitStrength" not in engine.options:
            raise RuntimeError("this engine has no UCI_Elo; use nodes: or depth: instead")
        low = option.min if option.min is not None else 1320
        high = option.max if option.max is not None else 3190
        target = min(max(level.elo, low), high)
        if target != level.elo:
            print(
                f"[spar] elo:{level.elo} is outside this engine's {low}-{high} range, "
                f"clamped to {target}. Below the floor there is no Elo setting to ask "
                f"for; use nodes: to go weaker.",
                file=sys.stderr,
            )
        settings["UCI_LimitStrength"] = True
        settings["UCI_Elo"] = target

    if level.skill is not None:
        if "Skill Level" not in engine.options:
            raise RuntimeError("this engine has no Skill Level; use nodes: or depth: instead")
        settings["Skill Level"] = min(max(level.skill, 0), 20)

    engine.configure(settings)


def _start() -> chess.engine.SimpleEngine:
    global _engine
    if _engine is None:
        path = find_engine()
        _engine = chess.engine.SimpleEngine.popen_uci(path)
        _configure(_engine, LEVEL)
        name = _engine.id.get("name", "unknown")
        print(f"[spar] {name} at {LEVEL.label} ({path})", file=sys.stderr)
    return _engine


def _shutdown() -> None:
    """Close the engine on the way out.

    SimpleEngine runs its event loop on a thread that keeps the interpreter alive,
    so without this a script that imports this module never exits. It also matters
    for benchmark runs: harness/sandbox.py kills the runner outright between games,
    and a few hundred games is a few hundred orphaned engines if they do not go.
    Stockfish does exit on stdin EOF when its parent dies, so this is belt and
    braces rather than the only thing holding it up.
    """
    global _engine
    engine, _engine = _engine, None
    if engine is not None:
        with contextlib.suppress(Exception):
            engine.close()


atexit.register(_shutdown)


# --------------------------------------------------------------------------
# Game state
# --------------------------------------------------------------------------

# We are handed a FEN per move and nothing else, so the move history has to be
# rebuilt: find the legal move that turns the position we last saw into the one
# we were just given, and push it. Stockfish then sees repetitions and the
# fifty-move clock, which it needs to convert won endings instead of shuffling.

_board: chess.Board | None = None


def _sync(fen: str) -> chess.Board:
    global _board
    target = chess.Board(fen)
    target_key = target._transposition_key()

    if _board is not None:
        if _board._transposition_key() == target_key:
            return _board
        for move in _board.legal_moves:
            _board.push(move)
            if _board._transposition_key() == target_key:
                return _board
            _board.pop()

    _board = target
    return _board


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def _ceiling_seconds(time_left_ms: int, level: Level) -> float:
    """A wall-clock cap applied in every mode so the sparring partner cannot flag."""
    usable = time_left_ms - level.overhead_ms
    if usable <= _MIN_CEILING_MS:
        return 0.0
    return max(_MIN_CEILING_MS, usable * _TIME_FRACTION) / 1000.0


def _limit(level: Level, time_left_ms: int) -> chess.engine.Limit:
    ceiling = _ceiling_seconds(time_left_ms, level)
    if level.limit == "clock":
        # Stockfish does its own time management. It only ever learns our own
        # clock, so the opponent is given the same, which is close enough here.
        seconds = max(0.05, (time_left_ms - level.overhead_ms) / 1000.0)
        return chess.engine.Limit(
            white_clock=seconds,
            black_clock=seconds,
            white_inc=_INCREMENT_S,
            black_inc=_INCREMENT_S,
        )
    if level.limit == "nodes":
        return chess.engine.Limit(nodes=max(1, level.value), time=ceiling)
    if level.limit == "depth":
        return chess.engine.Limit(depth=max(1, level.value), time=ceiling)
    return chess.engine.Limit(time=min(max(0.001, level.value / 1000.0), ceiling))


def _noisy_move(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    limit: chess.engine.Limit,
    level: Level,
) -> chess.Move | None:
    """Pick uniformly among moves within noise_cp of the best. None if unavailable."""
    global _multipv_ok
    if not _multipv_ok:
        return None
    try:
        infos = engine.analyse(board, limit, multipv=8, game="spar")
    except chess.engine.EngineError:
        _multipv_ok = False
        print("[spar] MultiPV unavailable, noise disabled", file=sys.stderr)
        return None
    scored: list[tuple[int, chess.Move]] = []
    for info in infos:
        principal = info.get("pv")
        score = info.get("score")
        if not principal or score is None:
            continue
        centipawns = score.pov(board.turn).score(mate_score=100_000)
        if centipawns is None:
            continue
        scored.append((centipawns, principal[0]))
    if not scored:
        return None
    best = max(centipawns for centipawns, _ in scored)
    pool = [move for centipawns, move in scored if best - centipawns <= level.noise_cp]
    return _random.choice(pool) if pool else None


def _pick(board: chess.Board, time_left_ms: int) -> chess.Move:
    engine = _start()
    limit = _limit(LEVEL, time_left_ms)

    if LEVEL.noise_cp > 0:
        move = _noisy_move(engine, board, limit, LEVEL)
        if move is not None and move in board.legal_moves:
            return move

    result = engine.play(board, limit, game="spar")
    if result.move is None or result.move not in board.legal_moves:
        raise RuntimeError(f"engine returned {result.move!r}, which is not legal here")
    return result.move


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation, matching the platform's contract."""
    board = _sync(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    if time_left_ms <= LEVEL.overhead_ms or len(legal) == 1:
        move = legal[0]
    else:
        try:
            move = _pick(board, time_left_ms)
        except Exception as error:
            # Loud, then legal. A silent fallback here would look like a weak
            # opponent rather than a broken one, which is the worst failure mode
            # for something you are using as a ruler.
            print(f"[spar] ENGINE FAILURE, playing a legal move: {error!r}", file=sys.stderr)
            move = legal[0]
    board.push(move)
    return move.uci()


def _warmup() -> None:
    """Start the engine inside the 60s init budget rather than on move one."""
    global _board
    try:
        _start()
        get_move(chess.STARTING_FEN, 1_000)
    except Exception as error:
        print(f"[spar] warm-up failed: {error!r}", file=sys.stderr)
    _board = None


_warmup()
