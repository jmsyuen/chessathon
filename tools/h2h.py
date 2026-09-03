"""Head-to-head between two builds, checkpointed after every game, with SPRT.

tools/bench.py plays a fixed number of games in one process and prints at the
end. A hundred games is over an hour, and anything that interrupts the run loses
all of it. This does the same job but appends each result to a JSON file, so a
run can be resumed, extended, or read while it is still going.

    # one process, as before
    python3 -m tools.h2h --agent versions/bot5 --opponent versions/bot4 --games 8

    # six in parallel on the PC, pooled SPRT, stops as soon as it is decisive
    python3 -m tools.h2h --agent versions/bot5 --opponent versions/bot4 \
        --games 130 --workers 6 --out results/bot5_vs_bot4.json

    # read a finished or running set
    python3 -m tools.h2h --report --out results/bot5_vs_bot4.json

The verdict is a sequential probability ratio test with elo0=0, elo1=5 and
alpha=beta=0.05, which is the merge gate the iteration log commits to. SPRT stops
as soon as the evidence is decisive instead of at a game count chosen in advance.

Sharding, and why it is by pair rather than by game
---------------------------------------------------
Games are played in pairs: the same opening from both sides, so an opening that
favours White cannot bias the result. The unit of work is therefore the *pair*,
not the game. Worker k of n takes pairs k, k+n, k+2n, ... which keeps every
worker internally colour-balanced and gives every worker a disjoint set of
openings.

Sharding by game instead would hand consecutive indices to different workers and
split pairs across them, so a run that ended unevenly would have unpaired
colours. Not sharding at all is worse still: each worker computes its index from
the length of *its own* checkpoint file, so six copies would every one of them
start at index 0 and replay the same games. That reads as a completed SPRT and is
six times the cost for one game's worth of information.

The pooled LLR is what decides. A per-shard LLR is meaningless — six undecided
shards can be one decisive result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import PLY_CAP
from harness.sandbox import local

try:
    from tools.openings import OPENINGS
except ImportError:  # fall back to the twelve that ship with bench.py
    from tools.bench import OPENINGS

# SPRT parameters. elo1=5 asks "is this at least five Elo better", which is the
# smallest difference worth a merge and still reachable in a few hundred games.
ELO0 = 0.0
ELO1 = 5.0
ALPHA = 0.05
BETA = 0.05
LOWER_BOUND = math.log(BETA / (1.0 - ALPHA))
UPPER_BOUND = math.log((1.0 - BETA) / ALPHA)

# How often the parent pools the shard files to check whether SPRT has decided.
POLL_SECONDS = 5.0

# No early stop below this many games however lopsided the result looks. A short
# streak is common and the regularisation below cannot fully price it.
MIN_GAMES_TO_DECIDE = 20


def _elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def log_likelihood_ratio(wins: int, draws: int, losses: int) -> float:
    """Normalised-score LLR, the standard 5-3-1 pentanomial-free version.

    Uses the score mean and variance directly, which is accurate enough at these
    game counts and avoids needing paired results.

    Half a win and half a loss are added as pseudo-observations. Without them a
    clean sweep has zero measured variance, and dividing by a clamped 1e-6 gives
    an LLR around 1e10 — which reads as overwhelming evidence from four games.
    That was harmless when a human read the number at the end of a serial run.
    It is not harmless now: `decided()` drives process termination, so an
    unregularised sweep would stop a 130-game run after four lucky results and
    report a merge.
    """
    games = wins + draws + losses
    if games == 0:
        return 0.0
    padded_wins = wins + 0.5
    padded_losses = losses + 0.5
    total = padded_wins + draws + padded_losses
    score = (padded_wins + draws * 0.5) / total
    variance = (
        padded_wins * (1.0 - score) ** 2
        + draws * (0.5 - score) ** 2
        + padded_losses * (0.0 - score) ** 2
    ) / total
    if variance <= 0.0:
        variance = 1e-6
    target0 = _elo_to_score(ELO0)
    target1 = _elo_to_score(ELO1)
    return games * ((target1 - target0) * (2 * score - target0 - target1)) / (2 * variance)


def score_interval(wins: int, draws: int, losses: int) -> tuple[float, float, float]:
    games = wins + draws + losses
    if games == 0:
        return 0.0, 0.0, 0.0
    score = (wins + draws / 2.0) / games
    variance = (
        wins * (1.0 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * (0.0 - score) ** 2
    ) / games
    error = 1.96 * math.sqrt(variance / games)
    return score, max(0.0, score - error), min(1.0, score + error)


def elo_difference(score: float) -> float:
    if score <= 0.0:
        return float("-inf")
    if score >= 1.0:
        return float("inf")
    return -400.0 * math.log10(1.0 / score - 1.0)


# --------------------------------------------------------------------------
# Checkpoint files
# --------------------------------------------------------------------------


def load(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            # A shard caught mid-write by a reader. Atomic writes make this rare,
            # but a poll every few seconds will eventually race a rename.
            return {"games": []}
    return {"games": []}


def save(path: Path, state: dict[str, Any]) -> None:
    """Write atomically so a polling parent never reads a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state))
    os.replace(temporary, path)


def shard_paths(out: Path, workers: int) -> list[Path]:
    if workers <= 1:
        return [out]
    return [out.with_suffix(f".shard{k}{out.suffix}") for k in range(workers)]


def pool(out: Path, workers: int) -> dict[str, Any]:
    """Merge every shard belonging to this run, plus the base file if present."""
    candidates = [out, *shard_paths(out, workers)] if workers > 1 else [out]
    # Also pick up shards from an earlier run with a different worker count.
    candidates += sorted(out.parent.glob(f"{out.stem}.shard*{out.suffix}"))

    merged: dict[str, Any] = {"games": []}
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        state = load(path)
        for key in ("agent", "opponent", "base_ms", "increment_ms"):
            if key in state:
                merged.setdefault(key, state[key])
        merged["games"].extend(state.get("games", []))
    return merged


def tally(games: list[dict[str, Any]]) -> tuple[int, int, int]:
    wins = sum(1 for g in games if g["outcome"] == "win")
    draws = sum(1 for g in games if g["outcome"] == "draw")
    losses = sum(1 for g in games if g["outcome"] == "loss")
    return wins, draws, losses


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def report_state(state: dict[str, Any]) -> int:
    games = state.get("games", [])
    assert isinstance(games, list)
    wins, draws, losses = tally(games)
    terminations: dict[str, int] = {}
    for game in games:
        name = game["termination"]
        terminations[name] = terminations.get(name, 0) + 1

    score, low, high = score_interval(wins, draws, losses)
    llr = log_likelihood_ratio(wins, draws, losses)
    elo = elo_difference(score)

    print(f"{state.get('agent')} vs {state.get('opponent')}")
    print(f"  {len(games)} games at {state.get('base_ms')}ms + {state.get('increment_ms')}ms")
    print(f"  +{wins} ={draws} -{losses}")
    print(f"  score {score:.1%}  (95% interval {low:.1%} to {high:.1%})")
    if math.isinf(elo):
        print(f"  elo   {'+inf' if elo > 0 else '-inf'} (a clean sweep bounds nothing)")
    else:
        print(
            f"  elo   {elo:+.0f}  (95% interval "
            f"{elo_difference(low):+.0f} to {elo_difference(high):+.0f})"
        )
    print(f"  LLR   {llr:+.2f}  (accept above {UPPER_BOUND:+.2f}, reject below {LOWER_BOUND:+.2f})")
    if llr >= UPPER_BOUND:
        print("  SPRT: PASS. The change is at least five Elo better; merge is justified.")
    elif llr <= LOWER_BOUND:
        print("  SPRT: FAIL. The change is not five Elo better; do not merge.")
    else:
        print("  SPRT: undecided. Keep playing.")
    if terminations:
        print("  terminations: " + ", ".join(f"{n} {c}" for n, c in sorted(terminations.items())))

    broken = {n: c for n, c in terminations.items() if n in FAILED_TERMINATIONS}
    if broken:
        print()
        print("  FAILED GAMES: " + ", ".join(f"{n} {c}" for n, c in broken.items()))
        print("  The zero-failure gate comes first. Fix these before reading anything above.")
        return 1
    return 0


def decided(state: dict[str, Any]) -> bool:
    """Whether the pooled evidence is decisive enough to stop the workers."""
    wins, draws, losses = tally(state.get("games", []))
    if wins + draws + losses < MIN_GAMES_TO_DECIDE:
        return False
    llr = log_likelihood_ratio(wins, draws, losses)
    return llr >= UPPER_BOUND or llr <= LOWER_BOUND


# --------------------------------------------------------------------------
# Playing
# --------------------------------------------------------------------------


def play(arguments: argparse.Namespace, out: Path, shard: int, shards: int) -> None:
    """Play this shard's allocation, checkpointing after every game."""
    state = load(out)
    state["agent"] = str(arguments.agent)
    state["opponent"] = str(arguments.opponent)
    state["base_ms"] = arguments.base_ms
    state["increment_ms"] = arguments.increment_ms
    state["shard"] = shard
    state["shards"] = shards
    games = state["games"]
    assert isinstance(games, list)

    agent_path = arguments.agent.resolve()
    opponent_path = arguments.opponent.resolve()
    started = time.monotonic()

    for _ in range(arguments.games):
        if arguments.deadline_s > 0 and time.monotonic() - started > arguments.deadline_s:
            print("deadline reached, stopping cleanly")
            break

        local_index = len(games)
        # Pairs, not games: this worker takes pairs shard, shard+shards, ... and
        # plays each one from both sides before moving to the next.
        pair = shard + (local_index // 2) * shards
        plays_white = local_index % 2 == 0
        opening = OPENINGS[pair % len(OPENINGS)]
        index = pair * 2 + (0 if plays_white else 1)

        white, black = (
            (agent_path, opponent_path) if plays_white else (opponent_path, agent_path)
        )
        game_started = time.monotonic()
        outcome = play_match(
            local(white),
            local(black),
            arguments.base_ms,
            arguments.increment_ms,
            ply_cap=arguments.ply_cap,
            start_fen=opening,
        )
        if outcome.result in ("draw", "void"):
            result = "draw"
        elif (outcome.result == "white") == plays_white:
            result = "win"
        else:
            result = "loss"
        games.append(
            {
                "index": index,
                "pair": pair,
                "colour": "white" if plays_white else "black",
                "outcome": result,
                "termination": outcome.termination,
                "seconds": round(time.monotonic() - game_started, 1),
                "opening": opening,
            }
        )
        # Checkpoint after every game. Background processes are reaped between
        # sandbox tool calls, and a long PC run should survive a lost terminal.
        save(out, state)
        symbol = {"win": "+", "draw": "=", "loss": "-"}[result]
        label = f"s{shard} " if shards > 1 else ""
        print(
            f"{label}game {index:>4} as {'white' if plays_white else 'black':<5} {symbol} "
            f"{outcome.termination:<22} {games[-1]['seconds']:>5.1f}s",
            flush=True,
        )


def supervise(arguments: argparse.Namespace) -> int:
    """Spawn worker shards, poll the pooled LLR, stop everything once decisive."""
    workers = arguments.workers
    paths = shard_paths(arguments.out, workers)

    # Round each worker's allocation up to an even number so it never ends on a
    # half-finished pair, which would leave that opening colour-imbalanced.
    per_worker = -(-arguments.games // workers)
    per_worker += per_worker % 2

    processes: list[subprocess.Popen[bytes]] = []
    for shard, path in enumerate(paths):
        command = [
            sys.executable, "-m", "tools.h2h",
            "--agent", str(arguments.agent),
            "--opponent", str(arguments.opponent),
            "--games", str(per_worker),
            "--base-ms", str(arguments.base_ms),
            "--increment-ms", str(arguments.increment_ms),
            "--ply-cap", str(arguments.ply_cap),
            "--out", str(path),
            "--shard", str(shard),
            "--shards", str(workers),
            "--deadline-s", str(arguments.deadline_s),
        ]
        processes.append(subprocess.Popen(command))

    print(
        f"{workers} workers x {per_worker} games, pairs split {workers} ways. "
        f"Pooling every {POLL_SECONDS:.0f}s; stopping early if SPRT decides.\n",
        flush=True,
    )

    try:
        while any(process.poll() is None for process in processes):
            time.sleep(POLL_SECONDS)
            if decided(pool(arguments.out, workers)):
                print("\nSPRT decided on the pooled result; stopping workers.\n", flush=True)
                break
    except KeyboardInterrupt:
        print("\ninterrupted; stopping workers (results so far are checkpointed)\n")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()

    # Deliberately not written back to arguments.out. The pooled view is
    # derived, and persisting it there would make the base file a second copy of
    # every shard's games — which pool() would then add to the shards again, so
    # resuming a run would count everything twice and compound each time.
    print()
    return report_state(pool(arguments.out, workers))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("versions/bot4"))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/bot1fix"))
    parser.add_argument("--games", type=int, default=8, help="games to add, across all workers")
    parser.add_argument("--base-ms", type=int, default=8_000)
    parser.add_argument("--increment-ms", type=int, default=500)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--out", type=Path, default=Path("/tmp/h2h.json"))
    parser.add_argument("--report", action="store_true", help="pool, print the summary, exit")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="parallel shards. Cap at physical cores; one game busies one core",
    )
    parser.add_argument("--shard", type=int, default=0, help="set by --workers; rarely manual")
    parser.add_argument("--shards", type=int, default=1, help="set by --workers; rarely manual")
    parser.add_argument(
        "--deadline-s", type=float, default=0.0,
        help="stop starting games after this many seconds. 0 means no limit. "
             "Sandbox runs want ~250 because bash calls are killed at ~5 min",
    )
    arguments = parser.parse_args()

    if arguments.report:
        return report_state(pool(arguments.out, max(arguments.workers, 1)))

    if arguments.workers > 1:
        return supervise(arguments)

    shards = max(arguments.shards, 1)
    play(arguments, arguments.out, arguments.shard, shards)
    if shards > 1:
        # A spawned worker. The parent pools every shard and reports once; a
        # per-shard report would interleave with the others and, worse, invite
        # someone to read a shard's LLR as if it meant something.
        return 0
    print()
    return report_state(pool(arguments.out, 1))


if __name__ == "__main__":
    sys.exit(main())
