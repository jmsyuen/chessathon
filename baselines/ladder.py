"""Find where an agent sits on a ladder of Stockfish levels.

    python3 -m tools.ladder --games 20
    python3 -m tools.ladder --games 40 --rungs nodes:1000,nodes:4000,nodes:16000
    python3 -m tools.ladder --games 40 --rungs elo:1320,elo:1500,elo:1700
    python3 -m tools.ladder --agent versions/v3 --games 30 --base-ms 30000

tools/bench.py answers "did this change help", by playing you against your own
previous version. This answers "where am I", by playing you against a fixed
external ruler and reporting the rung where you score 50%.

Track that one number across versions. It is machine independent, because a node
limit does not care how fast your laptop is, and it does not drift, because
Stockfish at 4000 nodes is the same opponent in a week as it is today.

The ruler is not the field. Stockfish is enormously stronger per node than
anything you will write in Python, and it blunders in different places, so do
not tune against it. Use it to know your altitude, and tools/bench.py to decide
what to keep.

Requires baselines/stockfish and a stockfish binary. See baselines/stockfish/spar.py.
That directory must never be added to a submission zip.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import PLY_CAP
from harness.sandbox import local
from tools.bench import OPENINGS, elo_difference, score_interval

# Roughly a factor of three per rung, which is about 100 to 150 Elo of engine
# strength each step and coarse enough that a 20 game sample can tell them apart.
DEFAULT_RUNGS: tuple[str, ...] = (
    "nodes:100",
    "nodes:300",
    "nodes:1000",
    "nodes:3000",
    "nodes:10000",
    "nodes:30000",
    "nodes:100000",
)

SPARRING_DIR = Path("baselines/stockfish")


class Rung:
    """One level, and the result of playing it."""

    def __init__(self, spec: str) -> None:
        self.spec = spec
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.terminations: dict[str, int] = {}

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return score_interval(self.wins, self.draws, self.losses)[0]

    def strength(self) -> float | None:
        """The number the rung is scaled by, for interpolation. None if unscaled."""
        _, _, raw = self.spec.partition(":")
        try:
            return float(raw)
        except ValueError:
            return None


def play_rung(
    rung: Rung, agent: Path, opponent: Path, games: int, base_ms: int, increment_ms: int,
    ply_cap: int, quiet: bool,
) -> None:
    # The harness spawns the agent with subprocess.Popen, which inherits this
    # environment, so setting the variable here is how the level reaches it.
    os.environ["SPAR_LEVEL"] = rung.spec
    for game in range(games):
        opening = OPENINGS[(game // 2) % len(OPENINGS)]
        plays_white = game % 2 == 0
        white, black = (agent, opponent) if plays_white else (opponent, agent)
        outcome = play_match(
            local(white), local(black), base_ms, increment_ms,
            ply_cap=ply_cap, start_fen=opening,
        )
        rung.terminations[outcome.termination] = rung.terminations.get(outcome.termination, 0) + 1
        if outcome.result in ("draw", "void"):
            rung.draws += 1
            symbol = "="
        elif (outcome.result == "white") == plays_white:
            rung.wins += 1
            symbol = "+"
        else:
            rung.losses += 1
            symbol = "-"
        if not quiet:
            print(f"  {rung.spec:<16} game {game + 1:>3}/{games} {symbol} {outcome.termination}")


def crossover(rungs: list[Rung]) -> tuple[float, str, str] | None:
    """Interpolate the rung strength at which the agent would score 50%.

    Strength is geometric across the rungs, so interpolate in log space. Returns
    the interpolated value plus the two rungs it sits between.
    """
    played = [r for r in rungs if r.games and r.strength() is not None]
    for lower, upper in zip(played, played[1:]):
        if lower.score >= 0.5 > upper.score:
            low_strength, high_strength = lower.strength(), upper.strength()
            if not low_strength or not high_strength:
                return None
            span = math.log(high_strength) - math.log(low_strength)
            drop = lower.score - upper.score
            if drop <= 0:
                return None
            fraction = (lower.score - 0.5) / drop
            return math.exp(math.log(low_strength) + span * fraction), lower.spec, upper.spec
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=SPARRING_DIR)
    parser.add_argument("--games", type=int, default=20, help="games per rung, kept even")
    parser.add_argument("--base-ms", type=int, default=10_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument(
        "--rungs", default=",".join(DEFAULT_RUNGS),
        help="comma separated SPAR_LEVEL specs, weakest first",
    )
    parser.add_argument(
        "--stop-below", type=float, default=0.05,
        help="skip the remaining rungs once a score falls under this",
    )
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    arguments = parser.parse_args()

    if not (arguments.opponent / "agent.py").exists():
        print(f"{arguments.opponent}/agent.py is missing", file=sys.stderr)
        return 1

    games = arguments.games + (arguments.games % 2)  # even, so colours balance
    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    rungs = [Rung(spec.strip()) for spec in arguments.rungs.split(",") if spec.strip()]
    started = time.monotonic()

    for index, rung in enumerate(rungs):
        play_rung(
            rung, agent, opponent, games, arguments.base_ms, arguments.increment_ms,
            arguments.ply_cap, arguments.quiet,
        )
        if not arguments.quiet:
            print()
        if rung.score < arguments.stop_below and index + 1 < len(rungs):
            print(
                f"  scored {rung.score:.1%} at {rung.spec}; skipping the "
                f"{len(rungs) - index - 1} stronger rung(s)\n"
            )
            break

    print(f"{arguments.agent} against Stockfish")
    print(f"  {games} games per rung at {arguments.base_ms}ms + {arguments.increment_ms}ms")
    print()
    print(f"  {'rung':<16} {'W':>3} {'D':>3} {'L':>3} {'score':>7}   {'95% interval':<18} elo")
    for rung in rungs:
        if not rung.games:
            continue
        score, low, high = score_interval(rung.wins, rung.draws, rung.losses)
        elo = elo_difference(score)
        shown = "  -inf" if math.isinf(elo) and elo < 0 else (
            "  +inf" if math.isinf(elo) else f"{elo:+6.0f}"
        )
        print(
            f"  {rung.spec:<16} {rung.wins:>3} {rung.draws:>3} {rung.losses:>3} "
            f"{score:>6.1%}   {low:>5.1%} to {high:<7.1%}  {shown}"
        )

    print()
    landed = crossover(rungs)
    if landed is not None:
        value, lower, upper = landed
        unit = lower.split(":")[0]
        print(f"  50% crossover: about {unit} {value:,.0f}, between {lower} and {upper}.")
        print("  That is the number to carry forward. Re-run this after each change.")
    else:
        played = [r for r in rungs if r.games]
        if played and all(r.score >= 0.5 for r in played):
            print("  Over 50% on every rung. Add stronger rungs above the top of this range.")
        elif played and all(r.score < 0.5 for r in played):
            print("  Under 50% on every rung. Add weaker rungs, or lengthen the time control.")
        else:
            print("  No clean crossover. More games per rung would resolve it.")

    broken: dict[str, int] = {}
    for rung in rungs:
        for name, count in rung.terminations.items():
            if name in FAILED_TERMINATIONS:
                broken[name] = broken.get(name, 0) + count
    if broken:
        print()
        print("  FAILED GAMES: " + ", ".join(f"{n} {c}" for n, c in broken.items()))
        print("  Check whose fault they were before trusting any of the above. Games where")
        print("  the sparring engine died are not wins, and games where you flagged are the")
        print("  thing to fix first.")

    print(f"\n  took {(time.monotonic() - started) / 60:.1f} min")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
