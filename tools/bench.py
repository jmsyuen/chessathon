"""Measure one agent against another and say whether the difference is real.

    python3 -m tools.bench --opponent baselines/minimax --games 40
    python3 -m tools.bench --opponent versions/v1 --games 200 --base-ms 20000

Why this exists rather than harness/arena.py:

  * Rated games start from curated near-level positions, not the standard start,
    so measuring only from the start position measures the wrong thing.
  * Colours alternate over the same opening, so an opening that happens to favour
    one side cannot bias the result.
  * It reports an Elo estimate with a confidence interval. A 3% strength gain is
    invisible in 20 games and it is very easy to convince yourself otherwise.

Keep this file inside tools/. harness/package.py zips every *.py at the repo
root, so a benchmark script left at the root would ship inside your submission.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import PLY_CAP
from harness.sandbox import local

# Quiet, roughly balanced positions a few moves in. This is a stand-in for the
# organisers' curated set, which is not published. Replace or extend it freely;
# the only thing that matters is that the positions are near level and varied.
OPENINGS: tuple[str, ...] = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
    "rnbqkb1r/pp2pppp/3p1n2/2pP4/8/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 5",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 1 3",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
    "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 5",
    "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 1 3",
    "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 1 3",
)


def elo_difference(score: float) -> float:
    """Convert a score rate into an Elo difference. Infinite at 0% and 100%."""
    if score <= 0.0:
        return float("-inf")
    if score >= 1.0:
        return float("inf")
    return -400.0 * math.log10(1.0 / score - 1.0)


def score_interval(wins: int, draws: int, losses: int) -> tuple[float, float, float]:
    """Score rate with a 95% interval, from the per-game standard deviation.

    Draws count a half point and contribute no variance of their own, which is
    why draw-heavy matchups need fewer games to resolve than they look like they
    should.
    """
    games = wins + draws + losses
    if games == 0:
        return 0.0, 0.0, 0.0
    score = (wins + draws / 2.0) / games
    # Variance of a single game's score around the mean.
    variance = (
        wins * (1.0 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * (0.0 - score) ** 2
    ) / games
    error = 1.96 * math.sqrt(variance / games)
    return score, max(0.0, score - error), min(1.0, score + error)


def games_needed(margin: float = 0.05) -> int:
    """Rough number of games to resolve a score difference of this size."""
    return math.ceil((1.96 * 0.4 / margin) ** 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/minimax"))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--base-ms", type=int, default=10_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    arguments = parser.parse_args()

    agent_path = arguments.agent.resolve()
    opponent_path = arguments.opponent.resolve()

    wins = draws = losses = 0
    terminations: dict[str, int] = {}
    started = time.monotonic()

    for game in range(arguments.games):
        # Alternate colours over the same opening, so each position is played
        # from both sides before moving on.
        opening = OPENINGS[(game // 2) % len(OPENINGS)]
        plays_white = game % 2 == 0
        white, black = (
            (agent_path, opponent_path) if plays_white else (opponent_path, agent_path)
        )
        outcome = play_match(
            local(white),
            local(black),
            arguments.base_ms,
            arguments.increment_ms,
            ply_cap=arguments.ply_cap,
            start_fen=opening,
        )
        terminations[outcome.termination] = terminations.get(outcome.termination, 0) + 1
        if outcome.result in ("draw", "void"):
            draws += 1
            symbol = "="
        elif (outcome.result == "white") == plays_white:
            wins += 1
            symbol = "+"
        else:
            losses += 1
            symbol = "-"
        if not arguments.quiet:
            colour = "white" if plays_white else "black"
            print(
                f"game {game + 1:>3}/{arguments.games} as {colour:<5} {symbol} "
                f"{outcome.termination}"
            )

    elapsed = time.monotonic() - started
    score, low, high = score_interval(wins, draws, losses)
    elo = elo_difference(score)
    elo_low = elo_difference(low)
    elo_high = elo_difference(high)

    print()
    print(f"{arguments.agent} vs {arguments.opponent}")
    print(f"  {arguments.games} games at {arguments.base_ms}ms + {arguments.increment_ms}ms")
    print(f"  +{wins} ={draws} -{losses}")
    print(f"  score {score:.1%}  (95% interval {low:.1%} to {high:.1%})")
    if math.isinf(elo):
        print(f"  elo   {'+inf' if elo > 0 else '-inf'} (a clean sweep bounds nothing)")
    else:
        span_low = "-inf" if math.isinf(elo_low) else f"{elo_low:+.0f}"
        span_high = "+inf" if math.isinf(elo_high) else f"{elo_high:+.0f}"
        print(f"  elo   {elo:+.0f}  (95% interval {span_low} to {span_high})")
    print(f"  took  {elapsed / 60:.1f} min")
    print("  terminations: " + ", ".join(f"{n} {c}" for n, c in sorted(terminations.items())))

    broken = {n: c for n, c in terminations.items() if n in FAILED_TERMINATIONS}
    if broken:
        print()
        print("  FAILED GAMES: " + ", ".join(f"{n} {c}" for n, c in broken.items()))
        print("  Fix these before measuring anything else. A crash or a flag is a free loss.")
        return 1

    if low < 0.5 < high:
        print()
        print(
            f"  This does not separate the two agents. Around {games_needed(0.05)} games "
            "are needed to resolve a 5% difference."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
