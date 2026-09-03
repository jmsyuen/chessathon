"""Head-to-head between two builds, checkpointed after every game, with SPRT.

tools/bench.py plays a fixed number of games in one process and prints at the
end. On a one-core box a hundred games is over an hour, and anything that
interrupts the run loses all of it. This does the same job but appends each
result to a JSON file, so a run can be resumed, extended, or read while it is
still going.

    python3 -m tools.h2h --agent versions/bot4 --opponent baselines/bot1fix \
        --games 8 --base-ms 8000 --increment-ms 500 --out /tmp/bot4_vs_bot1fix.json
    python3 -m tools.h2h --report --out /tmp/bot4_vs_bot1fix.json

The verdict is a sequential probability ratio test with elo0=0, elo1=5 and
alpha=beta=0.05, which is the merge gate the iteration log commits to. SPRT
stops as soon as the evidence is decisive instead of at a game count chosen in
advance, which on one core is the difference between a verdict and a guess.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

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


def _elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def log_likelihood_ratio(wins: int, draws: int, losses: int) -> float:
    """Normalised-score LLR, the standard 5-3-1 pentanomial-free version.

    Uses the score mean and variance directly, which is accurate enough at these
    game counts and avoids needing paired results.
    """
    games = wins + draws + losses
    if games == 0:
        return 0.0
    score = (wins + draws * 0.5) / games
    if score in (0.0, 1.0):
        score = min(max(score, 1e-6), 1 - 1e-6)
    variance = (
        wins * (1.0 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * (0.0 - score) ** 2
    ) / games
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


def load(path: Path) -> dict[str, object]:
    if path.exists():
        return json.loads(path.read_text())
    return {"games": []}


def report(path: Path) -> int:
    state = load(path)
    games = state.get("games", [])
    assert isinstance(games, list)
    wins = sum(1 for g in games if g["outcome"] == "win")
    draws = sum(1 for g in games if g["outcome"] == "draw")
    losses = sum(1 for g in games if g["outcome"] == "loss")
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
    print("  terminations: " + ", ".join(f"{n} {c}" for n, c in sorted(terminations.items())))

    broken = {n: c for n, c in terminations.items() if n in FAILED_TERMINATIONS}
    if broken:
        print()
        print("  FAILED GAMES: " + ", ".join(f"{n} {c}" for n, c in broken.items()))
        print("  The zero-failure gate comes first. Fix these before reading anything above.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("versions/bot4"))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/bot1fix"))
    parser.add_argument("--games", type=int, default=8, help="games to add in this call")
    parser.add_argument("--base-ms", type=int, default=8_000)
    parser.add_argument("--increment-ms", type=int, default=500)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--out", type=Path, default=Path("/tmp/h2h.json"))
    parser.add_argument("--report", action="store_true", help="print the summary and exit")
    parser.add_argument("--deadline-s", type=float, default=250.0, help="stop starting games")
    arguments = parser.parse_args()

    if arguments.report:
        return report(arguments.out)

    state = load(arguments.out)
    state["agent"] = str(arguments.agent)
    state["opponent"] = str(arguments.opponent)
    state["base_ms"] = arguments.base_ms
    state["increment_ms"] = arguments.increment_ms
    games = state["games"]
    assert isinstance(games, list)

    agent_path = arguments.agent.resolve()
    opponent_path = arguments.opponent.resolve()
    started = time.monotonic()

    for _ in range(arguments.games):
        if time.monotonic() - started > arguments.deadline_s:
            print("deadline reached, stopping cleanly")
            break
        index = len(games)
        # Each opening is played from both sides before moving on, so an opening
        # that happens to favour White cannot bias the result.
        opening = OPENINGS[(index // 2) % len(OPENINGS)]
        plays_white = index % 2 == 0
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
                "colour": "white" if plays_white else "black",
                "outcome": result,
                "termination": outcome.termination,
                "seconds": round(time.monotonic() - game_started, 1),
                "opening": opening,
            }
        )
        # Checkpoint after every game: background processes are reaped between
        # tool calls, so a run that only writes at the end writes nothing.
        arguments.out.write_text(json.dumps(state))
        symbol = {"win": "+", "draw": "=", "loss": "-"}[result]
        print(
            f"game {index + 1:>3} as {'white' if plays_white else 'black':<5} {symbol} "
            f"{outcome.termination:<22} {games[-1]['seconds']:>5.1f}s"
        )

    print()
    return report(arguments.out)


if __name__ == "__main__":
    sys.exit(main())
