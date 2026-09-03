"""Node rate, depth and ordering quality for one build, on a fixed position set.

    python3 -m tools.kernelbench --agent versions/bot4 --think-ms 2000

Metric 3 in the iteration log is nodes per second on a fixed suite. It is only
meaningful if every build is measured on the same positions with the same
budget, so the suite lives here rather than being whatever position was handy.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

SUITE: tuple[tuple[str, str], ...] = (
    ("start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("italian", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 5"),
    ("sicilian", "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ("french", "rnbqkb1r/pp3ppp/4pn2/2pp4/3P1B2/4PN2/PPP2PPP/RN1QKB1R w KQkq - 0 6"),
    ("queens gambit", "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5"),
    ("kings indian", "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 7"),
    ("open middlegame", "r2q1rk1/pp2bppp/2n1bn2/3p4/3P4/2N1BN2/PP2BPPP/R2Q1RK1 w - - 0 12"),
    ("tactical", "r1bq1rk1/pp1nbppp/2p1pn2/3p4/2PP4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9"),
    ("sharp", "r1b1k2r/ppppqppp/2n2n2/2b5/3NP3/2P5/PP3PPP/RNBQKB1R w KQkq - 0 7"),
    ("rook endgame", "8/5pk1/6p1/7p/1R6/5PKP/6P1/3r4 w - - 0 40"),
    ("pawn endgame", "8/5pk1/6p1/7p/7P/5PK1/6P1/8 w - - 0 45"),
    ("queenless", "r4rk1/pp2ppbp/2n3p1/8/3P4/2N1B3/PP3PPP/2R1KB1R w K - 0 14"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("versions/bot4"))
    parser.add_argument("--think-ms", type=int, default=2_000)
    arguments = parser.parse_args()

    sys.path.insert(0, str(arguments.agent.resolve()))
    agent = importlib.import_module("agent")
    agent.DEBUG = False
    if hasattr(agent, "STATS"):
        agent.STATS = True

    # The budget function turns a clock into a per-move allowance, so hand it a
    # clock that yields roughly the requested think time.
    clock = arguments.think_ms * 20

    total_nodes = 0
    total_seconds = 0.0
    depths: list[int] = []
    cutoff_numerator = 0
    cutoff_denominator = 0
    print(f"{arguments.agent}  target {arguments.think_ms}ms/move")
    print(f"  {'position':<18} {'depth':>5} {'nodes':>9} {'nps':>8} {'cutoff':>7}")
    import time as _time

    for name, fen in SUITE:
        agent._board = None
        agent._history_keys = []
        agent._tt.clear()
        if hasattr(agent, "_tt_old"):
            agent._tt_old.clear()
        started = _time.monotonic()
        agent.get_move(fen, clock)
        elapsed = _time.monotonic() - started
        nodes = agent._nodes
        total_nodes += nodes
        total_seconds += elapsed
        nps = nodes / max(elapsed, 1e-9)
        cutoff = ""
        if hasattr(agent, "_cutoffs") and agent._cutoffs:
            rate = agent._first_move_cutoffs / agent._cutoffs
            cutoff_numerator += agent._first_move_cutoffs
            cutoff_denominator += agent._cutoffs
            cutoff = f"{rate:.1%}"
        print(f"  {name:<18} {'':>5} {nodes:>9,} {nps:>8,.0f} {cutoff:>7}")

    print()
    print(f"  overall {total_nodes:,} nodes in {total_seconds:.1f}s "
          f"= {total_nodes / max(total_seconds, 1e-9):,.0f} nps")
    if cutoff_denominator:
        print(f"  first-move cutoff rate {cutoff_numerator / cutoff_denominator:.1%} "
              f"over {cutoff_denominator:,} cutoffs")
    _ = depths
    return 0


if __name__ == "__main__":
    sys.exit(main())
