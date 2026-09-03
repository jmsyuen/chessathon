"""Correctness and safety checks for agent.py.

Run from the repo root:  python3 -m tools.selftest

These are the checks that catch free losses: illegal moves, crashes on odd
positions, and blowing the clock. They run in well under a minute, so run them
before every upload. Strength is measured separately, by tools/bench.py.
"""

from __future__ import annotations

import sys
import time

import chess

import agent

# --------------------------------------------------------------------------
# Tactics. Each entry is (name, fen, set of acceptable UCI moves, think_ms).
# --------------------------------------------------------------------------

TACTICS: list[tuple[str, str, set[str], int]] = [
    (
        "mate in 1, back rank",
        "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        {"a1a8"},
        3_000,
    ),
    (
        "mate in 1, scholar's",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        {"f3f7"},
        3_000,
    ),
    (
        "mate in 2, smothered",
        "6rk/6pp/8/6N1/8/8/8/6QK w - - 0 1",
        {"g1a1", "g1b1", "g1c1", "g1d1", "g1e1", "g1f1", "g5f7", "g1g7"},
        6_000,
    ),
    (
        "hangs nothing, must recapture",
        "rnbqkb1r/ppp1pppp/5n2/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3",
        {"f6d5", "d8d5"},
        4_000,
    ),
    (
        "takes a free queen on the file",
        "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",
        {"d1d5"},
        3_000,
    ),
    (
        "takes a free queen next door",
        "6k1/8/8/8/8/8/1q6/1R2K3 w - - 0 1",
        {"b1b2"},
        3_000,
    ),
    (
        "mate in 3, white",  # only f6a6 forces it; verified by brute force
        "r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1",
        {"f6a6"},
        12_000,
    ),
    (
        "mate in 3, black",  # only d6d1 forces it; verified by brute force
        "1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1",
        {"d6d1"},
        12_000,
    ),
    (
        "promotes",
        "8/P6k/8/8/8/8/6K1/8 w - - 0 1",
        {"a7a8q"},
        3_000,
    ),
    (
        "takes the draw when lost",
        "7k/6p1/8/8/8/8/1q6/K7 w - - 0 1",
        set(),  # any legal move; this only has to not crash
        3_000,
    ),
]

# --------------------------------------------------------------------------
# Edge cases that must not crash and must return something legal.
# --------------------------------------------------------------------------

EDGE_CASES: list[tuple[str, str, int]] = [
    ("start position", chess.STARTING_FEN, 5_000),
    ("only one legal move", "7k/8/8/8/8/8/5PPP/6rK w - - 0 1", 5_000),
    (
        "en passant available",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        3_000,
    ),
    ("black to move, promotion race", "8/1P6/8/8/8/8/6p1/K6k b - - 0 1", 3_000),
    ("bare kings plus pawn", "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1", 3_000),
    ("fifty-move clock nearly up", "8/8/4k3/8/8/4K3/8/6R1 w - - 98 120", 3_000),
    ("castling rights all round", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", 3_000),
    ("very low clock", chess.STARTING_FEN, 120),
    ("absurdly low clock", chess.STARTING_FEN, 5),
    ("zero clock", chess.STARTING_FEN, 0),
    ("negative clock", chess.STARTING_FEN, -100),
]


def _reset() -> None:
    """Forget the tracked game, as if a fresh process had started."""
    agent._board = None
    agent._history_keys = []
    agent._tt.clear()


def _check_legal(fen: str, uci: str) -> bool:
    board = chess.Board(fen)
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return False
    return move in board.legal_moves


def run_tactics() -> list[str]:
    failures: list[str] = []
    for name, fen, expected, think_ms in TACTICS:
        _reset()
        played = agent.get_move(fen, think_ms)
        if not _check_legal(fen, played):
            failures.append(f"tactics/{name}: illegal move {played!r}")
            continue
        if expected and played not in expected:
            failures.append(f"tactics/{name}: played {played}, wanted one of {sorted(expected)}")
        else:
            print(f"  ok   tactics/{name}: {played}")
    return failures


def run_edge_cases() -> list[str]:
    failures: list[str] = []
    for name, fen, think_ms in EDGE_CASES:
        _reset()
        try:
            played = agent.get_move(fen, think_ms)
        except Exception as error:
            failures.append(f"edge/{name}: raised {type(error).__name__}: {error}")
            continue
        if not _check_legal(fen, played):
            failures.append(f"edge/{name}: illegal move {played!r}")
        else:
            print(f"  ok   edge/{name}: {played}")
    return failures


def run_malformed_input() -> list[str]:
    """Nothing the platform sends should ever get here, but a crash is a loss."""
    failures: list[str] = []
    bad_inputs = ["", "not a fen", "8/8/8/8/8/8/8/8 w - - 0 1", chess.STARTING_FEN + " extra"]
    for fen in bad_inputs:
        _reset()
        try:
            result = agent.get_move(fen, 1_000)
        except Exception as error:
            failures.append(f"malformed/{fen!r}: raised {type(error).__name__}: {error}")
            continue
        if not isinstance(result, str):
            failures.append(f"malformed/{fen!r}: returned {type(result).__name__}, not str")
        else:
            print(f"  ok   malformed/{fen[:24]!r}: {result}")
    return failures


def run_clock_discipline() -> list[str]:
    """Play a full game against itself and assert we never overspend.

    The referee deducts wall-clock time around the whole request and flags the
    instant the clock goes negative, so the check here is the real one.
    """
    failures: list[str] = []
    _reset()
    board = chess.Board()
    clock = {chess.WHITE: float(agent.INCREMENT_MS * 0 + 120_000), chess.BLACK: 120_000.0}
    worst_ratio = 0.0
    worst_move = ""
    plies = 0

    while plies < 120:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            break
        mover = board.turn
        # One tracked agent cannot play both sides, so resync each ply. This is
        # the pessimistic case: no carried state, every move searched cold.
        _reset()
        started = time.monotonic()
        uci = agent.get_move(board.fen(), int(clock[mover]))
        elapsed_ms = (time.monotonic() - started) * 1000.0
        clock[mover] -= elapsed_ms
        if clock[mover] < 0:
            failures.append(f"clock: flagged at ply {plies} after {elapsed_ms:.0f}ms")
            break
        _, hard = agent._budget(int(clock[mover] + elapsed_ms), plies)
        if hard > 0:
            ratio = elapsed_ms / hard
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_move = f"ply {plies}, {elapsed_ms:.0f}ms against a {hard:.0f}ms limit"
            if elapsed_ms > hard + agent.OVERHEAD_MS:
                failures.append(
                    f"clock: ply {plies} took {elapsed_ms:.0f}ms, hard limit was {hard:.0f}ms"
                )
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            failures.append(f"clock: illegal move {uci} at ply {plies}")
            break
        board.push(move)
        clock[mover] += agent.INCREMENT_MS
        plies += 1

    print(f"  ok   clock: {plies} plies, worst {worst_move or 'n/a'}")
    white_left = clock[chess.WHITE] / 1000
    black_left = clock[chess.BLACK] / 1000
    print(f"       white {white_left:.1f}s left, black {black_left:.1f}s left")
    return failures


def run_repetition_tracking() -> list[str]:
    """The agent only ever sees a FEN, so it has to rebuild history itself.

    Walk a shuffling line and confirm the tracked history grows one entry per
    ply instead of resetting, which is what would happen if the opponent's move
    could not be reconstructed.
    """
    failures: list[str] = []
    _reset()
    board = chess.Board()
    rounds = 6
    for index in range(rounds):
        uci = agent.get_move(board.fen(), 400)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            failures.append(f"repetition: illegal move {uci} at round {index}")
            break
        board.push(move)
        replies = list(board.legal_moves)
        if not replies:
            break
        board.push(replies[0])  # a scripted opponent, so the chain must reconstruct

        # Two positions land in history per round: the one we were handed and the
        # one after our own move. Anything less means _sync failed to reconstruct
        # the opponent's move and fell back to a resync.
        expected = 2 * (index + 1)
        if len(agent._history_keys) != expected:
            failures.append(
                f"repetition: after round {index} the history had "
                f"{len(agent._history_keys)} entries, expected {expected}"
            )
            break

    tracked = len(agent._history_keys)
    print(f"  ok   repetition: reconstructed {tracked} positions from FENs alone")

    # Now the real test: from a position where we are winning easily, the agent
    # should not walk into a repetition it can avoid.
    _reset()
    winning = "6k1/5ppp/8/8/8/8/5PPP/1R4K1 w - - 0 1"
    seen: list[str] = []
    board = chess.Board(winning)
    for _ in range(6):
        uci = agent.get_move(board.fen(), 600)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            failures.append(f"repetition: illegal move {uci}")
            break
        seen.append(uci)
        board.push(move)
        replies = list(board.legal_moves)
        if not replies:
            break
        board.push(replies[0])
    if len(set(seen)) <= 1 and len(seen) > 2:
        failures.append(f"repetition: agent repeated the same move throughout: {seen}")
    print(f"  ok   repetition: winning line played {seen}")
    return failures


CONVERSIONS: list[tuple[str, str]] = [
    ("K+R vs K", "8/8/8/4k3/8/8/8/R3K3 w - - 0 1"),
    ("K+Q vs K", "8/8/8/4k3/8/8/8/3QK3 w - - 0 1"),
    ("K+2B vs K", "8/8/8/4k3/8/8/8/2B1KB2 w - - 0 1"),
    ("K+R+B vs K", "8/8/4k3/8/8/8/8/R1B1K3 w - - 0 1"),
]

CONVERSION_PLY_LIMIT = 70  # comfortably inside the fifty-move rule


def run_conversions() -> list[str]:
    """Basic mates must actually get mated.

    Material and piece-square tables alone will shuffle a won K+R ending until
    the referee claims a threefold or fifty-move draw. This is the check that the
    mate-drive term is doing its job.
    """
    failures: list[str] = []
    for name, fen in CONVERSIONS:
        _reset()
        board = chess.Board(fen)
        plies = 0
        result = "no mate"
        while plies < CONVERSION_PLY_LIMIT:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                result = outcome.termination.name.lower()
                break
            if board.turn == chess.WHITE:
                move = chess.Move.from_uci(agent.get_move(board.fen(), 800))
                if move not in board.legal_moves:
                    failures.append(f"conversion/{name}: illegal move {move}")
                    break
            else:
                # The bare king runs: pick the move furthest from the enemy king.
                strong = board.king(chess.WHITE)
                assert strong is not None
                move = max(
                    board.legal_moves,
                    key=lambda candidate: max(
                        abs((candidate.to_square & 7) - (strong & 7)),
                        abs((candidate.to_square >> 3) - (strong >> 3)),
                    ),
                )
            board.push(move)
            plies += 1
        if result != "checkmate":
            failures.append(f"conversion/{name}: {result} after {plies} plies")
        else:
            print(f"  ok   conversion/{name}: mate in {plies} plies")
    return failures


def main() -> int:
    print("tactics")
    failures = run_tactics()
    print("edge cases")
    failures += run_edge_cases()
    print("malformed input")
    failures += run_malformed_input()
    print("endgame conversion")
    failures += run_conversions()
    print("repetition tracking")
    failures += run_repetition_tracking()
    print("clock discipline")
    failures += run_clock_discipline()

    print()
    if failures:
        print(f"{len(failures)} failure(s):")
        for failure in failures:
            print(f"  FAIL {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
