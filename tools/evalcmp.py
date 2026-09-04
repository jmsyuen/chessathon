"""Which evaluation is more accurate, and where does each one fail?

    python3 -m tools.evalcmp --from-openings --max-imbalance 120 --positions 2000

This is the hard gate for the NNUE line, and it exists because validation loss
cannot see the failure that matters. Across the change that demonstrably added
the missing positional signal, val loss moved the *wrong way*: 0.00526 for the
blind network against 0.00537 for the one with signal. A network can score
r=0.95 against engine labels overall, beat the hand-written evaluation on mean
absolute error, and still correlate at r~=0 on level positions — because in a
lopsided training distribution, counting material explains nearly all the
variance. That network lost 280 Elo. That is regression bug #8.

So the number to read is **r on quiet, level positions**, and nothing else.
bot1's piece-square evaluation manages 0.336 there. A learned evaluation that
cannot beat 0.40 should not be played, because bot4's search *amplifies*
evaluation error: reverse futility, futility, late move pruning and the
improving flag all key off the static score, so an evaluation with no signal in
level positions does not merely choose badly, it prunes badly.

Exits non-zero when the gate fails, so a run plan can stop on it.

Replaces `bots/bot3_nnue/v3_evalcmp.py`, which hardcoded the two agents as "."
and "versions/bot1" and drew openings from `tools/bench.py`'s twelve rather than
the generated set.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path

import chess
import numpy as np

from tools.gendata import SF, Engine, king_danger, material_balance

# The gate is relative, not absolute. An earlier version required r > 0.40 on
# quiet level positions, a threshold taken from bot1's recorded 0.336. But the
# reference is measured on the same positions in the same run, so there is no
# reason to compare against a remembered number from a different sample — and a
# fixed 0.40 is a bar a network can clear while still being worse than the
# evaluation it replaces. Beat the reference, on this run, or do not play.
GATE_MARGIN = 0.0  # how much r the subject must beat the reference by


def load_agent(directory: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, directory / "agent.py")
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import an agent from {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def summarise(name: str, predicted: np.ndarray, labels: np.ndarray) -> float:
    if len(labels) < 8:
        print(f"  {name:<24} too few positions ({len(labels)})")
        return float("nan")
    error = np.abs(predicted - labels)
    correlation = float(np.corrcoef(predicted, labels)[0, 1])
    slope = float(np.polyfit(labels, predicted, 1)[0])
    sign = float(np.mean(np.sign(predicted) == np.sign(labels)))
    print(
        f"  {name:<24} n={len(labels):<5} mae={error.mean():6.0f}  r={correlation:6.3f}  "
        f"slope={slope:5.2f}  sign={sign:5.1%}"
    )
    return correlation


def label(engine: Engine, fen: str, depth: int) -> int:
    """Relabel one position at a proper depth; the play-out score was cheap."""
    engine._send(f"position fen {fen}")
    engine._send(f"go depth {depth}")
    assert engine.process.stdout is not None
    score = 0
    while True:
        line = engine.process.stdout.readline()
        if not line:
            break
        if line.startswith("info ") and " score " in line:
            parts = line.split()
            index = parts.index("score")
            value = int(parts[index + 2])
            score = value if parts[index + 1] == "cp" else (3000 if value > 0 else -3000)
        elif line.startswith("bestmove"):
            break
    return max(-2500, min(2500, score))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("versions/bot5"))
    parser.add_argument("--reference", type=Path, default=Path("versions/bot4"))
    parser.add_argument("--positions", type=int, default=2000)
    parser.add_argument("--label-depth", type=int, default=10)
    parser.add_argument("--play-nodes", type=int, default=900)
    parser.add_argument("--max-imbalance", type=int, default=100_000)
    parser.add_argument("--from-openings", action="store_true",
                        help="seed from the near-level set instead of random plies")
    parser.add_argument("--net-weight", type=int, default=256,
                        help="which arm of the agent to measure")
    parser.add_argument("--gate-margin", type=float, default=GATE_MARGIN,
                        help="r the subject must beat the reference by on quiet level")
    parser.add_argument("--engine", type=str, default=SF)
    parser.add_argument("--seed", type=int, default=999)
    arguments = parser.parse_args()

    os.environ["CHESSATHON_NET_WEIGHT"] = str(arguments.net_weight)
    subject = load_agent(arguments.agent, "_evalcmp_subject")
    reference = load_agent(arguments.reference, "_evalcmp_reference")
    has_net = getattr(subject, "_forward", None) is not None
    print(f"{arguments.agent}: network loaded {has_net}, arm {arguments.net_weight}/256")
    if not has_net:
        print("the subject is running its fallback evaluation; there is nothing to gate")

    engine = Engine(arguments.engine)
    rng = random.Random(arguments.seed)
    rows: list[tuple[chess.Board, bool, bool, int]] = []

    from tools.openings import OPENINGS

    while len(rows) < arguments.positions:
        if arguments.from_openings:
            board = chess.Board(rng.choice(OPENINGS))
        else:
            board = chess.Board()
            for _ in range(rng.choice((4, 6, 8))):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
        temperature = 0.03 if arguments.from_openings else 0.10
        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 200:
            best, _score = engine.search(board.fen(), arguments.play_nodes)
            if best == "(none)":
                break
            move = chess.Move.from_uci(best)
            if move not in board.legal_moves:
                break
            if abs(material_balance(board)) <= arguments.max_imbalance:
                rows.append(
                    (board.copy(stack=False), board.is_check(), board.is_capture(move),
                     king_danger(board))
                )
            if len(rows) >= arguments.positions:
                break
            if rng.random() < temperature:
                move = rng.choice(list(board.legal_moves))
            board.push(move)
        if len(rows) % 200 < 2:
            print(f"  collected {len(rows)}/{arguments.positions}", flush=True)

    labels = np.array([label(engine, board.fen(), arguments.label_depth)
                       for board, *_ in rows], dtype=np.float64)
    engine.close()

    boards = [row[0] for row in rows]
    in_check = np.array([row[1] for row in rows])
    capture_best = np.array([row[2] for row in rows])
    danger = np.array([row[3] for row in rows])
    material = np.array([material_balance(board) for board in boards], dtype=np.float64)

    subject_out = np.array([subject.evaluate(board) for board in boards], dtype=np.float64)
    reference_out = np.array([reference.evaluate(board) for board in boards], dtype=np.float64)

    quiet = ~in_check & ~capture_best
    level = np.abs(material) <= 60
    exposed = danger >= 6

    print("\nmae in centipawns against Stockfish; slope 1.00 means the same scale")
    subject_r = reference_r = float("nan")
    for name, mask in (
        ("all positions", np.ones(len(rows), dtype=bool)),
        ("quiet", quiet),
        ("quiet + level material", quiet & level),
        ("quiet + king exposed", quiet & exposed),
        ("in check or capture best", in_check | capture_best),
    ):
        print(f" {name}")
        value = (
            summarise(f"{arguments.agent.name}", subject_out[mask], labels[mask]),
            summarise(f"{arguments.reference.name}", reference_out[mask], labels[mask]),
        )
        if name == "quiet + level material":
            subject_r, reference_r = value

    print()
    if not (np.isfinite(subject_r) and np.isfinite(reference_r)):
        print("GATE INCONCLUSIVE: too few quiet level positions. Raise --positions.")
        return 1
    required = reference_r + arguments.gate_margin
    verdict = (
        f"quiet level r: {arguments.agent.name} {subject_r:.3f} against "
        f"{arguments.reference.name} {reference_r:.3f}"
    )
    if subject_r > required:
        print(f"GATE PASS: {verdict}")
        return 0
    print(
        f"GATE FAIL: {verdict}. The network is not a better evaluator than the one it\n"
        "replaces, on the positions the search actually prunes in. This is bug #8.\n"
        "Ship NET_WEIGHT = 0 and fix the data before spending games on it."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
