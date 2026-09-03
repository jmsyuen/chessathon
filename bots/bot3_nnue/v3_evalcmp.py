"""Which evaluation is actually more accurate, and where does each one fail?

bot2 lost decisively to bot1 while having a faster search and the same depth,
so the evaluation is the suspect. This scores both against fresh Stockfish
labels on positions drawn the way a real game draws them, not the way the
training filter drew them.

    python3 -m tools.evalcmp --positions 240

The split that matters: the training filter kept only positions that were not in
check and where Stockfish's best move was not a capture. That is a sensible
filter for a static evaluation, but it also throws away most positions where a
king is in danger, so the net may never have been taught king safety.
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from pathlib import Path

import chess
import numpy as np

from tools.gendata import SF, Engine


def load_agent(directory: str, name: str):  # type: ignore[no-untyped-def]
    """Import an agent from a directory under a private module name."""
    saved = list(sys.path)
    sys.path.insert(0, directory)
    for key in list(sys.modules):
        if key == "agent":
            del sys.modules[key]
    module = importlib.import_module("agent")
    sys.modules[name] = module
    del sys.modules["agent"]
    sys.path[:] = saved
    return module


def king_danger(board: chess.Board) -> int:
    """How many enemy pieces attack the three-by-three box around each king."""
    total = 0
    for color in (chess.WHITE, chess.BLACK):
        king = board.king(color)
        if king is None:
            continue
        zone = chess.BB_KING_ATTACKS[king] | chess.BB_SQUARES[king]
        attackers = 0
        bits = zone
        while bits:
            square = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            attackers += chess.popcount(board.attackers_mask(not color, square))
        total = max(total, attackers)
    return total


def summarise(name: str, predicted: np.ndarray, labels: np.ndarray) -> None:
    if len(labels) < 8:
        print(f"  {name:<28} too few positions ({len(labels)})")
        return
    error = np.abs(predicted - labels)
    correlation = np.corrcoef(predicted, labels)[0, 1]
    slope = float(np.polyfit(labels, predicted, 1)[0])
    sign = float(np.mean(np.sign(predicted) == np.sign(labels)))
    print(
        f"  {name:<28} n={len(labels):<5} mae={error.mean():6.0f}  r={correlation:5.3f}  "
        f"slope={slope:4.2f}  sign={sign:5.1%}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=240)
    parser.add_argument("--label-depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--from-openings", action="store_true",
                        help="seed from near-level openings instead of random plies")
    parser.add_argument("--max-imbalance", type=int, default=100_000)
    arguments = parser.parse_args()

    bot2 = load_agent(".", "bot2")
    bot1 = load_agent("versions/bot1", "bot1")
    print(f"bot2 network loaded: {bot2._forward is not None}")

    engine = Engine(SF)
    rng = random.Random(arguments.seed)
    rows: list[tuple[chess.Board, int, bool, bool, int]] = []

    while len(rows) < arguments.positions:
        if arguments.from_openings:
            from tools.bench import OPENINGS
            board = chess.Board(rng.choice(OPENINGS))
        else:
            board = chess.Board()
            for _ in range(rng.choice((4, 6, 8))):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 200:
            best, score = engine.search(board.fen(), 900)
            if best == "(none)":
                break
            move = chess.Move.from_uci(best)
            if move not in board.legal_moves:
                break
            keep = abs(bot1._material_balance(board)) <= arguments.max_imbalance
            if keep:
                rows.append(
                    (
                        board.copy(stack=False),
                        score,
                        board.is_check(),
                        board.is_capture(move),
                        king_danger(board),
                    )
                )
            if len(rows) >= arguments.positions:
                break
            if rng.random() < (0.10 if not arguments.from_openings else 0.03):
                move = rng.choice(list(board.legal_moves))
            board.push(move)

    # Relabel every kept position at a proper depth; the play-out score was cheap.
    labels = np.zeros(len(rows), dtype=np.float64)
    for index, (board, *_rest) in enumerate(rows):
        engine._send(f"position fen {board.fen()}")
        engine._send(f"go depth {arguments.label_depth}")
        assert engine.process.stdout is not None
        score = 0
        while True:
            line = engine.process.stdout.readline()
            if line.startswith("info ") and " score " in line:
                parts = line.split()
                position = parts.index("score")
                value = int(parts[position + 2])
                score = value if parts[position + 1] == "cp" else (3000 if value > 0 else -3000)
            elif line.startswith("bestmove"):
                break
        labels[index] = max(-2500, min(2500, score))
    engine.close()

    boards = [row[0] for row in rows]
    in_check = np.array([row[2] for row in rows])
    capture_best = np.array([row[3] for row in rows])
    danger = np.array([row[4] for row in rows])
    material = np.array([bot1._material_balance(board) for board in boards], dtype=np.float64)

    predicted2 = np.array([bot2.evaluate(board) for board in boards], dtype=np.float64)
    predicted1 = np.array([bot1.evaluate(board) for board in boards], dtype=np.float64)

    quiet = ~in_check & ~capture_best
    level = np.abs(material) <= 60
    exposed = danger >= 6

    print()
    print("mae in centipawns against Stockfish; slope 1.00 means the same scale")
    for name, mask in (
        ("all positions", np.ones(len(rows), dtype=bool)),
        ("quiet (the training filter)", quiet),
        ("quiet + level material", quiet & level),
        ("quiet + king exposed", quiet & exposed),
        ("in check or capture best", in_check | capture_best),
    ):
        print(f" {name}")
        summarise("bot2 net", predicted2[mask], labels[mask])
        summarise("bot1 piece-square", predicted1[mask], labels[mask])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
