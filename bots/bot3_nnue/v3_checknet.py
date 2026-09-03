"""Does the shipped kernel agree with the model that was trained?

Quantisation is where a good net silently becomes a bad one, and a scale error
does not crash, it just makes the search thrash. Run from the repo root:

    python3 -m tools.checknet
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path.cwd()))
import agent  # noqa: E402

from tools.train import PAD, encode  # noqa: E402

SCALE = 400.0


def float_eval(state: dict[str, np.ndarray], fen: str) -> float:
    fields = fen.split(" ")
    us, them = encode(fields[0], fields[1] == "w")
    w0, b0, w1, b1 = state["w0"], state["b0"], state["w1"], state["b1"]
    hidden = w0.shape[1]
    accumulator_us = b0 + w0[us].sum(axis=0)
    accumulator_them = b0 + w0[them].sum(axis=0)
    clipped_us = np.clip(accumulator_us, 0.0, 1.0) ** 2
    clipped_them = np.clip(accumulator_them, 0.0, 1.0) ** 2
    out = clipped_us @ w1[:hidden] + clipped_them @ w1[hidden:] + b1[0]
    return float(out) * SCALE


REFERENCE = [
    ("start position", chess.STARTING_FEN),
    ("white a queen up", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("black a queen up", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"),
    ("white a rook up", "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Kk - 0 1"),
    ("white a knight up", "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("white a pawn up", "rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("K+R vs K", "8/8/8/4k3/8/8/8/R3K3 w - - 0 1"),
    ("king and pawn", "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"),
    ("italian, level", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
]


def main() -> int:
    if agent._forward is None:
        print("FAIL: the agent did not load a network")
        return 1
    state_path = Path("/home/claude/data/state.npz")
    state = {key: value for key, value in np.load(state_path).items()} if state_path.exists() else {}

    print(f"hidden {agent._HIDDEN}, weights {Path('weights/nnue.npz').stat().st_size:,} bytes")
    print()
    print(f"{'position':<22} {'quantised':>10} {'float':>10} {'diff':>7}")
    for name, fen in REFERENCE:
        quantised = agent.evaluate(chess.Board(fen))
        if state:
            reference = float_eval(state, fen)
            print(f"{name:<22} {quantised:>10} {reference:>10.0f} {quantised - reference:>7.0f}")
        else:
            print(f"{name:<22} {quantised:>10}")

    if not state:
        return 0

    # The real test: agreement over a sample of the positions it was trained on.
    # Read FENs straight from the csv so nothing can drift out of alignment.
    with Path("/home/claude/data/train.csv").open() as handle:
        lines = handle.readlines()
    rng = np.random.default_rng(0)
    sample = rng.choice(len(lines), size=3000, replace=False)

    float_out = np.empty(len(sample))
    quantised_out = np.empty(len(sample))
    labels = np.empty(len(sample))
    for position, index in enumerate(sample):
        fen, score_text, _ = lines[int(index)].rsplit(",", 2)
        board = chess.Board(fen)
        float_out[position] = float_eval(state, fen)
        quantised_out[position] = agent._forward(
            agent._pack(board, agent._BITBOARDS),
            int(board.turn),
            agent._NET_W0,
            agent._NET_B0,
            agent._NET_W1,
            agent._NET_B1,
        )
        labels[position] = float(score_text)

    difference = quantised_out - float_out
    print()
    print(f"quantisation error over {len(sample)} positions:")
    print(f"  mean abs {np.abs(difference).mean():.1f} cp, max {np.abs(difference).max():.0f} cp")
    print()
    print("agreement with the Stockfish labels it was trained against:")
    print(f"  mean abs error {np.abs(quantised_out - labels).mean():.0f} cp")
    print(f"  correlation    {np.corrcoef(quantised_out, labels)[0, 1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
