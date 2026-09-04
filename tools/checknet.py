"""Does the shipped kernel agree with the model that was trained?

    python3 -m tools.checknet --state data/state.npz --data data/train.csv

Quantisation is where a good network silently becomes a bad one. A scale error
does not crash; it just makes the search thrash, and the only way to see it is
to compare the int16 kernel against the float checkpoint it came from.

Two corrections against `bots/bot3_nnue/v3_checknet.py`, which this replaces:

* Paths are arguments. The original hardcoded `/home/claude/data/...`, which is
  a sandbox path and does not exist on the machine that now runs the training.
* The float reference adds the material skip connection. The agent's kernel
  computes material inside the forward pass and adds it to the output, and the
  trainer adds it as a fixed offset, but the original float reference did
  neither — so it was comparing a score with material against a score without,
  and the "quantisation error" it reported was mostly the material balance.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import chess
import numpy as np

from tools.train import encode, material_of

SCALE = 400.0

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


def load_agent(directory: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, directory / "agent.py")
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import an agent from {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def float_eval(state: dict[str, np.ndarray], fen: str) -> float:
    """The float model's score in centipawns, including the material skip."""
    fields = fen.split(" ")
    white_to_move = fields[1] == "w"
    us, them = encode(fields[0], white_to_move)
    w0, b0, w1, b1 = state["w0"], state["b0"], state["w1"], state["b1"]
    hidden = w0.shape[1]
    clipped_us = np.clip(b0 + w0[us].sum(axis=0), 0.0, 1.0) ** 2
    clipped_them = np.clip(b0 + w0[them].sum(axis=0), 0.0, 1.0) ** 2
    out = clipped_us @ w1[:hidden] + clipped_them @ w1[hidden:] + b1[0]
    return float(out) * SCALE + material_of(fields[0], white_to_move)


def raw_kernel(agent, board: chess.Board) -> float:  # type: ignore[no-untyped-def]
    """The kernel's own output, before the agent's centipawn calibration."""
    return float(
        agent._forward(
            agent._pack(board, agent._BITBOARDS), int(board.turn),
            agent._NET_W0, agent._NET_B0, agent._NET_W1, agent._NET_B1,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("versions/bot5"))
    parser.add_argument("--state", type=Path, default=Path("data/state.npz"),
                        help="float checkpoint written by tools/train.py --state")
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--sample", type=int, default=3000)
    arguments = parser.parse_args()

    agent = load_agent(arguments.agent, "_checknet_agent")
    if agent._forward is None:
        print(f"FAIL: {arguments.agent} did not load a network")
        return 1
    print(f"hidden {agent._HIDDEN}, cp_scale {agent._CP_SCALE}, weights {agent.WEIGHTS_PATH}")

    state = (
        dict(np.load(arguments.state).items()) if arguments.state.exists() else {}
    )
    if not state:
        print(f"no float checkpoint at {arguments.state}; reference values only\n")

    print(f"{'position':<22} {'kernel':>10} {'float':>10} {'diff':>7}")
    for name, fen in REFERENCE:
        board = chess.Board(fen)
        kernel = raw_kernel(agent, board)
        if state:
            reference = float_eval(state, fen)
            print(f"{name:<22} {kernel:>10.0f} {reference:>10.0f} {kernel - reference:>7.0f}")
        else:
            print(f"{name:<22} {kernel:>10.0f}")

    if not state or not arguments.data.exists():
        return 0

    # The real test: agreement over a sample of the positions it was trained on.
    # FENs come straight from the csv so nothing can drift out of alignment.
    with arguments.data.open() as handle:
        lines = handle.readlines()
    rng = np.random.default_rng(0)
    size = min(arguments.sample, len(lines))
    sample = rng.choice(len(lines), size=size, replace=False)

    float_out = np.empty(size)
    kernel_out = np.empty(size)
    labels = np.empty(size)
    for position, index in enumerate(sample):
        fen, score_text, _ = lines[int(index)].rsplit(",", 2)
        float_out[position] = float_eval(state, fen)
        kernel_out[position] = raw_kernel(agent, chess.Board(fen))
        labels[position] = float(score_text)

    difference = kernel_out - float_out
    print(f"\nquantisation error over {size:,} positions:")
    print(f"  mean abs {np.abs(difference).mean():.1f} cp, max {np.abs(difference).max():.0f} cp")
    print("\nagreement with the Stockfish labels it was trained against:")
    print(f"  mean abs error {np.abs(kernel_out - labels).mean():.0f} cp")
    print(f"  correlation    {np.corrcoef(kernel_out, labels)[0, 1]:.3f}")
    if np.abs(difference).mean() > 5.0:
        print("\nFAIL: quantisation error above 5 cp. QA=1024/QB=2048 measured 0.8 cp mean.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
