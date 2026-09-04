"""Put the network on the hand-crafted evaluation's centipawn scale.

    python3 -m tools.calibrate --weights weights/nnue.npz --reference versions/bot4

Why this exists
---------------
bot4's pruning margins are all expressed in units of bot4's evaluation: reverse
futility at 80 a ply, futility at 110 + 95 a ply, delta pruning at 190, the
aspiration window at 25, and the null-move reduction's `(static - beta) // 200`.
None of them is a centipawn constant in the abstract — each one is a constant
relative to whatever `evaluate()` reports.

The iteration log records bot1's evaluation as having a slope of 1.60 against
Stockfish, meaning it over-reports by 60%, and the bot3 network's as 1.11. Drop
a 1.11-slope evaluation into a search tuned around a 1.60-slope one and every
margin above is silently about 45% wider in real terms, so the search prunes
harder than it was ever measured doing. That does not crash and it does not look
like a bug; it looks like the evaluation got slightly worse.

So: regress the network's output on the reference evaluation over a sample of
real positions, and store the slope in the weights file as a fixed-point integer
the agent applies on every call. One number, measured rather than assumed.

This runs after training and before any game is played. It writes into the npz
in place, so `weights/nnue.npz` is self-describing and the agent needs no
matching source edit — which is the point, because the agent is frozen before
these weights exist.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import chess
import numpy as np

CP_SCALE_ONE = 1024  # fixed-point 1.0, matching the agent


def load_agent(directory: Path, name: str):  # type: ignore[no-untyped-def]
    """Import an agent from a directory under a private module name."""
    path = directory / "agent.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import an agent from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sample_positions(count: int, seed: int) -> list[chess.Board]:
    """Positions drawn the way a game draws them, from the near-level set."""
    from tools.openings import OPENINGS

    rng = random.Random(seed)
    boards: list[chess.Board] = []
    while len(boards) < count:
        board = chess.Board(rng.choice(OPENINGS))
        for _ in range(rng.randint(0, 60)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))
            # Only quiet positions: the calibration has to describe the number
            # the search actually prunes on, and the search never prunes on a
            # static score while in check.
            if not board.is_check() and len(boards) < count and rng.random() < 0.25:
                boards.append(board.copy(stack=False))
    return boards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/nnue.npz"))
    parser.add_argument("--agent", type=Path, default=Path("versions/bot5"),
                        help="the build whose network is being calibrated")
    parser.add_argument("--reference", type=Path, default=Path("versions/bot4"),
                        help="the build whose scale the margins are tuned to")
    parser.add_argument("--positions", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=31337)
    parser.add_argument("--dry-run", action="store_true", help="report, do not write")
    arguments = parser.parse_args()

    boards = sample_positions(arguments.positions, arguments.seed)
    reference = load_agent(arguments.reference, "_calib_reference")

    import os

    os.environ["CHESSATHON_NET_WEIGHT"] = "256"
    os.environ["CHESSATHON_WEIGHTS"] = str(arguments.weights.resolve())
    subject = load_agent(arguments.agent, "_calib_subject")
    if subject._forward is None:
        raise SystemExit(f"no usable network at {arguments.weights}; nothing to calibrate")

    # Raw network output, before any calibration the file already carries.
    existing = int(subject._CP_SCALE)
    subject._CP_SCALE = CP_SCALE_ONE

    net = np.array([subject._net_evaluate(board) for board in boards], dtype=np.float64)
    hce = np.array([reference.evaluate(board) for board in boards], dtype=np.float64)

    # Slope through the origin: both evaluations are already zero-centred by
    # construction, and forcing an intercept would let a constant offset absorb
    # scale error. Fit is on the reference, so a slope above one means the
    # network under-reports and has to be scaled up.
    denominator = float(np.dot(net, net))
    if denominator <= 0.0:
        raise SystemExit("the network reported zero everywhere; refusing to calibrate")
    slope = float(np.dot(net, hce)) / denominator
    correlation = float(np.corrcoef(net, hce)[0, 1])
    cp_scale = round(slope * CP_SCALE_ONE)

    print(f"positions        {len(boards):,}")
    print(f"reference        {arguments.reference}")
    print(f"slope (hce/net)  {slope:.4f}   r={correlation:.3f}")
    print(f"cp_scale         {cp_scale}  (file currently carries {existing})")
    print(f"net  spread      sd {net.std():7.1f}  |mean| {abs(net.mean()):6.1f}")
    print(f"hce  spread      sd {hce.std():7.1f}  |mean| {abs(hce.mean()):6.1f}")

    if not 0 < cp_scale < 1 << 16:
        raise SystemExit(f"cp_scale {cp_scale} is outside what the agent accepts; not writing")
    if not 0.2 < slope < 5.0:
        raise SystemExit(
            f"slope {slope:.3f} is implausible. That is a broken network or a wrong "
            "quantisation, not a scale to paper over. Not writing."
        )
    if arguments.dry_run:
        print("dry run: nothing written")
        return 0

    with np.load(arguments.weights) as blob:
        fields = {key: blob[key] for key in blob.files}
    fields["cp_scale"] = np.int32(cp_scale)
    np.savez_compressed(arguments.weights, **fields)
    print(f"wrote cp_scale={cp_scale} into {arguments.weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
