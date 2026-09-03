"""Generate Stockfish-labelled positions for NNUE training.

Training data only. Stockfish never ships inside the submission; the rules allow
engine-annotated training data and ban only what runs in the zip.

    python3 -m tools.gendata --out data/train.csv --minutes 60

Each row is `fen,score,result`. score is centipawns from the side to move,
result is the game result from White's point of view (1, 0.5, 0).
"""

from __future__ import annotations

import argparse
import random
import subprocess
import time
from pathlib import Path

import chess

SF = "/home/claude/bin/stockfish"
MATE_CP = 3000


class Engine:
    """Minimal synchronous UCI pipe. python-chess's asyncio layer is too slow here."""

    def __init__(self, path: str, hash_mb: int = 64) -> None:
        self.process = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self._send("uci")
        self._await("uciok")
        self._send("setoption name Threads value 1")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("setoption name UCI_Chess960 value false")
        self.ready()

    def _send(self, line: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def _await(self, token: str) -> None:
        assert self.process.stdout is not None
        while True:
            line = self.process.stdout.readline()
            if not line or line.startswith(token):
                return

    def ready(self) -> None:
        self._send("isready")
        self._await("readyok")

    def new_game(self) -> None:
        self._send("ucinewgame")
        self.ready()

    def search(self, fen: str, nodes: int) -> tuple[str, int]:
        """Return (bestmove uci, score in centipawns from the side to move)."""
        assert self.process.stdout is not None
        self._send(f"position fen {fen}")
        self._send(f"go nodes {nodes}")
        score = 0
        while True:
            line = self.process.stdout.readline()
            if not line:
                return "(none)", 0
            if line.startswith("info ") and " score " in line:
                parts = line.split()
                index = parts.index("score")
                kind = parts[index + 1]
                value = int(parts[index + 2])
                score = value if kind == "cp" else (MATE_CP if value > 0 else -MATE_CP)
            elif line.startswith("bestmove"):
                return line.split()[1], score

    def close(self) -> None:
        with_stdin = self.process.stdin
        if with_stdin is not None:
            self._send("quit")
        self.process.wait(timeout=5)


def random_opening(rng: random.Random) -> chess.Board:
    """A varied, mostly-sane start.

    Rated games begin from curated near-level positions, so the data should not
    be all standard-opening theory. Random plies give breadth; the label is
    correct either way and the net has to evaluate lopsided positions too.
    """
    board = chess.Board()
    plies = rng.choice((2, 4, 6, 8, 8, 10, 12))
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            return chess.Board()
        board.push(rng.choice(moves))
        if board.is_game_over():
            return chess.Board()
    return board


def play_game(
    engine: Engine, rng: random.Random, nodes: int, ply_cap: int
) -> tuple[list[tuple[str, int]], float]:
    """Play one game. Returns quiet (fen, score) samples and the White result."""
    board = random_opening(rng)
    engine.new_game()
    samples: list[tuple[str, int]] = []
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        fen = board.fen()
        best, score = engine.search(fen, nodes)
        if best == "(none)":
            break
        move = chess.Move.from_uci(best)
        if move not in board.legal_moves:
            break
        # Quiet positions only: a static evaluation trained on positions that are
        # mid-exchange learns noise. Our own search does quiescence, so it only
        # ever calls the net on positions it believes are quiet.
        if not board.is_check() and not board.is_capture(move) and abs(score) < MATE_CP:
            samples.append((fen, score))
        if rng.random() < 0.10:
            move = rng.choice(list(board.legal_moves))  # temperature, for diversity
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is not None and outcome.winner is not None:
        result = 1.0 if outcome.winner == chess.WHITE else 0.0
    elif outcome is not None:
        result = 0.5
    else:
        # Hit the ply cap. Fall back to the last score, converted to a soft result,
        # so an unfinished game still carries a usable outcome signal.
        if not samples:
            return samples, 0.5
        fen, score = samples[-1]
        white_score = score if fen.split()[1] == "w" else -score
        result = 1.0 / (1.0 + 10.0 ** (-white_score / 400.0))
    return samples, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--nodes", type=int, default=2500)
    parser.add_argument("--ply-cap", type=int, default=240)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(arguments.seed)
    engine = Engine(SF)
    deadline = time.monotonic() + arguments.minutes * 60.0
    written = 0
    games = 0
    started = time.monotonic()

    with arguments.out.open("a", buffering=1) as handle:
        while time.monotonic() < deadline:
            samples, result = play_game(engine, rng, arguments.nodes, arguments.ply_cap)
            for fen, score in samples:
                handle.write(f"{fen},{score},{result:.3f}\n")
                written += 1
            games += 1
            if games % 25 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"{games} games, {written} positions, "
                    f"{written / max(elapsed, 1e-9):.0f} pos/s",
                    flush=True,
                )

    engine.close()
    print(f"done: {games} games, {written} positions -> {arguments.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
