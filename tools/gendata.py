"""Generate Stockfish-labelled positions for NNUE training.

Training data only. Stockfish never ships inside the submission; the rules allow
engine-annotated training data and ban only what runs in the zip.

    # one worker
    python3 -m tools.gendata --out data/train.0.csv --seed 0 --positions 1700000

    # six workers, then concatenate
    for k in 0 1 2 3 4 5; do
      python3 -m tools.gendata --out data/train.$k.csv --seed $k --positions 1700000 &
    done; wait
    cat data/train.*.csv > data/train.csv

Each row is `fen,score,result`. score is centipawns from the side to move,
result is the game result from White's point of view (1, 0.5, 0).

What is different from v3_gendata.py, and why
---------------------------------------------
v3 produced 926,724 positions whose median |material| was 350 cp, with only
18.3% inside 60 cp of level. In that distribution material counting explains
almost all the variance, so gradient descent never had to learn a positional
feature: the resulting net scored r=0.95 against engine labels overall and
r~=0 on level positions, and lost 280 Elo. That is regression bug #8.

Four changes, in the order they matter:

1. Seed from tools/openings.py rather than 2-12 uniformly random plies. Random
   plies hand out material in the first few moves, so a game is lopsided before
   it starts. Rated games begin from curated near-level positions, which is what
   the opening set imitates.
2. Temperature 10% -> 3%. A random move every ten plies is a blunder every ten
   plies, and the resulting imbalance persists for the rest of the game.
3. Truncate a game once it is decided. Most of v3's lopsided tail was decided
   games played out to 240 plies, every ply of which is another lopsided sample.
   Stopping at the point of decision removes the tail at its source and costs
   nothing, because those searches were being spent on positions that would be
   rejected anyway.
4. Reject-sample on |material| against a per-bucket quota, so what survives is
   flat rather than merely less skewed.

Positions where a king is under pressure are oversampled, because the iteration
log records both evaluations correlating *negatively* with Stockfish on quiet
king-exposed positions, which is the largest single weakness identified so far.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import chess

# Located from the environment so the same file runs in the sandbox and on the
# PC. v3 hardcoded a sandbox path, which is dead anywhere else.
SF: str = (
    os.environ.get("SPAR_ENGINE")
    or shutil.which("stockfish")
    or str(Path.home() / "bin" / "stockfish")
)

MATE_CP = 3000

# A game is over as a source of level positions once one side is this far ahead
# for this many plies. Not a resignation: the game stops, the result is recorded
# from the sign, and the engine moves on to a fresh opening.
RESIGN_CP = 600
RESIGN_PLIES = 6

BUCKET_CP = 60  # width of a material-balance bucket
BUCKET_COUNT = 11  # 0-60, 60-120, ... 540-600, and everything past 600

# Share of the target that each bucket may hold. Not uniform, deliberately.
# Material is a fixed skip connection in the kernel, so the network is *given*
# the balance and can only learn the positional residual on top of it — which
# means the distribution no longer has to be flat to stop it counting material,
# the architecture already does that. What it has to be is dense where the games
# are: 45% of this profile sits inside 120 cp of level against v3's 31.5%, and
# 71% inside 240 cp, while still keeping a real tail so the residual is learned
# across the whole range. A uniform quota also fights the resign truncation,
# which exists precisely to stop producing lopsided positions.
BUCKET_WEIGHT = (7, 7, 4, 4, 2, 2, 1, 1, 1, 1, 1)
KING_DANGER_BONUS = 0.25  # extra share of a full bucket reserved for king danger
KING_DANGER_MIN = 6  # attacker count on a king zone that counts as exposed


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
        if self.process.stdin is not None:
            self._send("quit")
        self.process.wait(timeout=5)


PIECE_VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500,
               chess.QUEEN: 900}


def material_balance(board: chess.Board) -> int:
    """Centipawn balance for the side to move. Matches the agent and the trainer."""
    total = 0
    for piece_type, value in PIECE_VALUE.items():
        total += value * chess.popcount(board.pieces_mask(piece_type, chess.WHITE))
        total -= value * chess.popcount(board.pieces_mask(piece_type, chess.BLACK))
    return total if board.turn == chess.WHITE else -total


def king_danger(board: chess.Board) -> int:
    """How many enemy pieces attack the box around each king. Same as v3_evalcmp."""
    total = 0
    for color in (chess.WHITE, chess.BLACK):
        king = board.king(color)
        if king is None:
            continue
        attackers = 0
        bits = chess.BB_KING_ATTACKS[king] | chess.BB_SQUARES[king]
        while bits:
            square = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            attackers += chess.popcount(board.attackers_mask(not color, square))
        total = max(total, attackers)
    return total


class Stratifier:
    """Per-bucket quotas over |material|, so what survives is flat.

    A bucket that is full still admits king-exposed positions up to a small
    extra share. Rejecting those as well would reproduce the training filter's
    other blind spot: it kept only quiet positions with no capture available,
    which is most of the reason the net never learned king safety.
    """

    def __init__(self, target: int, king_danger_bonus: float) -> None:
        total = sum(BUCKET_WEIGHT)
        self.quotas = [max(1, target * weight // total) for weight in BUCKET_WEIGHT]
        self.bonus = king_danger_bonus
        self.counts = [0] * BUCKET_COUNT
        self.kept = 0

    def bucket(self, balance: int) -> int:
        return min(abs(balance) // BUCKET_CP, BUCKET_COUNT - 1)

    def accept(self, balance: int, exposed: bool) -> bool:
        index = self.bucket(balance)
        quota = self.quotas[index]
        limit = quota * (1.0 + self.bonus) if exposed else float(quota)
        if self.counts[index] >= limit:
            return False
        self.counts[index] += 1
        self.kept += 1
        return True

    def full(self) -> bool:
        return all(count >= quota for count, quota in zip(self.counts, self.quotas, strict=True))

    def histogram(self) -> str:
        return " ".join(f"{index * BUCKET_CP}:{count}" for index, count in enumerate(self.counts))


def play_game(
    engine: Engine, board: chess.Board, rng: random.Random, nodes: int, ply_cap: int,
    temperature: float,
) -> tuple[list[tuple[str, int, int, int]], float]:
    """Play one game from `board`. Returns quiet samples and the White result.

    A sample is (fen, score, material balance for the side to move, king danger).
    Filtering happens in the caller so that this stays a pure play-out.
    """
    engine.new_game()
    samples: list[tuple[str, int, int, int]] = []
    decided = 0
    white_result: float | None = None

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        fen = board.fen()
        best, score = engine.search(fen, nodes)
        if best == "(none)":
            break
        move = chess.Move.from_uci(best)
        if move not in board.legal_moves:
            break
        # Quiet positions only: a static evaluation trained on positions that are
        # mid-exchange learns noise. Our own search runs quiescence, so it only
        # ever calls the net on positions it believes are quiet.
        if not board.is_check() and not board.is_capture(move) and abs(score) < MATE_CP:
            samples.append((fen, score, material_balance(board), king_danger(board)))

        white_score = score if board.turn == chess.WHITE else -score
        decided = decided + 1 if abs(white_score) >= RESIGN_CP else 0
        if decided >= RESIGN_PLIES:
            # Every further ply of a decided game is another lopsided sample, and
            # it is exactly the tail that made the v3 distribution useless.
            white_result = 1.0 if white_score > 0 else 0.0
            break

        if rng.random() < temperature:
            move = rng.choice(list(board.legal_moves))
        board.push(move)

    if white_result is not None:
        return samples, white_result
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return samples, 0.5 if outcome.winner is None else float(outcome.winner == chess.WHITE)
    if not samples:
        return samples, 0.5
    fen, score, _balance, _danger = samples[-1]
    white_score = score if fen.split()[1] == "w" else -score
    return samples, 1.0 / (1.0 + 10.0 ** (-white_score / 400.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--positions", type=int, default=1_700_000, help="target for this worker")
    parser.add_argument("--minutes", type=float, default=0.0, help="stop early after this long")
    parser.add_argument("--nodes", type=int, default=1200)
    parser.add_argument("--ply-cap", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--king-danger-bonus", type=float, default=KING_DANGER_BONUS)
    parser.add_argument("--engine", type=str, default=SF)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    from tools.openings import OPENINGS

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(arguments.seed)
    engine = Engine(arguments.engine)
    strata = Stratifier(arguments.positions, arguments.king_danger_bonus)
    deadline = time.monotonic() + arguments.minutes * 60.0 if arguments.minutes else float("inf")
    started = time.monotonic()
    games = 0
    seen = 0

    # Line buffered on purpose: a killed run keeps every position it wrote, which
    # is what makes six parallel workers safe to interrupt.
    with arguments.out.open("a", buffering=1) as handle:
        while strata.kept < arguments.positions and time.monotonic() < deadline:
            board = chess.Board(rng.choice(OPENINGS))
            samples, result = play_game(
                engine, board, rng, arguments.nodes, arguments.ply_cap, arguments.temperature
            )
            for fen, score, balance, danger in samples:
                seen += 1
                if strata.accept(balance, danger >= KING_DANGER_MIN):
                    handle.write(f"{fen},{score},{result:.3f}\n")
            games += 1
            if games % 50 == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"{games} games, {strata.kept:,} kept of {seen:,} "
                    f"({strata.kept / max(seen, 1):.0%}), {strata.kept / elapsed:.0f} kept/s",
                    flush=True,
                )

    engine.close()
    print(f"done: {games} games, {strata.kept:,} kept of {seen:,} -> {arguments.out}", flush=True)
    print(f"histogram |material|: {strata.histogram()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
