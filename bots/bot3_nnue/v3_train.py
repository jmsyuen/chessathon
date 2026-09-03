"""Train the NNUE that ships in bots/bot2_nnue.py.

    python3 -m tools.train --data data/train.csv --out weights/nnue.npz

Architecture: (768 -> H) x 2 perspectives, squared clipped ReLU, one output.
Both accumulators share one feature transformer, which is the standard trick: the
net sees the position from both sides and the output layer learns the asymmetry.

Only numpy and numba are used. The gather and scatter over the sparse input are
jitted because that is the whole cost of a step; everything else is a small dense
matmul. torch would work too, it just is not needed for a net this size.

Nothing here ships. This file trains the weights; bots/bot2_nnue.py reads them.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from numba import njit

PIECE_INDEX = {
    "p": (0, 0), "n": (1, 0), "b": (2, 0), "r": (3, 0), "q": (4, 0), "k": (5, 0),
    "P": (0, 1), "N": (1, 1), "B": (2, 1), "R": (3, 1), "Q": (4, 1), "K": (5, 1),
}
MAX_PIECES = 32
MATERIAL = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900, "k": 0}
PAD = 768  # a dummy feature row, held at zero, so padded slots need no masking
SCALE = 400.0


def encode(fen_board: str, white_to_move: bool) -> tuple[list[int], list[int]]:
    """Feature indices for both perspectives. Must match the agent's kernel exactly."""
    us: list[int] = []
    them: list[int] = []
    square = 56  # a8; FEN starts at rank 8
    for char in fen_board:
        if char == "/":
            square -= 16
        elif char.isdigit():
            square += int(char)
        else:
            piece_type, color = PIECE_INDEX[char]
            same = 0 if (color == 1) == white_to_move else 1
            if white_to_move:
                us_square, them_square = square, square ^ 56
            else:
                us_square, them_square = square ^ 56, square
            us.append(same * 384 + piece_type * 64 + us_square)
            them.append((1 - same) * 384 + piece_type * 64 + them_square)
            square += 1
    return us, them


def material_of(fen_board: str, white_to_move: bool) -> int:
    """Centipawn material balance for the side to move. Must match the agent kernel."""
    total = 0
    for char in fen_board:
        value = MATERIAL.get(char.lower())
        if value:
            total += value if char.isupper() else -value
    return total if white_to_move else -total


def load(path: Path, cache: Path) -> tuple[np.ndarray, ...]:
    """Parse the csv into packed arrays, caching the result next to it."""
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        blob = np.load(cache)
        return blob["us"], blob["them"], blob["score"], blob["result"], blob["material"]

    us_rows: list[np.ndarray] = []
    them_rows: list[np.ndarray] = []
    scores: list[int] = []
    results: list[float] = []
    materials: list[int] = []
    started = time.perf_counter()
    with path.open() as handle:
        for line in handle:
            fen, score_text, result_text = line.rsplit(",", 2)
            fields = fen.split(" ")
            if len(fields) < 2:
                continue
            white_to_move = fields[1] == "w"
            us, them = encode(fields[0], white_to_move)
            if not 2 <= len(us) <= MAX_PIECES:
                continue
            us_row = np.full(MAX_PIECES, PAD, dtype=np.int16)
            them_row = np.full(MAX_PIECES, PAD, dtype=np.int16)
            us_row[: len(us)] = us
            them_row[: len(them)] = them
            us_rows.append(us_row)
            them_rows.append(them_row)
            scores.append(int(score_text))
            materials.append(material_of(fields[0], white_to_move))
            white_result = float(result_text)
            results.append(white_result if white_to_move else 1.0 - white_result)

    us_array = np.asarray(us_rows, dtype=np.int16)
    them_array = np.asarray(them_rows, dtype=np.int16)
    score_array = np.asarray(scores, dtype=np.float32)
    result_array = np.asarray(results, dtype=np.float32)
    print(f"parsed {len(score_array):,} positions in {time.perf_counter() - started:.1f}s")
    material_array = np.asarray(materials, dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, us=us_array, them=them_array, score=score_array,
             result=result_array, material=material_array)
    return us_array, them_array, score_array, result_array, material_array


@njit(cache=False, fastmath=True)
def gather(weights: np.ndarray, bias: np.ndarray, index: np.ndarray, out: np.ndarray) -> None:
    """out[b] = bias + sum of the weight rows this sample's features select."""
    rows, slots = index.shape
    hidden = weights.shape[1]
    for row in range(rows):
        for column in range(hidden):
            out[row, column] = bias[column]
        for slot in range(slots):
            feature = index[row, slot]
            for column in range(hidden):
                out[row, column] += weights[feature, column]


@njit(cache=False, fastmath=True)
def scatter(grad: np.ndarray, index: np.ndarray, source: np.ndarray) -> None:
    """The transpose of gather: accumulate each sample's gradient into its rows."""
    rows, slots = index.shape
    hidden = grad.shape[1]
    for row in range(rows):
        for slot in range(slots):
            feature = index[row, slot]
            for column in range(hidden):
                grad[feature, column] += source[row, column]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Adam:
    def __init__(self, shapes: list[tuple[int, ...]], lr: float) -> None:
        self.lr = lr
        self.step = 0
        self.m = [np.zeros(shape, dtype=np.float32) for shape in shapes]
        self.v = [np.zeros(shape, dtype=np.float32) for shape in shapes]

    def apply(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.step += 1
        bias1 = 1.0 - 0.9**self.step
        bias2 = 1.0 - 0.999**self.step
        for param, grad, m, v in zip(params, grads, self.m, self.v, strict=True):
            m *= 0.9
            m += 0.1 * grad
            v *= 0.999
            v += 0.001 * grad * grad
            param -= self.lr * (m / bias1) / (np.sqrt(v / bias2) + 1e-8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--out", type=Path, default=Path("weights/nnue.npz"))
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-eval", type=float, default=0.75)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--minutes", type=float, default=0.0, help="stop early after this long")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state", type=Path, default=Path("data/state.npz"),
                        help="float checkpoint, so training can span several sessions")
    arguments = parser.parse_args()

    cache = arguments.data.with_suffix(".npz")
    us, them, score, result, material = load(arguments.data, cache)
    total = len(score)
    rng = np.random.default_rng(arguments.seed)
    order = rng.permutation(total)
    us, them, score, result, material = (
        us[order], them[order], score[order], result[order], material[order]
    )
    # Material is handed to the network for free, as a fixed skip connection, so the
    # only thing gradient descent can learn is the positional residual. Without
    # this the net just learns to count material and is blind in level positions.
    offset = (material / SCALE).astype(np.float32)

    # The target every NNUE is trained on: the engine score squashed into a win
    # probability, blended with what actually happened in the game.
    target = arguments.lambda_eval * sigmoid(score / SCALE) + (
        1.0 - arguments.lambda_eval
    ) * result
    target = target.astype(np.float32)

    split = max(arguments.batch, int(total * arguments.val_fraction))
    val = slice(0, split)
    train = slice(split, total)
    hidden = arguments.hidden
    print(f"{total - split:,} train, {split:,} val, hidden {hidden}")

    start_epoch = 0
    state = None
    if arguments.state is not None and arguments.state.exists():
        state = np.load(arguments.state)
        if int(state["hidden"]) == hidden:
            w0, b0, w1, b1 = (state[k].copy() for k in ("w0", "b0", "w1", "b1"))
            start_epoch = int(state["epoch"])
            print(f"resumed from {arguments.state} at epoch {start_epoch}")
        else:
            state = None
    if state is None:
        limit = np.float32(1.0 / np.sqrt(32.0))
        w0 = (rng.standard_normal((PAD + 1, hidden)) * limit * 0.1).astype(np.float32)
        b0 = np.zeros(hidden, dtype=np.float32)
        w1 = (rng.standard_normal(2 * hidden) * 0.1).astype(np.float32)
        b1 = np.zeros(1, dtype=np.float32)
    w0[PAD] = 0.0

    optimiser = Adam([w0.shape, b0.shape, w1.shape, b1.shape], arguments.lr)
    if state is not None:
        optimiser.step = int(state["step"])
        optimiser.m = [state[f"m{i}"].copy() for i in range(4)]
        optimiser.v = [state[f"v{i}"].copy() for i in range(4)]
    best = float(state["best"]) if state is not None else float("inf")
    batch = arguments.batch
    acc_us = np.zeros((batch, hidden), dtype=np.float32)
    acc_them = np.zeros((batch, hidden), dtype=np.float32)
    grad_w0 = np.zeros_like(w0)
    deadline = time.monotonic() + arguments.minutes * 60.0 if arguments.minutes else float("inf")
    started = time.monotonic()

    train_count = total - split
    steps = train_count // batch

    def evaluate_split(where: slice) -> float:
        losses = []
        for start in range(where.start, where.stop - batch + 1, batch):
            stop = start + batch
            gather(w0, b0, us[start:stop], acc_us)
            gather(w0, b0, them[start:stop], acc_them)
            clipped_us = np.clip(acc_us, 0.0, 1.0)
            clipped_them = np.clip(acc_them, 0.0, 1.0)
            out = (
                (clipped_us * clipped_us) @ w1[:hidden]
                + (clipped_them * clipped_them) @ w1[hidden:]
                + b1[0]
                + offset[start:stop]
            )
            losses.append(float(np.mean((sigmoid(out) - target[start:stop]) ** 2)))
        return float(np.mean(losses)) if losses else float("nan")

    for epoch in range(start_epoch, arguments.epochs):
        # Cosine decay. The last epochs at a small learning rate are where the
        # quantisation-sensitive fine structure settles.
        optimiser.lr = arguments.lr * (
            0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * epoch / max(1, arguments.epochs)))
        )
        shuffle = rng.permutation(train_count) + split
        epoch_loss = 0.0
        for step in range(steps):
            rows = shuffle[step * batch : (step + 1) * batch]
            index_us = np.ascontiguousarray(us[rows])
            index_them = np.ascontiguousarray(them[rows])
            batch_target = target[rows]

            gather(w0, b0, index_us, acc_us)
            gather(w0, b0, index_them, acc_them)
            clipped_us = np.clip(acc_us, 0.0, 1.0)
            clipped_them = np.clip(acc_them, 0.0, 1.0)
            squared_us = clipped_us * clipped_us
            squared_them = clipped_them * clipped_them
            out = (
                squared_us @ w1[:hidden] + squared_them @ w1[hidden:] + b1[0] + offset[rows]
            )
            prediction = sigmoid(out)
            epoch_loss += float(np.mean((prediction - batch_target) ** 2))

            upstream = (2.0 * (prediction - batch_target) * prediction * (1.0 - prediction)) / batch
            grad_w1 = np.concatenate((squared_us.T @ upstream, squared_them.T @ upstream))
            grad_b1 = np.array([upstream.sum()], dtype=np.float32)
            grad_w1 = grad_w1.astype(np.float32, copy=False)

            active_us = (acc_us > 0.0) & (acc_us < 1.0)
            active_them = (acc_them > 0.0) & (acc_them < 1.0)
            delta_us = (upstream[:, None] * w1[None, :hidden]) * (2.0 * clipped_us) * active_us
            delta_them = (upstream[:, None] * w1[None, hidden:]) * (2.0 * clipped_them) * active_them
            grad_b0 = (delta_us.sum(axis=0) + delta_them.sum(axis=0)).astype(np.float32)
            grad_w0.fill(0.0)
            scatter(grad_w0, index_us, delta_us.astype(np.float32))
            scatter(grad_w0, index_them, delta_them.astype(np.float32))
            grad_w0[PAD] = 0.0

            optimiser.apply([w0, b0, w1, b1], [grad_w0, grad_b0, grad_w1, grad_b1])
            # Keep everything inside the range int16 quantisation can represent.
            np.clip(w0, -1.98, 1.98, out=w0)
            np.clip(w1, -1.98, 1.98, out=w1)
            w0[PAD] = 0.0

        validation = evaluate_split(val)
        elapsed = time.monotonic() - started
        print(
            f"epoch {epoch + 1:>3}/{arguments.epochs}  train {epoch_loss / max(1, steps):.5f}  "
            f"val {validation:.5f}  lr {optimiser.lr:.4f}  {elapsed / 60:.1f} min",
            flush=True,
        )
        if not np.isfinite(validation):
            validation = epoch_loss / max(1, steps)
        if validation < best:
            best = validation
            arguments.out.parent.mkdir(parents=True, exist_ok=True)
            save(arguments.out, w0, b0, w1, b1, hidden)
        if arguments.state is not None:
            arguments.state.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                arguments.state, w0=w0, b0=b0, w1=w1, b1=b1, hidden=np.int32(hidden),
                epoch=np.int32(epoch + 1), step=np.int32(optimiser.step),
                best=np.float32(best),
                **{f"m{i}": optimiser.m[i] for i in range(4)},
                **{f"v{i}": optimiser.v[i] for i in range(4)},
            )
        if time.monotonic() > deadline:
            print("time budget reached", flush=True)
            break

    print(f"best val {best:.5f} -> {arguments.out}")
    return 0


QA = 1024
QB = 2048


def save(path: Path, w0, b0, w1, b1, hidden: int) -> None:  # type: ignore[no-untyped-def]
    """Quantise and write. The agent reads exactly these four arrays."""
    quantised_w0 = np.rint(w0[:PAD] * QA).astype(np.int16)
    quantised_b0 = np.rint(b0 * QA).astype(np.int16)
    quantised_w1 = np.rint(w1 * QB).astype(np.int16)
    quantised_b1 = np.int32(np.rint(b1[0] * QA * QB))
    np.savez_compressed(
        path,
        w0=quantised_w0,
        b0=quantised_b0,
        w1=quantised_w1,
        b1=quantised_b1,
        hidden=np.int32(hidden),
        qa=np.int32(QA),
        qb=np.int32(QB),
        scale=np.int32(int(SCALE)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
