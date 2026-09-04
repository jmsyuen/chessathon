"""AI Chessathon agent: alpha-beta over a numba bitboard move generator.

GENERATED FILE. Built by tools/flatten.py from agent.py, ac_board.py,
ac_bitboard.py, ac_eval.py and ac_search.py. Edit those, not this.

Layout, in order:
  1. ac_bitboard  bitboards, jitted move generation, make/unmake, Zobrist
  2. ac_board     the frozen board contract, plus the python-chess reference
  3. ac_eval      material and piece-square tables, jitted
  4. ac_search    alpha-beta, iterative deepening, quiescence, transposition table
  5. agent        get_move, the clock, and game history reconstruction
"""

from __future__ import annotations


# ==========================================================================
# ac_bitboard.py
# ==========================================================================

from typing import Any

import numpy as np
from numba import njit

MAX_PLY = 128
MAX_MOVES = 256

WHITE = 0
BLACK = 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

FLAG_CAPTURE = 1 << 15
FLAG_EP = 1 << 16
FLAG_CASTLE = 1 << 17

CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8

# Direction order: rays 0-3 run toward higher square indices, 4-7 toward lower.
# That split is what decides whether the first blocker is found with lsb or msb.
DIRECTIONS = (8, 9, 1, 7, -8, -9, -1, -7)
ROOK_DIRS = (0, 2, 4, 6)
BISHOP_DIRS = (1, 3, 5, 7)


# --------------------------------------------------------------------------
# Tables, built once at import in plain numpy
# --------------------------------------------------------------------------


def _build_tables() -> tuple[np.ndarray, ...]:
    knight = np.zeros(64, np.uint64)
    king = np.zeros(64, np.uint64)
    pawn = np.zeros((2, 64), np.uint64)
    rays = np.zeros((8, 64), np.uint64)

    for square in range(64):
        file, rank = square & 7, square >> 3
        for df, dr in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
            f, r = file + df, rank + dr
            if 0 <= f < 8 and 0 <= r < 8:
                knight[square] |= np.uint64(1) << np.uint64(r * 8 + f)
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0:
                    continue
                f, r = file + df, rank + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    king[square] |= np.uint64(1) << np.uint64(r * 8 + f)
        for df in (-1, 1):
            if 0 <= file + df < 8:
                if rank + 1 < 8:
                    pawn[WHITE][square] |= np.uint64(1) << np.uint64((rank + 1) * 8 + file + df)
                if rank - 1 >= 0:
                    pawn[BLACK][square] |= np.uint64(1) << np.uint64((rank - 1) * 8 + file + df)
        for index, delta in enumerate(DIRECTIONS):
            df = {8: 0, 9: 1, 1: 1, 7: -1, -8: 0, -9: -1, -1: -1, -7: 1}[delta]
            dr = {8: 1, 9: 1, 1: 0, 7: 1, -8: -1, -9: -1, -1: 0, -7: -1}[delta]
            f, r = file + df, rank + dr
            while 0 <= f < 8 and 0 <= r < 8:
                rays[index][square] |= np.uint64(1) << np.uint64(r * 8 + f)
                f += df
                r += dr
    return knight, king, pawn, rays


KNIGHT_ATTACKS, KING_ATTACKS, PAWN_ATTACKS, RAYS = _build_tables()

# Zobrist keys. These are NOT the same numbers the python-chess layer uses, and
# they do not need to be: a hash only has to be self-consistent within one
# process, and the two backends never run in the same one. The differential test
# compares legal move sets, not keys, for exactly this reason.
_rng = np.random.default_rng(0x9E3779B9)
Z_PIECE = _rng.integers(0, 1 << 63, size=(12, 64), dtype=np.uint64)
Z_SIDE = np.uint64(_rng.integers(0, 1 << 63, dtype=np.uint64))
Z_CASTLE = _rng.integers(0, 1 << 63, size=16, dtype=np.uint64)
Z_EP = _rng.integers(0, 1 << 63, size=8, dtype=np.uint64)


# --------------------------------------------------------------------------
# Bit primitives
# --------------------------------------------------------------------------


@njit(cache=False, inline="always")
def _msb(bits: Any) -> int:
    n = 0
    if bits >> np.uint64(32):
        bits >>= np.uint64(32)
        n += 32
    if bits >> np.uint64(16):
        bits >>= np.uint64(16)
        n += 16
    if bits >> np.uint64(8):
        bits >>= np.uint64(8)
        n += 8
    if bits >> np.uint64(4):
        bits >>= np.uint64(4)
        n += 4
    if bits >> np.uint64(2):
        bits >>= np.uint64(2)
        n += 2
    if bits >> np.uint64(1):
        n += 1
    return n


@njit(cache=False, inline="always")
def _lsb(bits: Any) -> int:
    return _msb(bits & (np.uint64(0) - bits))


@njit(cache=False, inline="always")
def _bit(square: int) -> Any:
    return np.uint64(1) << np.uint64(square)


@njit(cache=False)
def _popcount(bits: Any) -> int:
    total = 0
    while bits:
        bits &= bits - np.uint64(1)
        total += 1
    return total


# --------------------------------------------------------------------------
# Attacks
# --------------------------------------------------------------------------


@njit(cache=False, inline="always")
def _ray(direction: int, square: int, occupied: Any) -> Any:
    attacks = RAYS[direction][square]
    blockers = attacks & occupied
    if blockers:
        # Rays 0-3 climb the board, so the nearest blocker is the low bit.
        first = _lsb(blockers) if direction < 4 else _msb(blockers)
        attacks ^= RAYS[direction][first]
    return attacks


@njit(cache=False)
def _bishop_attacks(square: int, occupied: Any) -> Any:
    return (
        _ray(1, square, occupied)
        | _ray(3, square, occupied)
        | _ray(5, square, occupied)
        | _ray(7, square, occupied)
    )


@njit(cache=False)
def _rook_attacks(square: int, occupied: Any) -> Any:
    return (
        _ray(0, square, occupied)
        | _ray(2, square, occupied)
        | _ray(4, square, occupied)
        | _ray(6, square, occupied)
    )


@njit(cache=False)
def _attacked(bb: Any, square: int, by: int) -> bool:
    """Is `square` attacked by side `by`? Occupancy is read from bb[14]."""
    base = by * 6
    if PAWN_ATTACKS[1 - by][square] & bb[base]:
        return True
    if KNIGHT_ATTACKS[square] & bb[base + 1]:
        return True
    if KING_ATTACKS[square] & bb[base + 5]:
        return True
    occupied = bb[14]
    diagonal = bb[base + 2] | bb[base + 4]
    if diagonal and _bishop_attacks(square, occupied) & diagonal:
        return True
    straight = bb[base + 3] | bb[base + 4]
    return bool(straight and _rook_attacks(square, occupied) & straight)


@njit(cache=False)
def _in_check(bb: Any, side: int) -> bool:
    king = bb[side * 6 + 5]
    if not king:
        return False
    return _attacked(bb, _lsb(king), 1 - side)


# --------------------------------------------------------------------------
# Make and unmake
# --------------------------------------------------------------------------


@njit(cache=False, inline="always")
def _place(bb: Any, colour: int, piece: int, square: int) -> None:
    mask = _bit(square)
    bb[colour * 6 + piece - 1] |= mask
    bb[12 + colour] |= mask
    bb[14] |= mask


@njit(cache=False, inline="always")
def _lift(bb: Any, colour: int, piece: int, square: int) -> None:
    mask = ~_bit(square)
    bb[colour * 6 + piece - 1] &= mask
    bb[12 + colour] &= mask
    bb[14] &= mask


@njit(cache=False)
def _ep_is_legal(bb: Any, st: Any) -> bool:
    """Can the side to move actually take en passant, king safety included?

    This decides whether the ep file enters the hash at all, so it has to be
    exact rather than "a pawn is adjacent". The capture is played directly on
    the bitboards and undone, which avoids recursing through make().
    """
    target = st[2]
    if target < 0:
        return False
    side = st[0]
    movers = PAWN_ATTACKS[1 - side][target] & bb[side * 6]
    if not movers:
        return False
    victim_square = target - 8 if side == WHITE else target + 8
    while movers:
        source = _lsb(movers)
        movers &= movers - np.uint64(1)
        _lift(bb, side, PAWN, source)
        _lift(bb, 1 - side, PAWN, victim_square)
        _place(bb, side, PAWN, target)
        safe = not _in_check(bb, side)
        _lift(bb, side, PAWN, target)
        _place(bb, 1 - side, PAWN, victim_square)
        _place(bb, side, PAWN, source)
        if safe:
            return True
    return False


@njit(cache=False)
def _ep_key(bb: Any, st: Any) -> Any:
    if st[2] < 0:
        return np.uint64(0)
    if not _ep_is_legal(bb, st):
        return np.uint64(0)
    return Z_EP[st[2] & 7]


@njit(cache=False)
def _castle_update(rights: int, square: int) -> int:
    """A rook or king leaving or being taken on these squares kills the right."""
    if square == 4:
        rights &= ~(CASTLE_WK | CASTLE_WQ)
    elif square == 60:
        rights &= ~(CASTLE_BK | CASTLE_BQ)
    elif square == 7:
        rights &= ~CASTLE_WK
    elif square == 0:
        rights &= ~CASTLE_WQ
    elif square == 63:
        rights &= ~CASTLE_BK
    elif square == 56:
        rights &= ~CASTLE_BQ
    return rights


@njit(cache=False)
def make(bb: Any, st: Any, key: Any, undo: Any, move: int) -> None:
    source = move & 63
    target = (move >> 6) & 63
    promotion = (move >> 12) & 7
    victim = (move >> 18) & 7
    mover = (move >> 21) & 7
    side = st[0]
    depth = st[4]

    undo[depth, 0] = st[1]
    undo[depth, 1] = st[2]
    undo[depth, 2] = st[3]
    undo[depth, 3] = victim
    undo[depth, 4] = move

    running = key[0] ^ Z_SIDE ^ Z_CASTLE[st[1]] ^ _ep_key(bb, st)

    if move & FLAG_EP:
        victim_square = target - 8 if side == WHITE else target + 8
        _lift(bb, 1 - side, PAWN, victim_square)
        running ^= Z_PIECE[(1 - side) * 6][victim_square]
    elif victim:
        _lift(bb, 1 - side, victim, target)
        running ^= Z_PIECE[(1 - side) * 6 + victim - 1][target]

    _lift(bb, side, mover, source)
    running ^= Z_PIECE[side * 6 + mover - 1][source]
    landed = promotion if promotion else mover
    _place(bb, side, landed, target)
    running ^= Z_PIECE[side * 6 + landed - 1][target]

    if move & FLAG_CASTLE:
        if target > source:
            rook_from, rook_to = target + 1, target - 1
        else:
            rook_from, rook_to = target - 2, target + 1
        _lift(bb, side, ROOK, rook_from)
        _place(bb, side, ROOK, rook_to)
        running ^= Z_PIECE[side * 6 + ROOK - 1][rook_from]
        running ^= Z_PIECE[side * 6 + ROOK - 1][rook_to]

    st[1] = _castle_update(_castle_update(st[1], source), target)
    if mover == PAWN and abs(target - source) == 16:
        st[2] = (source + target) // 2
    else:
        st[2] = -1
    st[3] = 0 if (mover == PAWN or victim or (move & FLAG_EP)) else st[3] + 1
    st[0] = 1 - side
    st[4] = depth + 1

    key[0] = running ^ Z_CASTLE[st[1]] ^ _ep_key(bb, st)


@njit(cache=False)
def unmake(bb: Any, st: Any, key: Any, undo: Any, prior: Any) -> None:
    depth = st[4] - 1
    move = undo[depth, 4]
    source = move & 63
    target = (move >> 6) & 63
    promotion = (move >> 12) & 7
    victim = undo[depth, 3]
    mover = (move >> 21) & 7
    side = 1 - st[0]

    landed = promotion if promotion else mover
    _lift(bb, side, landed, target)
    _place(bb, side, mover, source)

    if move & FLAG_CASTLE:
        if target > source:
            rook_from, rook_to = target + 1, target - 1
        else:
            rook_from, rook_to = target - 2, target + 1
        _lift(bb, side, ROOK, rook_to)
        _place(bb, side, ROOK, rook_from)
    elif move & FLAG_EP:
        victim_square = target - 8 if side == WHITE else target + 8
        _place(bb, 1 - side, PAWN, victim_square)
    elif victim:
        _place(bb, 1 - side, victim, target)

    st[1] = undo[depth, 0]
    st[2] = undo[depth, 1]
    st[3] = undo[depth, 2]
    st[0] = side
    st[4] = depth
    key[0] = prior


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


@njit(cache=False, inline="always")
def _encode(
    source: int, target: int, promotion: int, flags: int, victim: int, mover: int
) -> int:
    return (
        source
        | (target << 6)
        | (promotion << 12)
        | flags
        | (victim << 18)
        | (mover << 21)
    )


@njit(cache=False)
def _victim_at(bb: Any, colour: int, square: int) -> int:
    mask = _bit(square)
    base = colour * 6
    for piece in range(6):
        if bb[base + piece] & mask:
            return piece + 1
    return 0


@njit(cache=False)
def generate_pseudo(bb: Any, st: Any, out: Any, captures_only: bool) -> int:
    side = st[0]
    them = 1 - side
    base = side * 6
    occupied = bb[14]
    theirs = bb[12 + them]
    mine = bb[12 + side]
    targets = theirs if captures_only else ~mine
    count = 0

    # --- pawns ---
    pawns = bb[base]
    forward = 8 if side == WHITE else -8
    start_rank = 1 if side == WHITE else 6
    last_rank = 7 if side == WHITE else 0
    while pawns:
        source = _lsb(pawns)
        pawns &= pawns - np.uint64(1)
        rank = source >> 3

        capture_mask = PAWN_ATTACKS[side][source] & theirs
        while capture_mask:
            target = _lsb(capture_mask)
            capture_mask &= capture_mask - np.uint64(1)
            victim = _victim_at(bb, them, target)
            if (target >> 3) == last_rank:
                for promotion in (QUEEN, ROOK, BISHOP, KNIGHT):
                    out[count] = _encode(source, target, promotion, FLAG_CAPTURE, victim, PAWN)
                    count += 1
            else:
                out[count] = _encode(source, target, 0, FLAG_CAPTURE, victim, PAWN)
                count += 1

        if st[2] >= 0 and (PAWN_ATTACKS[side][source] & _bit(st[2])):
            out[count] = _encode(
                source, st[2], 0, FLAG_CAPTURE | FLAG_EP, PAWN, PAWN
            )
            count += 1

        if captures_only:
            continue
        one = source + forward
        if not (occupied & _bit(one)):
            if (one >> 3) == last_rank:
                for promotion in (QUEEN, ROOK, BISHOP, KNIGHT):
                    out[count] = _encode(source, one, promotion, 0, 0, PAWN)
                    count += 1
            else:
                out[count] = _encode(source, one, 0, 0, 0, PAWN)
                count += 1
                if rank == start_rank:
                    two = one + forward
                    if not (occupied & _bit(two)):
                        out[count] = _encode(source, two, 0, 0, 0, PAWN)
                        count += 1

    # --- knights, king ---
    for piece, table in ((KNIGHT, KNIGHT_ATTACKS), (KING, KING_ATTACKS)):
        pieces = bb[base + piece - 1]
        while pieces:
            source = _lsb(pieces)
            pieces &= pieces - np.uint64(1)
            moves = table[source] & targets
            while moves:
                target = _lsb(moves)
                moves &= moves - np.uint64(1)
                victim = _victim_at(bb, them, target)
                flags = FLAG_CAPTURE if victim else 0
                out[count] = _encode(source, target, 0, flags, victim, piece)
                count += 1

    # --- sliders ---
    for piece in (BISHOP, ROOK, QUEEN):
        pieces = bb[base + piece - 1]
        while pieces:
            source = _lsb(pieces)
            pieces &= pieces - np.uint64(1)
            if piece == BISHOP:
                moves = _bishop_attacks(source, occupied)
            elif piece == ROOK:
                moves = _rook_attacks(source, occupied)
            else:
                moves = _bishop_attacks(source, occupied) | _rook_attacks(source, occupied)
            moves &= targets
            while moves:
                target = _lsb(moves)
                moves &= moves - np.uint64(1)
                victim = _victim_at(bb, them, target)
                flags = FLAG_CAPTURE if victim else 0
                out[count] = _encode(source, target, 0, flags, victim, piece)
                count += 1

    # --- castling ---
    # Checked here rather than by the make/test filter, because the filter only
    # looks at the final position and castling also forbids starting in check
    # or crossing an attacked square.
    if not captures_only:
        rights = st[1]
        king_square = 4 if side == WHITE else 60
        if bb[base + 5] & _bit(king_square) and not _attacked(bb, king_square, them):
            short_right = CASTLE_WK if side == WHITE else CASTLE_BK
            long_right = CASTLE_WQ if side == WHITE else CASTLE_BQ
            short_empty = _bit(king_square + 1) | _bit(king_square + 2)
            if (
                (rights & short_right)
                and not (occupied & short_empty)
                and not _attacked(bb, king_square + 1, them)
            ):
                out[count] = _encode(king_square, king_square + 2, 0, FLAG_CASTLE, 0, KING)
                count += 1
            long_empty = _bit(king_square - 1) | _bit(king_square - 2) | _bit(king_square - 3)
            if (
                (rights & long_right)
                and not (occupied & long_empty)
                and not _attacked(bb, king_square - 1, them)
            ):
                out[count] = _encode(king_square, king_square - 2, 0, FLAG_CASTLE, 0, KING)
                count += 1
    return count


@njit(cache=False)
def generate_legal(
    bb: Any,
    st: Any,
    key: Any,
    undo: Any,
    scratch: Any,
    out: Any,
    captures_only: bool,
) -> int:
    pseudo = generate_pseudo(bb, st, scratch, captures_only)
    side = st[0]
    saved = key[0]

    # Verifying every pseudo-move by make/test/unmake costs about 0.35us a move
    # and was the whole reason this layer only doubled python-chess rather than
    # beating it properly. Almost none of those moves can be illegal.
    #
    # A non-king move is only ever illegal because the piece was pinned, and a
    # pinned piece must be standing on a queen ray from its own king. So one
    # attack computation per position gives a conservative superset of the
    # pieces worth checking. En passant stays on the list because it removes two
    # pawns from one rank at once and can uncover a rook along it, which no
    # source-square test catches. In check, everything is verified.
    king_board = bb[side * 6 + 5]
    king_square = _lsb(king_board) if king_board else -1
    in_check_now = king_square >= 0 and _attacked(bb, king_square, 1 - side)
    suspect = np.uint64(0)
    if king_square >= 0 and not in_check_now:
        occupied = bb[14]
        suspect = _bishop_attacks(king_square, occupied) | _rook_attacks(king_square, occupied)

    count = 0
    for index in range(pseudo):
        move = scratch[index]
        source = move & 63
        if (
            in_check_now
            or source == king_square
            or (move & FLAG_EP)
            or (_bit(source) & suspect)
        ):
            make(bb, st, key, undo, move)
            legal = not _in_check(bb, side)
            unmake(bb, st, key, undo, saved)
            if not legal:
                continue
        out[count] = move
        count += 1
    return count


@njit(cache=False)
def compute_key(bb: Any, st: Any) -> Any:
    running = np.uint64(0)
    for index in range(12):
        pieces = bb[index]
        while pieces:
            square = _lsb(pieces)
            pieces &= pieces - np.uint64(1)
            running ^= Z_PIECE[index][square]
    if st[0] == WHITE:
        running ^= Z_SIDE
    running ^= Z_CASTLE[st[1]]
    running ^= _ep_key(bb, st)
    return running


@njit(cache=False)
def perft(
    bb: Any, st: Any, key: Any, undo: Any, scratch: Any, buffers: Any, depth: int, ply: int
) -> int:
    """Node count, jitted end to end so the figure measures generation.

    Leaves are counted by generating and taking the list length, which is the
    standard definition, so this number is comparable to published counts.
    """
    count = generate_legal(bb, st, key, undo, scratch[ply], buffers[ply], False)
    if depth <= 1:
        return count
    total = 0
    saved = key[0]
    for index in range(count):
        move = buffers[ply][index]
        make(bb, st, key, undo, move)
        total += perft(bb, st, key, undo, scratch, buffers, depth - 1, ply + 1)
        unmake(bb, st, key, undo, saved)
    return total


# --------------------------------------------------------------------------
# The contract surface. Identical to ac_board.Board, method for method.
# --------------------------------------------------------------------------

_PIECE_CHARS = ".PNBRQK"
_CHAR_PIECE = {"p": PAWN, "n": KNIGHT, "b": BISHOP, "r": ROOK, "q": QUEEN, "k": KING}
_FILES = "abcdefgh"
_PROMO_CHAR = {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}
NULL_MOVE = 0

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def square_name(square: int) -> str:
    return _FILES[square & 7] + str((square >> 3) + 1)


def move_to_uci(move: int) -> str:
    if move == NULL_MOVE:
        return "0000"
    text = square_name(move & 63) + square_name((move >> 6) & 63)
    promotion = (move >> 12) & 7
    return text + _PROMO_CHAR[promotion] if promotion else text


class BitboardBoard:
    """Bitboard implementation of the day-1 board contract."""

    __slots__ = ("_bb", "_key", "_keys", "_pool", "_scratch", "_st", "_stack", "_undo", "buffers")

    def __init__(self, fen: str = STARTING_FEN) -> None:
        self._bb = np.zeros(15, np.uint64)
        self._st = np.zeros(8, np.int64)
        self._key = np.zeros(1, np.uint64)
        self._undo = np.zeros((MAX_PLY + 8, 5), np.int64)
        self._scratch = np.zeros((MAX_PLY + 8, MAX_MOVES), np.int64)
        self._pool = np.zeros((MAX_PLY + 8, MAX_MOVES), np.int64)
        self.buffers = self._pool
        self._parse(fen)
        self._key[0] = compute_key(self._bb, self._st)
        self._keys: list[int] = [int(self._key[0])]
        self._stack: list[int] = []

    # ---- FEN ----------------------------------------------------------

    def _parse(self, fen: str) -> None:
        parts = fen.split()
        placement = parts[0]
        square = 56
        for char in placement:
            if char == "/":
                square -= 16
            elif char.isdigit():
                square += int(char)
            else:
                colour = WHITE if char.isupper() else BLACK
                _place_py(self._bb, colour, _CHAR_PIECE[char.lower()], square)
                square += 1
        self._st[0] = WHITE if parts[1] == "w" else BLACK
        rights = 0
        castling = parts[2] if len(parts) > 2 else "-"
        if "K" in castling:
            rights |= CASTLE_WK
        if "Q" in castling:
            rights |= CASTLE_WQ
        if "k" in castling:
            rights |= CASTLE_BK
        if "q" in castling:
            rights |= CASTLE_BQ
        # Drop a right whose rook or king is not actually home. python-chess
        # normalises the same way and the hash has to agree with it.
        if not (self._bb[ROOK - 1] & (np.uint64(1) << np.uint64(7))):
            rights &= ~CASTLE_WK
        if not (self._bb[ROOK - 1] & np.uint64(1)):
            rights &= ~CASTLE_WQ
        if not (self._bb[6 + ROOK - 1] & (np.uint64(1) << np.uint64(63))):
            rights &= ~CASTLE_BK
        if not (self._bb[6 + ROOK - 1] & (np.uint64(1) << np.uint64(56))):
            rights &= ~CASTLE_BQ
        if not (self._bb[KING - 1] & (np.uint64(1) << np.uint64(4))):
            rights &= ~(CASTLE_WK | CASTLE_WQ)
        if not (self._bb[6 + KING - 1] & (np.uint64(1) << np.uint64(60))):
            rights &= ~(CASTLE_BK | CASTLE_BQ)
        self._st[1] = rights
        target = parts[3] if len(parts) > 3 else "-"
        self._st[2] = -1 if target == "-" else _FILES.index(target[0]) + (int(target[1]) - 1) * 8
        self._st[3] = int(parts[4]) if len(parts) > 4 else 0
        self._st[4] = 0

    def fen(self) -> str:
        rows = []
        for rank in range(7, -1, -1):
            row, empty = "", 0
            for file in range(8):
                piece = self.piece_at(rank * 8 + file)
                if piece == 0:
                    empty += 1
                    continue
                if empty:
                    row += str(empty)
                    empty = 0
                char = _PIECE_CHARS[abs(piece)]
                row += char if piece > 0 else char.lower()
            if empty:
                row += str(empty)
            rows.append(row)
        rights = ""
        castling_chars = (
            (CASTLE_WK, "K"), (CASTLE_WQ, "Q"), (CASTLE_BK, "k"), (CASTLE_BQ, "q"),
        )
        for bit_value, char in castling_chars:
            if self._st[1] & bit_value:
                rights += char
        target = "-"
        if self._st[2] >= 0 and _ep_is_legal(self._bb, self._st):
            target = square_name(int(self._st[2]))
        side = "w" if self._st[0] == WHITE else "b"
        move_number = len(self._keys) // 2 + 1
        return (
            f"{'/'.join(rows)} {side} {rights or '-'} {target} "
            f"{int(self._st[3])} {move_number}"
        )

    # ---- identity -----------------------------------------------------

    @property
    def turn(self) -> bool:
        return bool(self._st[0] == WHITE)

    @property
    def key(self) -> int:
        return int(self._key[0])

    @property
    def halfmove_clock(self) -> int:
        return int(self._st[3])

    def ply(self) -> int:
        return int(self._st[4])

    def peek_native(self) -> Any:
        import chess

        return chess.Board(self.fen())

    def verify_key(self) -> bool:
        return int(self._key[0]) == int(compute_key(self._bb, self._st))

    # ---- generation ---------------------------------------------------

    def generate(self, ply: int) -> int:
        return int(
            generate_legal(
                self._bb, self._st, self._key, self._undo,
                self._scratch[ply], self._pool[ply], False,
            )
        )

    def generate_captures(self, ply: int) -> int:
        return int(
            generate_legal(
                self._bb, self._st, self._key, self._undo,
                self._scratch[ply], self._pool[ply], True,
            )
        )

    def move_from_uci(self, uci: str) -> int:
        scratch = MAX_PLY + 1
        count = self.generate(scratch)
        for index in range(count):
            move = int(self._pool[scratch][index])
            if move_to_uci(move) == uci:
                return move
        return NULL_MOVE

    # ---- make / unmake -------------------------------------------------

    def make(self, move: int) -> None:
        self._stack.append(int(self._key[0]))
        make(self._bb, self._st, self._key, self._undo, move)
        self._keys.append(int(self._key[0]))

    def unmake(self) -> None:
        prior = self._stack.pop()
        unmake(self._bb, self._st, self._key, self._undo, np.uint64(prior))
        self._keys.pop()

    def make_null(self) -> None:
        """Unused by the day-2 search, but part of the contract, so it exists.

        Note for whoever adds null move pruning: a null flips the side to move
        without a ply of history, so is_repetition() is not meaningful across
        one. Reset the scan window rather than trusting it.
        """
        self._stack.append(int(self._key[0]))
        depth = int(self._st[4])
        self._undo[depth, 0] = self._st[1]
        self._undo[depth, 1] = self._st[2]
        self._undo[depth, 2] = self._st[3]
        self._undo[depth, 3] = 0
        self._undo[depth, 4] = 0
        running = int(self._key[0]) ^ int(Z_SIDE) ^ int(_ep_key(self._bb, self._st))
        self._st[2] = -1
        self._st[0] = 1 - self._st[0]
        self._st[4] = depth + 1
        self._key[0] = np.uint64(running)
        self._keys.append(int(self._key[0]))

    def unmake_null(self) -> None:
        depth = int(self._st[4]) - 1
        self._st[1] = self._undo[depth, 0]
        self._st[2] = self._undo[depth, 1]
        self._st[3] = self._undo[depth, 2]
        self._st[0] = 1 - self._st[0]
        self._st[4] = depth
        self._key[0] = np.uint64(self._stack.pop())
        self._keys.pop()

    # ---- queries -------------------------------------------------------

    def in_check(self) -> bool:
        return bool(_in_check(self._bb, int(self._st[0])))

    def piece_at(self, square: int) -> int:
        mask = np.uint64(1) << np.uint64(square)
        for index in range(12):
            if self._bb[index] & mask:
                piece = (index % 6) + 1
                return piece if index < 6 else -piece
        return 0

    def pieces(self, piece_type: int, white: bool) -> int:
        return int(self._bb[(0 if white else 6) + piece_type - 1])

    def occupancy(self, white: bool) -> int:
        return int(self._bb[12 if white else 13])

    def phase(self) -> int:
        total = 0
        for colour in (0, 6):
            total += _popcount_py(self._bb[colour + 1]) + _popcount_py(self._bb[colour + 2])
            total += 2 * _popcount_py(self._bb[colour + 3])
            total += 4 * _popcount_py(self._bb[colour + 4])
        return total if total < 24 else 24

    def is_repetition(self, count: int = 2) -> bool:
        keys = self._keys
        target = keys[-1]
        window = int(self._st[3])
        index = len(keys) - 1
        hits = 1
        back = 2
        while back <= window:
            index -= 2
            if index < 0:
                break
            if keys[index] == target:
                hits += 1
                if hits >= count:
                    return True
            back += 2
        return False

    def is_draw(self) -> bool:
        if self._st[3] >= 100:
            return True
        if self._insufficient():
            return True
        return self.is_repetition(2)

    def _insufficient(self) -> bool:
        if self._bb[0] or self._bb[6] or self._bb[3] or self._bb[9] or self._bb[4] or self._bb[10]:
            return False  # any pawn, rook or queen is enough
        minors = _popcount_py(self._bb[1]) + _popcount_py(self._bb[2])
        minors += _popcount_py(self._bb[7]) + _popcount_py(self._bb[8])
        return bool(minors <= 1)


def _place_py(bb: Any, colour: int, piece: int, square: int) -> None:
    mask = np.uint64(1) << np.uint64(square)
    bb[colour * 6 + piece - 1] |= mask
    bb[12 + colour] |= mask
    bb[14] |= mask


def _popcount_py(bits: Any) -> int:
    return int(bin(int(bits)).count("1"))


def _warmup_bitboard() -> None:
    """Compile every kernel at import, inside the 60s init budget.

    numba compiles per signature, so this has to call the real entry points with
    the real argument types. A kernel first compiled on the clock is a kernel
    compiled during a rated game.
    """
    board = BitboardBoard()
    board.generate(0)
    board.generate_captures(0)
    move = board.move_from_uci("e2e4")
    board.make(move)
    board.in_check()
    board.verify_key()
    board.unmake()
    perft(
        board._bb, board._st, board._key, board._undo,
        board._scratch, board._pool, 2, 0,
    )


# ==========================================================================
# ac_board.py
# ==========================================================================

import os
import random
from collections.abc import MutableSequence
from typing import Final

import chess

MAX_PLY: Final = 128
MAX_MOVES: Final = 256  # a legal position tops out at 218

NULL_MOVE: Final = 0

FLAG_CAPTURE: Final = 1 << 15
FLAG_EP: Final = 1 << 16
FLAG_CASTLE: Final = 1 << 17

_PROMO_SHIFT: Final = 12
_VICTIM_SHIFT: Final = 18
_MOVER_SHIFT: Final = 21


# Module-level accessors rather than methods: they are called once per move per
# node and an attribute lookup on a hot path is not free in CPython.
def move_from(move: int) -> int:
    return move & 0x3F


def move_to(move: int) -> int:
    return (move >> 6) & 0x3F


def move_promotion(move: int) -> int:
    return (move >> _PROMO_SHIFT) & 0x7


def move_victim(move: int) -> int:
    return (move >> _VICTIM_SHIFT) & 0x7


def move_piece(move: int) -> int:
    return (move >> _MOVER_SHIFT) & 0x7


def move_is_capture(move: int) -> bool:
    return bool(move & FLAG_CAPTURE)


def move_is_promotion(move: int) -> bool:
    return bool(move & 0x7000)


_PROMO_CHAR: Final = {chess.KNIGHT: "n", chess.BISHOP: "b", chess.ROOK: "r", chess.QUEEN: "q"}


def move_to_uci(move: int) -> str:
    if move == NULL_MOVE:
        return "0000"
    text = chess.SQUARE_NAMES[move & 0x3F] + chess.SQUARE_NAMES[(move >> 6) & 0x3F]
    promotion = (move >> _PROMO_SHIFT) & 0x7
    return text + _PROMO_CHAR[promotion] if promotion else text


# --------------------------------------------------------------------------
# Zobrist keys, fixed at import from a fixed seed so two processes agree.
# --------------------------------------------------------------------------

_rng = random.Random(0x9E3779B97F4A7C15)
_PIECE_KEYS: Final = tuple(
    tuple(tuple(_rng.getrandbits(64) for _ in range(64)) for _ in range(7)) for _ in range(2)
)
_SIDE_KEY: Final = _rng.getrandbits(64)
_CASTLE_KEYS: Final = tuple(_rng.getrandbits(64) for _ in range(16))
_EP_KEYS: Final = tuple(_rng.getrandbits(64) for _ in range(8))

_WHITE_INDEX: Final = 1
_BLACK_INDEX: Final = 0


def _castle_code(rights: int) -> int:
    code = 0
    if rights & chess.BB_H1:
        code |= 1
    if rights & chess.BB_A1:
        code |= 2
    if rights & chess.BB_H8:
        code |= 4
    if rights & chess.BB_A8:
        code |= 8
    return code


def _key_at(board: chess.Board, square: int) -> int:
    """Zobrist key of whatever stands on this square, 0 if it is empty."""
    piece_type = board.piece_type_at(square)
    if piece_type is None:
        return 0
    white = bool((board.occupied_co[chess.WHITE] >> square) & 1)
    return _PIECE_KEYS[_WHITE_INDEX if white else _BLACK_INDEX][piece_type][square]


class ChessLibBoard:
    """Day 1: a thin, slow wrapper over python-chess. Correctness now, speed day 3."""

    __slots__ = ("_board", "_key", "_keys", "_undo", "buffers")

    def __init__(self, fen: str = chess.STARTING_FEN) -> None:
        self._board = chess.Board(fen)
        # push() normalises castling rights to clean_castling_rights(); do the
        # same once at the root so the incremental hash and a from-scratch
        # recomputation can never disagree about a phantom right.
        self._board.castling_rights = self._board.clean_castling_rights()
        self.buffers: list[MutableSequence[int]] = []
        self._key = self._compute_key()
        self._keys: list[int] = [self._key]  # every position seen, current one last
        self._undo: list[int] = []

    # ---- identity -------------------------------------------------------

    @property
    def turn(self) -> bool:
        return self._board.turn

    @property
    def key(self) -> int:
        return self._key

    @property
    def halfmove_clock(self) -> int:
        return self._board.halfmove_clock

    def fen(self) -> str:
        return self._board.fen()

    def ply(self) -> int:
        return len(self._board.move_stack)

    def peek_native(self) -> chess.Board:
        """The underlying python-chess board. Test and harness code only.

        Nothing in ac_search may call this: it is the one thing that will not
        exist after the day-3 swap.
        """
        return self._board

    # ---- hashing --------------------------------------------------------

    def _ep_key(self) -> int:
        square = self._board.ep_square
        if square is None:
            return 0
        if not self._board.has_legal_en_passant():
            return 0
        return _EP_KEYS[square & 7]

    def _compute_key(self) -> int:
        board = self._board
        key = 0
        for square, piece in board.piece_map().items():
            index = _WHITE_INDEX if piece.color else _BLACK_INDEX
            key ^= _PIECE_KEYS[index][piece.piece_type][square]
        if board.turn == chess.WHITE:
            key ^= _SIDE_KEY
        key ^= _CASTLE_KEYS[_castle_code(board.clean_castling_rights())]
        key ^= self._ep_key()
        return key

    def verify_key(self) -> bool:
        """Incremental hash still matches a from-scratch recomputation.

        perft --check-hash calls this on every node. That test is what makes a
        hand-rolled incremental Zobrist safe to rely on.
        """
        return self._key == self._compute_key()

    # ---- move generation ------------------------------------------------

    def _buffer(self, ply: int) -> MutableSequence[int]:
        buffers = self.buffers
        while len(buffers) <= ply:
            buffers.append([0] * MAX_MOVES)
        return buffers[ply]

    def generate(self, ply: int) -> int:
        buffer = self._buffer(ply)
        board = self._board
        ep_square = board.ep_square
        count = 0
        for move in board.generate_legal_moves():
            source = move.from_square
            target = move.to_square
            mover = board.piece_type_at(source)
            victim = board.piece_type_at(target)
            flags = 0
            if victim is not None:
                flags = FLAG_CAPTURE
            elif mover == chess.PAWN and target == ep_square:
                flags = FLAG_CAPTURE | FLAG_EP
                victim = chess.PAWN
            elif mover == chess.KING and abs((target & 7) - (source & 7)) == 2:
                flags = FLAG_CASTLE
            buffer[count] = (
                source
                | (target << 6)
                | ((move.promotion or 0) << _PROMO_SHIFT)
                | flags
                | ((victim or 0) << _VICTIM_SHIFT)
                | ((mover or 0) << _MOVER_SHIFT)
            )
            count += 1
        return count

    def generate_captures(self, ply: int) -> int:
        buffer = self._buffer(ply)
        board = self._board
        count = 0
        for move in board.generate_legal_captures():
            source = move.from_square
            target = move.to_square
            mover = board.piece_type_at(source)
            victim = board.piece_type_at(target)
            flags = FLAG_CAPTURE
            if victim is None:  # generate_legal_captures only emits ep here
                flags |= FLAG_EP
                victim = chess.PAWN
            buffer[count] = (
                source
                | (target << 6)
                | ((move.promotion or 0) << _PROMO_SHIFT)
                | flags
                | (victim << _VICTIM_SHIFT)
                | ((mover or 0) << _MOVER_SHIFT)
            )
            count += 1
        return count

    def move_from_uci(self, uci: str) -> int:
        """Encoded move matching this UCI string, or NULL_MOVE if it is not legal."""
        count = self.generate(MAX_PLY + 1)  # a scratch buffer no search ply uses
        buffer = self.buffers[MAX_PLY + 1]
        for index in range(count):
            if move_to_uci(buffer[index]) == uci:
                return buffer[index]
        return NULL_MOVE

    # ---- make / unmake --------------------------------------------------

    def make(self, move: int) -> None:
        board = self._board
        source = move & 0x3F
        target = (move >> 6) & 0x3F
        promotion = (move >> _PROMO_SHIFT) & 0x7

        self._undo.append(self._key)
        key = self._key ^ _SIDE_KEY
        key ^= _CASTLE_KEYS[_castle_code(board.castling_rights)] ^ self._ep_key()

        # XOR out whatever stands on every square this move can touch, push,
        # then XOR back in whatever stands there afterwards. Deriving the square
        # set instead of enumerating move types is what keeps this short enough
        # to be obviously right.
        key ^= _key_at(board, source) ^ _key_at(board, target)
        rook_from = rook_to = -1
        ep_victim = -1
        if move & FLAG_CASTLE:
            if target > source:
                rook_from, rook_to = target + 1, target - 1
            else:
                rook_from, rook_to = target - 2, target + 1
            key ^= _key_at(board, rook_from) ^ _key_at(board, rook_to)
        elif move & FLAG_EP:
            ep_victim = target - 8 if board.turn == chess.WHITE else target + 8
            key ^= _key_at(board, ep_victim)

        board.push(chess.Move(source, target, promotion or None))

        key ^= _key_at(board, source) ^ _key_at(board, target)
        if rook_from >= 0:
            key ^= _key_at(board, rook_from) ^ _key_at(board, rook_to)
        elif ep_victim >= 0:
            key ^= _key_at(board, ep_victim)
        key ^= _CASTLE_KEYS[_castle_code(board.castling_rights)] ^ self._ep_key()

        self._key = key
        self._keys.append(key)

    def unmake(self) -> None:
        self._board.pop()
        self._keys.pop()
        self._key = self._undo.pop()

    def make_null(self) -> None:
        """Unused by the day-1 search. Note for whoever adds null move pruning:
        a null flips side to move without a ply of history, so is_repetition()
        is not meaningful across one. Reset the scan window, do not trust it."""
        board = self._board
        self._undo.append(self._key)
        key = self._key ^ _SIDE_KEY ^ self._ep_key()
        board.push(chess.Move.null())
        key ^= self._ep_key()
        self._key = key
        self._keys.append(key)

    def unmake_null(self) -> None:
        self.unmake()

    # ---- queries --------------------------------------------------------

    def in_check(self) -> bool:
        return self._board.is_check()

    def piece_at(self, square: int) -> int:
        piece_type = self._board.piece_type_at(square)
        if piece_type is None:
            return 0
        white = bool((self._board.occupied_co[chess.WHITE] >> square) & 1)
        return piece_type if white else -piece_type

    def pieces(self, piece_type: int, white: bool) -> int:
        return self._board.pieces_mask(piece_type, white)

    def occupancy(self, white: bool) -> int:
        return self._board.occupied_co[white]

    def phase(self) -> int:
        board = self._board
        phase = (
            chess.popcount(board.knights | board.bishops)
            + 2 * chess.popcount(board.rooks)
            + 4 * chess.popcount(board.queens)
        )
        return phase if phase < 24 else 24

    def is_repetition(self, count: int = 2) -> bool:
        """Has this exact position stood on the board `count` times, this one included?

        The search asks with count=2, not 3. The referee calls
        outcome(claim_draw=True), so a threefold is claimed against us the
        instant it becomes available; we want to see it coming a visit early.
        """
        keys = self._keys
        target = self._key
        window = self._board.halfmove_clock  # nothing repeats across a pawn move
        index = len(keys) - 1
        hits = 1
        back = 2
        while back <= window:
            index -= 2
            if index < 0:
                break
            if keys[index] == target:
                hits += 1
                if hits >= count:
                    return True
            back += 2
        return False

    def is_draw(self) -> bool:
        board = self._board
        if board.halfmove_clock >= 100:
            return True
        if board.is_insufficient_material():
            return True
        return self.is_repetition(2)


# --------------------------------------------------------------------------
# Backend selection
#
# The contract above is the python-chess implementation, kept because it is the
# reference the bitboard layer is differentially tested against and because it
# is the fallback if numba ever fails to compile on their image. AC_BACKEND=chess
# selects it; anything else, including unset, gets the jitted one.
#
# Nothing above this line changes when the backend does. That was the point of
# freezing the interface on day 1.
# --------------------------------------------------------------------------


# Rebinding a module-level class is something mypy has no way to model, so the
# ignores below are on the two lines that do it and nowhere else.
if os.environ.get("AC_BACKEND", "bitboard").lower() != "chess":
    Board = BitboardBoard  # type: ignore[assignment,misc]


# ==========================================================================
# ac_eval.py
# ==========================================================================

from typing import Any, Final

import chess
import numpy as np


# Centipawns, midgame and endgame. Pawns and rooks gain as the board empties.
PIECE_MG: Final = (0, 100, 320, 330, 500, 900, 0)
PIECE_EG: Final = (0, 120, 320, 340, 550, 950, 0)

TOTAL_PHASE: Final = 24


def _flatten(rows: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Tables are written rank 8 first, the way a board looks. Index by square."""
    flat: list[int] = []
    for rank in range(8):
        flat.extend(rows[7 - rank])
    return tuple(flat)


_PAWN_MG = _flatten((
    (0, 0, 0, 0, 0, 0, 0, 0),
    (60, 60, 60, 60, 60, 60, 60, 60),
    (20, 25, 35, 45, 45, 35, 25, 20),
    (10, 12, 20, 32, 32, 20, 12, 10),
    (4, 6, 12, 26, 26, 12, 6, 4),
    (2, 4, 6, 10, 10, 4, 2, 2),
    (0, 2, 2, -10, -10, 4, 4, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
))
_PAWN_EG = _flatten((
    (0, 0, 0, 0, 0, 0, 0, 0),
    (100, 100, 100, 100, 100, 100, 100, 100),
    (55, 55, 50, 50, 50, 50, 55, 55),
    (28, 28, 24, 22, 22, 24, 28, 28),
    (12, 12, 10, 8, 8, 10, 12, 12),
    (4, 4, 2, 2, 2, 2, 4, 4),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
))
_KNIGHT_MG = _flatten((
    (-50, -38, -28, -25, -25, -28, -38, -50),
    (-38, -18, 0, 4, 4, 0, -18, -38),
    (-28, 4, 16, 20, 20, 16, 4, -28),
    (-25, 8, 20, 26, 26, 20, 8, -25),
    (-25, 6, 18, 24, 24, 18, 6, -25),
    (-28, 2, 14, 18, 18, 14, 2, -28),
    (-38, -18, 0, 6, 6, 0, -18, -38),
    (-50, -30, -24, -18, -18, -24, -30, -50),
))
_KNIGHT_EG = _flatten((
    (-40, -30, -20, -18, -18, -20, -30, -40),
    (-30, -12, 0, 4, 4, 0, -12, -30),
    (-20, 4, 12, 16, 16, 12, 4, -20),
    (-18, 6, 16, 20, 20, 16, 6, -18),
    (-18, 6, 16, 20, 20, 16, 6, -18),
    (-20, 2, 12, 16, 16, 12, 2, -20),
    (-30, -12, 0, 4, 4, 0, -12, -30),
    (-40, -30, -20, -18, -18, -20, -30, -40),
))
_BISHOP_MG = _flatten((
    (-18, -10, -8, -8, -8, -8, -10, -18),
    (-10, 6, 2, 0, 0, 2, 6, -10),
    (-8, 8, 10, 10, 10, 10, 8, -8),
    (-8, 2, 12, 16, 16, 12, 2, -8),
    (-8, 4, 10, 16, 16, 10, 4, -8),
    (-8, 10, 12, 10, 10, 12, 10, -8),
    (-10, 12, 4, 4, 4, 4, 12, -10),
    (-18, -10, -12, -8, -8, -12, -10, -18),
))
_BISHOP_EG = _flatten((
    (-12, -6, -4, -2, -2, -4, -6, -12),
    (-6, 2, 2, 2, 2, 2, 2, -6),
    (-4, 2, 6, 8, 8, 6, 2, -4),
    (-2, 4, 8, 10, 10, 8, 4, -2),
    (-2, 4, 8, 10, 10, 8, 4, -2),
    (-4, 2, 6, 8, 8, 6, 2, -4),
    (-6, 2, 2, 2, 2, 2, 2, -6),
    (-12, -6, -4, -2, -2, -4, -6, -12),
))
_ROOK_MG = _flatten((
    (0, 2, 4, 6, 6, 4, 2, 0),
    (14, 18, 18, 20, 20, 18, 18, 14),
    (-4, 0, 2, 4, 4, 2, 0, -4),
    (-6, -2, 0, 2, 2, 0, -2, -6),
    (-6, -2, 0, 2, 2, 0, -2, -6),
    (-6, -2, 0, 2, 2, 0, -2, -6),
    (-6, -2, 0, 2, 2, 0, -2, -6),
    (-4, 0, 2, 8, 8, 4, 0, -4),
))
_ROOK_EG = _flatten((
    (8, 8, 8, 8, 8, 8, 8, 8),
    (10, 10, 10, 10, 10, 10, 10, 10),
    (4, 4, 4, 4, 4, 4, 4, 4),
    (2, 2, 2, 2, 2, 2, 2, 2),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (-2, -2, -2, -2, -2, -2, -2, -2),
    (-2, -2, -2, -2, -2, -2, -2, -2),
    (0, 0, 0, 0, 0, 0, 0, 0),
))
_QUEEN_MG = _flatten((
    (-16, -8, -8, -4, -4, -8, -8, -16),
    (-8, 0, 2, 0, 0, 2, 0, -8),
    (-8, 2, 4, 4, 4, 4, 2, -8),
    (-4, 0, 4, 6, 6, 4, 0, -4),
    (-4, 0, 4, 6, 6, 4, 0, -4),
    (-8, 2, 4, 4, 4, 4, 2, -8),
    (-8, 0, 2, 0, 0, 2, 0, -8),
    (-16, -8, -8, -4, -4, -8, -8, -16),
))
_QUEEN_EG = _flatten((
    (-20, -12, -8, -4, -4, -8, -12, -20),
    (-12, -4, 0, 4, 4, 0, -4, -12),
    (-8, 0, 6, 10, 10, 6, 0, -8),
    (-4, 4, 10, 14, 14, 10, 4, -4),
    (-4, 4, 10, 14, 14, 10, 4, -4),
    (-8, 0, 6, 10, 10, 6, 0, -8),
    (-12, -4, 0, 4, 4, 0, -4, -12),
    (-20, -12, -8, -4, -4, -8, -12, -20),
))
_KING_MG = _flatten((
    (-60, -70, -70, -80, -80, -70, -70, -60),
    (-60, -70, -70, -80, -80, -70, -70, -60),
    (-60, -70, -70, -80, -80, -70, -70, -60),
    (-50, -60, -60, -70, -70, -60, -60, -50),
    (-30, -40, -40, -50, -50, -40, -40, -30),
    (-14, -20, -20, -20, -20, -20, -20, -14),
    (14, 14, -6, -8, -8, -6, 14, 14),
    (18, 30, 8, 0, 0, 8, 30, 18),
))
_KING_EG = _flatten((
    (-56, -34, -22, -16, -16, -22, -34, -56),
    (-24, -8, 6, 14, 14, 6, -8, -24),
    (-14, 8, 24, 30, 30, 24, 8, -14),
    (-14, 10, 30, 38, 38, 30, 10, -14),
    (-16, 8, 28, 36, 36, 28, 8, -16),
    (-20, 2, 18, 24, 24, 18, 2, -20),
    (-30, -14, 2, 6, 6, 2, -14, -30),
    (-56, -38, -28, -22, -22, -28, -38, -56),
))

_EMPTY: Final = (0,) * 64
MG_TABLE: Final = (_EMPTY, _PAWN_MG, _KNIGHT_MG, _BISHOP_MG, _ROOK_MG, _QUEEN_MG, _KING_MG)
EG_TABLE: Final = (_EMPTY, _PAWN_EG, _KNIGHT_EG, _BISHOP_EG, _ROOK_EG, _QUEEN_EG, _KING_EG)


def _taper(midgame: int, endgame: int, phase: int) -> int:
    """Blend the two scores.

    Truncation toward zero, not floor division. Floor division rounds negatives
    the wrong way, which makes a position and its mirror differ by one
    centipawn; that bug was live in bot1 and hit 198 of 399 test positions.
    """
    total = midgame * phase + endgame * (TOTAL_PHASE - phase)
    return total // TOTAL_PHASE if total >= 0 else -((-total) // TOTAL_PHASE)


def evaluate(board: Board) -> int:
    """Centipawns from the point of view of the side to move."""
    midgame = 0
    endgame = 0

    for white in (True, False):
        sign = 1 if white else -1
        for piece_type in range(1, 7):
            bits = board.pieces(piece_type, white)
            if not bits:
                continue
            mg_table = MG_TABLE[piece_type]
            eg_table = EG_TABLE[piece_type]
            mg_value = PIECE_MG[piece_type]
            eg_value = PIECE_EG[piece_type]
            while bits:
                square = (bits & -bits).bit_length() - 1
                bits &= bits - 1
                # Black reads the same table from the far side of the board.
                index = square if white else square ^ 56
                midgame += sign * (mg_value + mg_table[index])
                endgame += sign * (eg_value + eg_table[index])

    score = _taper(midgame, endgame, board.phase())
    return score if board.turn == chess.WHITE else -score


# --------------------------------------------------------------------------
# Jitted kernel
#
# Identical arithmetic to evaluate() above, compiled. Not a different
# evaluation: the tables, the material values and the truncating taper are the
# same, and tools/regress.py asserts the two agree exactly across the opening
# set. It exists because at 21.5us a call the interpreted version was the entire
# cost of a node once the bitboard movegen landed, and no amount of faster move
# generation shows up in depth while that is true.
# --------------------------------------------------------------------------

_MG_FLAT = np.array([list(table) for table in MG_TABLE], dtype=np.int32)
_EG_FLAT = np.array([list(table) for table in EG_TABLE], dtype=np.int32)
_MG_VALUE = np.array(PIECE_MG, dtype=np.int32)
_EG_VALUE = np.array(PIECE_EG, dtype=np.int32)

try:
    from numba import njit

    @njit(cache=False)
    def _evaluate_kernel(bb: Any, st: Any) -> Any:
        midgame = 0
        endgame = 0
        phase = 0
        for colour in range(2):
            sign = 1 if colour == 0 else -1
            for piece in range(1, 7):
                bits = bb[colour * 6 + piece - 1]
                while bits:
                    square = 0
                    low = bits & (np.uint64(0) - bits)
                    if low >> np.uint64(32):
                        low >>= np.uint64(32)
                        square += 32
                    if low >> np.uint64(16):
                        low >>= np.uint64(16)
                        square += 16
                    if low >> np.uint64(8):
                        low >>= np.uint64(8)
                        square += 8
                    if low >> np.uint64(4):
                        low >>= np.uint64(4)
                        square += 4
                    if low >> np.uint64(2):
                        low >>= np.uint64(2)
                        square += 2
                    if low >> np.uint64(1):
                        square += 1
                    bits &= bits - np.uint64(1)
                    index = square if colour == 0 else square ^ 56
                    midgame += sign * (_MG_VALUE[piece] + _MG_FLAT[piece][index])
                    endgame += sign * (_EG_VALUE[piece] + _EG_FLAT[piece][index])
                    if piece == 2 or piece == 3:
                        phase += 1
                    elif piece == 4:
                        phase += 2
                    elif piece == 5:
                        phase += 4
        if phase > 24:
            phase = 24
        total = midgame * phase + endgame * (24 - phase)
        # Truncate toward zero, not floor: floor breaks mirror symmetry by a
        # centipawn on negative scores. That is regression checklist bug #5.
        score = total // 24 if total >= 0 else -((-total) // 24)
        return score if st[0] == 0 else -score

    _HAVE_KERNEL = True
except Exception:  # numba missing: the interpreted path is still correct
    _HAVE_KERNEL = False


_evaluate_python = evaluate


def evaluate(board: Board) -> int:  # type: ignore[no-redef]
    """Centipawns from the point of view of the side to move."""
    if _HAVE_KERNEL:
        bitboards = getattr(board, "_bb", None)
        state = getattr(board, "_st", None)
        if bitboards is not None and state is not None:
            return int(_evaluate_kernel(bitboards, state))
    return _evaluate_python(board)


def _warmup_eval() -> None:
    if _HAVE_KERNEL:
        board = BitboardBoard()
        _evaluate_kernel(board._bb, board._st)


# ==========================================================================
# ac_search.py
# ==========================================================================

import time
from collections.abc import MutableSequence
from dataclasses import dataclass, field
from typing import Final


INFINITY: Final = 1 << 24
MATE: Final = 32_000
MATE_BOUND: Final = 31_000  # anything past this is a mate score and needs ply adjustment

TT_EXACT: Final = 0
TT_LOWER: Final = 1
TT_UPPER: Final = 2

# The process lives for one whole game, so the table is worth keeping between
# moves. Entry is (depth, flag, score, move). Cleared wholesale when it gets
# too big, which is crude; a replacement scheme is a day-2 item.
TT_MAX_ENTRIES: Final = 600_000
_TT: dict[int, tuple[int, int, int, int]] = {}

# Cost of the ordering key for a capture: most valuable victim, cheapest attacker.
_MVV_LVA_BASE: Final = 1 << 20
_TT_MOVE_SCORE: Final = 1 << 24
_PIECE_ORDER: Final = (0, 1, 3, 3, 5, 9, 0)


class _Timeout(Exception):
    """Raised inside the search when the budget runs out."""


@dataclass(frozen=True)
class Limits:
    """A node budget or a time budget. Never both.

    soft_ms is part of the time budget, not a second kind of limit: time_ms is
    where the search is abandoned mid-iteration, soft_ms is the point past which
    starting another iteration is a bad bet. Each extra depth costs several
    times the one before, so a depth begun late finishes on the hard limit
    instead, and the whole budget buys nothing. Defaults to half of time_ms.
    """

    time_ms: float | None = None
    nodes: int | None = None
    soft_ms: float | None = None
    max_depth: int = 64

    def __post_init__(self) -> None:
        if (self.time_ms is None) == (self.nodes is None):
            raise ValueError("limits take exactly one of time_ms or nodes")
        if self.soft_ms is not None and self.time_ms is None:
            raise ValueError("soft_ms only means anything alongside time_ms")


@dataclass
class Info:
    depth: int = 0
    nodes: int = 0
    score: int = 0
    mate: int | None = None
    time_ms: float = 0.0
    pv: list[str] = field(default_factory=list)


def _to_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


def _from_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


class _Searcher:
    __slots__ = ("_root_ply", "_scores", "board", "deadline", "node_limit", "nodes")

    def __init__(self, board: Board, limits: Limits) -> None:
        self.board = board
        self.node_limit = limits.nodes
        self.deadline = (
            time.monotonic() + limits.time_ms / 1000.0 if limits.time_ms is not None else None
        )
        self.nodes = 0
        self._scores: list[list[int]] = []
        self._root_ply = board.ply()

    def _score_buffer(self, ply: int) -> list[int]:
        while len(self._scores) <= ply:
            self._scores.append([0] * 256)
        return self._scores[ply]

    def _tick(self) -> None:
        """Node budgets are checked exactly, so fixed-node runs are reproducible.
        The clock is read every 256 nodes: often enough that the overshoot past
        the hard limit stays inside a few milliseconds even when the whole
        budget is only a few thousand nodes, rare enough that the read costs
        nothing measurable."""
        self.nodes += 1
        if self.node_limit is not None and self.nodes >= self.node_limit:
            raise _Timeout
        # `and` short-circuits, so monotonic() is still only read once every 256
        # nodes. This is the hottest line in the engine; the shape matters.
        if (
            self.deadline is not None
            and not (self.nodes & 255)
            and time.monotonic() >= self.deadline
        ):
            raise _Timeout

    def _order(
        self,
        buffer: MutableSequence[int],
        scores: MutableSequence[int],
        count: int,
        ply: int,
        tt_move: int,
    ) -> None:
        for index in range(count):
            move = buffer[index]
            if move == tt_move:
                scores[index] = _TT_MOVE_SCORE
                continue
            victim = move_victim(move)
            if victim:
                scores[index] = (
                    _MVV_LVA_BASE + _PIECE_ORDER[victim] * 16 - _PIECE_ORDER[move_piece(move)]
                )
            else:
                scores[index] = _PIECE_ORDER[move_promotion(move)] * 64

    @staticmethod
    def _pick(
        buffer: MutableSequence[int], scores: MutableSequence[int], count: int, index: int
    ) -> int:
        """Selection sort one move at a time: most nodes cut off long before the
        tail of the list is reached, so sorting all of it is wasted work."""
        best = index
        for other in range(index + 1, count):
            if scores[other] > scores[best]:
                best = other
        if best != index:
            buffer[index], buffer[best] = buffer[best], buffer[index]
            scores[index], scores[best] = scores[best], scores[index]
        return buffer[index]

    # ---- quiescence -----------------------------------------------------

    def _quiescence(self, alpha: int, beta: int, ply: int) -> int:
        self._tick()
        board = self.board

        if ply >= MAX_PLY - 1:
            return evaluate(board)

        in_check = board.in_check()
        if in_check:
            # No stand-pat: a side in check has no right to decline to move, and
            # pretending otherwise scores a mate at the leaf as plain material.
            count = board.generate(ply)
            if count == 0:
                return -MATE + ply
            best = -INFINITY
        else:
            best = evaluate(board)
            if best >= beta:
                return best
            if best > alpha:
                alpha = best
            count = board.generate_captures(ply)
            if count == 0:
                return best

        buffer = board.buffers[ply]
        scores = self._score_buffer(ply)
        self._order(buffer, scores, count, ply, NULL_MOVE)

        for index in range(count):
            move = self._pick(buffer, scores, count, index)
            board.make(move)
            try:
                score = -self._quiescence(-beta, -alpha, ply + 1)
            finally:
                board.unmake()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    # ---- main search ----------------------------------------------------

    def _negamax(self, depth: int, alpha: int, beta: int, ply: int) -> int:
        self._tick()
        board = self.board

        if ply > 0 and board.is_draw():
            return 0

        key = board.key
        tt_move = NULL_MOVE
        entry = _TT.get(key)
        if entry is not None:
            stored_depth, flag, stored_score, tt_move = entry
            usable = ply > 0 and stored_depth >= depth
            if usable:
                score = _from_tt(stored_score, ply)
                if flag == TT_EXACT:
                    return score
                if flag == TT_LOWER and score >= beta:
                    return score
                if flag == TT_UPPER and score <= alpha:
                    return score

        if depth <= 0 or ply >= MAX_PLY - 1:
            return self._quiescence(alpha, beta, ply)

        count = board.generate(ply)
        if count == 0:
            return -MATE + ply if board.in_check() else 0

        buffer = board.buffers[ply]
        scores = self._score_buffer(ply)
        self._order(buffer, scores, count, ply, tt_move)

        alpha_original = alpha
        best_score = -INFINITY
        best_move = buffer[0]

        for index in range(count):
            move = self._pick(buffer, scores, count, index)
            board.make(move)
            try:
                score = -self._negamax(depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.unmake()
            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break

        if best_score <= alpha_original:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        if len(_TT) >= TT_MAX_ENTRIES:
            _TT.clear()
        _TT[key] = (depth, flag, _to_tt(best_score, ply), best_move)
        return best_score

    # ---- root -----------------------------------------------------------

    def root(self, moves: list[int], depth: int) -> tuple[int, int]:
        """One full-window iteration. Returns (score, best move)."""
        board = self.board
        alpha = -INFINITY
        best_move = moves[0]
        for index, move in enumerate(moves):
            board.make(move)
            try:
                score = -self._negamax(depth - 1, -INFINITY, -alpha, 1)
            finally:
                board.unmake()
            if index == 0 or score > alpha:
                alpha = score
                best_move = move
        return alpha, best_move

    def unwind(self) -> None:
        """Put the board back where the search found it.

        try/finally already balances every make, but a bug here loses a game
        outright by playing a move for the wrong side, so it is worth being
        certain rather than nearly certain.
        """
        while self.board.ply() > self._root_ply:
            self.board.unmake()


def _principal_variation(board: Board, first: int, limit: int) -> list[str]:
    """Walk the transposition table for a readable PV. Cosmetic, so it is
    defensive: a TT hit can be a hash collision and need not be legal here."""
    line: list[str] = []
    pushed = 0
    move = first
    try:
        while move != NULL_MOVE and len(line) < limit:
            count = board.generate(MAX_PLY + 2)
            legal = board.buffers[MAX_PLY + 2]
            if not any(legal[index] == move for index in range(count)):
                break
            line.append(move_to_uci(move))
            board.make(move)
            pushed += 1
            entry = _TT.get(board.key)
            if entry is None:
                break
            move = entry[3]
    finally:
        for _ in range(pushed):
            board.unmake()
    return line


def search(board: Board, limits: Limits) -> tuple[str, Info]:
    """Best move for the side to move, as UCI, plus what it took to find it.

    Never raises on a legal position: whatever the last completed iteration
    produced is returned, and if not even depth 1 finished, the first legally
    generated move is.
    """
    started = time.monotonic()
    info = Info()

    count = board.generate(0)
    if count == 0:
        return "0000", info
    moves = [board.buffers[0][index] for index in range(count)]

    searcher = _Searcher(board, limits)
    # A cheap first ordering, so an abandoned depth-1 search still returns
    # something sensible rather than whatever movegen emitted first.
    scores = [0] * count
    searcher._order(moves, scores, count, 0, NULL_MOVE)
    moves.sort(key=lambda move: -(_PIECE_ORDER[move_victim(move)] * 16))

    best_move = moves[0]
    best_score = 0

    if limits.time_ms is None:
        next_depth_deadline = None
    else:
        soft = limits.soft_ms if limits.soft_ms is not None else limits.time_ms / 2.0
        next_depth_deadline = started + soft / 1000.0

    try:
        for depth in range(1, limits.max_depth + 1):
            if (
                depth > 1
                and next_depth_deadline is not None
                and time.monotonic() >= next_depth_deadline
            ):
                break
            score, move = searcher.root(moves, depth)
            best_score, best_move = score, move
            info.depth = depth
            # The best move from this iteration is nearly always best again in
            # the next one, and searching it first is most of what makes
            # iterative deepening cheaper than it looks.
            moves.remove(move)
            moves.insert(0, move)
            if abs(score) >= MATE_BOUND:
                break  # a forced mate is not going to be improved on
    except _Timeout:
        searcher.unwind()

    info.nodes = searcher.nodes
    info.score = best_score
    if abs(best_score) >= MATE_BOUND:
        plies = MATE - abs(best_score)
        info.mate = (plies + 1) // 2 * (1 if best_score > 0 else -1)
    info.time_ms = (time.monotonic() - started) * 1000.0
    info.pv = _principal_variation(board, best_move, max(1, info.depth))
    return move_to_uci(best_move), info


def clear_tables() -> None:
    """Between games, or between unrelated positions in a test."""
    _TT.clear()


# ==========================================================================
# agent.py
# ==========================================================================

import contextlib
import os
from typing import Final

import chess


# The referee times the whole request/response round trip and flags the instant
# the clock goes negative. The 500ms watchdog grace in sandbox.py is not usable
# slack. This is held back for the IPC hop, JSON encoding and GC pauses.
OVERHEAD_MS: Final = 200

# The agent is never told the increment, only the clock, so the real control's
# 0.5s is baked in. AC_INCREMENT_MS exists so harness runs at a compressed
# control stay truthful; the platform never sets it, so rated play is unchanged.
INCREMENT_MS: Final = int(os.environ.get("AC_INCREMENT_MS", "500"))

MOVES_ASSUMED_LEFT: Final = 30
INCREMENT_FRACTION: Final = 0.8
CLOCK_CEILING: Final = 0.25  # no single move may commit more than this share of the clock
HARD_MULTIPLE: Final = 2.5  # how far past the target one running iteration may overrun

# Exposed for tools/selftest.py, which reaches into the module directly.
_tt = _TT


def _budget(time_left_ms: int, plies_played: int) -> tuple[float, float]:
    """Soft and hard limits in milliseconds.

    Soft is the target, and the point past which starting another iteration is
    a bad bet. Hard is where the search is abandoned mid-iteration; it exists
    only to bound the overrun of an iteration that was already running when the
    target passed, which is why it is a small multiple of soft rather than the
    clock ceiling itself. Setting hard to the ceiling directly lets a move that
    began one last deep iteration legally spend a quarter of the entire game
    clock finishing it. This engine did exactly that, for 19 seconds a move, and
    it only failed to flag because the ceiling is a fraction of what is left.

    Both limits being fractions of the remaining clock is what makes it decay
    geometrically instead of running out, and that is worth more than any
    amount of per-move cleverness.

    plies_played is unused today. It is in the signature because a move-count
    aware budget is the obvious next version and the harness already passes it.
    """
    usable = time_left_ms - OVERHEAD_MS
    if usable <= 0:
        return 0.0, 0.0
    ceiling = usable * CLOCK_CEILING
    soft = usable / MOVES_ASSUMED_LEFT + INCREMENT_FRACTION * INCREMENT_MS
    if soft > ceiling:
        soft = ceiling
    hard = min(ceiling, soft * HARD_MULTIPLE)
    return soft, hard


# --------------------------------------------------------------------------
# Game state
#
# We are handed a FEN and nothing else, but the referee calls
# outcome(claim_draw=True), so threefold and fifty-move draws are claimed
# against us automatically. So the game is rebuilt here: each call, work out
# which legal move the opponent played to reach the FEN we were given, push it,
# and keep the position history that repetition detection needs. If the chain
# breaks, resync from the FEN with a fresh history, which is the safe failure.
# --------------------------------------------------------------------------

# _board is the compatibility surface tools/selftest.py reaches for, so it is a
# plain chess.Board and setting it to None resets the agent. _tracked is the same
# game as an ac_board.Board; the two are never separate positions, because
# _tracked owns the chess.Board that _board points at.
_board: chess.Board | None = None
_tracked: Board | None = None
_history_keys: list[int] = []

_SCRATCH_PLY: Final = MAX_PLY + 3


def _sync(fen: str) -> Board:
    global _board, _tracked, _history_keys
    if _board is None:
        _tracked = None  # somebody reset us between games, or a test did
    target = Board(fen)

    if _tracked is not None:
        if _tracked.key == target.key:
            return _tracked
        count = _tracked.generate(_SCRATCH_PLY)
        buffer = _tracked.buffers[_SCRATCH_PLY]
        for index in range(count):
            move = buffer[index]
            _tracked.make(move)
            if _tracked.key == target.key:
                _history_keys.append(_tracked.key)
                return _tracked
            _tracked.unmake()

    _tracked = target
    _board = target.peek_native()
    _history_keys = [target.key]
    return _tracked


def _record(move_uci: str) -> None:
    """Push our own move, so next call has only one opponent move to reconstruct."""
    if _tracked is None:
        return
    move = _tracked.move_from_uci(move_uci)
    if move != NULL_MOVE:
        _tracked.make(move)
        _history_keys.append(_tracked.key)


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def _first_legal(board: Board) -> str:
    count = board.generate(_SCRATCH_PLY)
    if count == 0:
        return "0000"
    return move_to_uci(board.buffers[_SCRATCH_PLY][0])


def _think(fen: str, time_left_ms: int) -> str:
    board = _sync(fen)
    count = board.generate(_SCRATCH_PLY)
    if count == 0:
        return "0000"
    if count == 1:
        uci = move_to_uci(board.buffers[_SCRATCH_PLY][0])
        _record(uci)
        return uci

    soft_ms, hard_ms = _budget(time_left_ms, len(_history_keys))
    if hard_ms <= 0:
        # Out of clock. Returning instantly is the only thing left that helps.
        uci = _first_legal(board)
        _record(uci)
        return uci

    uci, _info = search(board, Limits(time_ms=hard_ms, soft_ms=soft_ms))
    _record(uci)
    return uci


def _fallback(fen: str) -> str:
    """Last resort. Its only job is to return something legal without raising."""
    try:
        for move in chess.Board(fen).legal_moves:
            return move.uci()
    except Exception:
        pass
    return "0000"


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    The platform's runner does not wrap this call, so an uncaught exception ends
    the process and loses the game. Everything is inside this try.
    """
    try:
        return _think(fen, time_left_ms)
    except Exception:
        return _fallback(fen)


def get_move_nodes(fen: str, nodes: int) -> str:
    """Fixed-node search. Harness only; the platform never calls this.

    A black-box get_move(fen, time_left_ms) cannot be driven at fixed nodes, so
    cross-engine comparison has to be timed. This exists so that A/B runs
    between versions we control are reproducible and parallelisable.
    """
    try:
        board = _sync(fen)
        if board.generate(_SCRATCH_PLY) == 0:
            return "0000"
        uci, _info = search(board, Limits(nodes=max(1, nodes)))
        _record(uci)
        return uci
    except Exception:
        return _fallback(fen)


def _warmup() -> None:
    """Compile and search once at import, inside the 60s init budget.

    numba compiles per signature, so every kernel is called here with the
    argument types the real search uses. A kernel first compiled on the clock is
    a kernel compiled during a rated game, and it is the whole init budget's
    reason for existing. The search pass afterwards is what catches a broken
    build during validation rather than on move one.
    """
    global _board, _tracked, _history_keys
    with contextlib.suppress(Exception):
        _warmup_bitboard()
        _warmup_eval()
    with contextlib.suppress(Exception):
        get_move(chess.STARTING_FEN, 3_000)
    _board = None
    _tracked = None
    _history_keys = []
    clear_tables()


_warmup()
