"""AI Chessathon agent: bot4's search on botB's numba bitboard kernel."""

from __future__ import annotations


# ==========================================================================
# ac_bitboard.py
# ==========================================================================

from typing import Any

import numpy as np
from numba import njit

MAX_PLY = 128
MAX_MOVES = 256
UNDO_SLOTS = 1024  # see BitboardBoard.__init__ for why this is not MAX_PLY

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
        # st[4] is a monotonic make/unmake stack pointer that counts every move
        # since this board was constructed, and _tracked holds the whole game.
        # So the index is (plies played) + (search ply), not search ply alone.
        # The kernel indexes undo[st[4]] with no bounds check, so overflowing it
        # is a silent heap corruption -- "free(): invalid next size" and a dead
        # process, not a Python exception get_move could catch. botB never hit
        # it because depth 4-5 kept the total under 136; at depth 14 it starts
        # crashing around move 50. Referee adjudication caps a game at 300
        # plies, so 1024 slots is the whole game plus the deepest search twice
        # over, for 40 KB.
        self._undo = np.zeros((UNDO_SLOTS, 5), np.int64)
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
# bot6 search
#
# This is bot4's search, moved onto the bitboard board contract above. The
# kernel, the board and the evaluation are byte-identical to botB and are not
# touched: perft proves them, and a change there would make a bench result
# unreadable. Everything below the kernel is new.
#
# What changed against botB's search, and why:
#
#   * two-tier transposition table that ages by swap instead of clearing
#   * killers, butterfly history with gravity, countermoves, and one ply of
#     continuation history
#   * static exchange evaluation, jitted, splitting winning from losing
#     captures and pruning in quiescence and at shallow depth
#   * principal variation search with progressive aspiration widening
#   * null move, reverse futility, late move pruning, futility, SEE pruning,
#     internal iterative reduction and a log-table late move reduction
#   * quiescence that generates evasions when in check instead of returning a
#     material score inside a mate
#   * bot4's time manager, which budgets from the clock it was handed and does
#     not spend its hard limit every move
#
# What was deliberately NOT ported:
#
#   * staged move generation. In bot4 the transposition move was tried before
#     list(board.legal_moves) was ever called, because generation cost 33 us
#     against 8 us for a legality test. Here generation costs 3.09 us, so the
#     saving is inside the noise and the extra state is a bug surface for
#     nothing. The transposition move is instead swapped to the front of a
#     normally generated list, which cannot desynchronise.
# ==========================================================================

import contextlib
import math
import time
from typing import Final

DEBUG: Final = False
STATS: Final = False

INFINITY: Final = 10_000_000
MATE: Final = 30_000
MATE_BOUND: Final = 29_000
MAX_SEARCH_PLY: Final = 96  # the move pools have MAX_PLY + 8 = 136 rows

OVERHEAD_MS: Final = 40
INCREMENT_MS: Final = 500

TT_MAX_ENTRIES: Final = 500_000

TT_EXACT: Final = 0
TT_LOWER: Final = 1
TT_UPPER: Final = 2

# The referee calls outcome(claim_draw=True), so a repetition is claimed against
# us the moment it becomes available. Being ahead and shuffling hands back a won
# game, so a draw is scored as a small loss when we are up material.
CONTEMPT_WINNING: Final = 35
CONTEMPT_LEVEL: Final = 10
CONTEMPT_LOSING: Final = -35
CONTEMPT_MARGIN: Final = 150

TEMPO: Final = 12

# bot4's margins, carried over unchanged. They are deliberately gentler than the
# published Stockfish numbers: those are tuned for engines seeing millions of
# nodes a move, and a margin that is merely optimistic at depth 30 throws away
# the only good move at depth 8. bot6 is faster than bot4 but nowhere near that
# regime, so the margins stay where they were until a measurement moves them.
RFP_MARGIN: Final = 80
RFP_MAX_DEPTH: Final = 7
FUTILITY_BASE: Final = 110
FUTILITY_MARGIN: Final = 95
FUTILITY_MAX_DEPTH: Final = 6
LMP_MAX_DEPTH: Final = 6
SEE_QUIET_MARGIN: Final = -70
SEE_CAPTURE_MARGIN: Final = -90
SEE_PRUNE_MAX_DEPTH: Final = 6
IIR_MIN_DEPTH: Final = 4
DELTA_MARGIN: Final = 190

_HISTORY_MAX: Final = 16_384
_CONT_SLOTS: Final = 384

SEE_VALUE: Final = (0, 100, 320, 330, 500, 900, 20_000)
SEE_VALUES = np.array(SEE_VALUE, dtype=np.int64)

# Clock. Same shape as bot4: a soft target that decides whether to start another
# iteration, and a hard limit that only bounds the overrun of an iteration
# already running. botB spent its hard limit on every single move, which is not
# a time manager, it is a constant.
MOVES_ASSUMED_LEFT: Final = 30
INCREMENT_FRACTION: Final = 0.8
CLOCK_CEILING: Final = 0.20
HARD_MULTIPLE: Final = 2.2
NEXT_DEPTH_FRACTION: Final = 0.55
FAIL_LOW_FRACTION: Final = 0.85

# bot4's stability break stops once the best move survives three iterations with
# little drift. The iteration log already flags it as untuned and suspect: bot4
# was efficient enough to trigger it with most of its budget unspent. bot6 is
# faster still, so it would trigger harder. It is off until it is measured.
STABLE_BREAK: Final = False
STABLE_SCORE_DRIFT: Final = 20
STABLE_MIN_DEPTH: Final = 6


# --------------------------------------------------------------------------
# Static exchange evaluation, jitted
#
# bot4 ran SEE in Python at about 10 us a capture, which was affordable only
# because a victim-outranks-attacker filter removed most calls. On bitboards the
# swap-off loop is a handful of mask operations, so it runs on every capture and
# the filter is gone.
# --------------------------------------------------------------------------


@njit(cache=False)
def _attackers_to(bb: Any, square: int, occupied: Any) -> Any:
    """Every piece of either colour attacking `square` for a given occupancy.

    Occupancy is a parameter rather than bb[14] because the swap-off loop
    removes pieces as they are captured, and an x-ray behind a departed piece
    has to become visible.
    """
    attackers = PAWN_ATTACKS[1][square] & bb[0]
    attackers |= PAWN_ATTACKS[0][square] & bb[6]
    attackers |= KNIGHT_ATTACKS[square] & (bb[1] | bb[7])
    attackers |= KING_ATTACKS[square] & (bb[5] | bb[11])
    diagonal = bb[2] | bb[4] | bb[8] | bb[10]
    if diagonal:
        attackers |= _bishop_attacks(square, occupied) & diagonal
    straight = bb[3] | bb[4] | bb[9] | bb[10]
    if straight:
        attackers |= _rook_attacks(square, occupied) & straight
    return attackers & occupied


@njit(cache=False)
def _see_kernel(bb: Any, st: Any, move: int) -> int:
    """Material won by the side to move if the exchange on the target square
    is played out with the cheapest attacker each time.

    Returns centipawns from the mover's point of view. Pins are ignored, which
    is standard: bot4 measured two sign errors in 2,543 captures, both
    conservative.
    """
    source = move & 0x3F
    target = (move >> 6) & 0x3F
    side = int(st[0])

    mover = (move >> 21) & 0x7
    victim = (move >> 18) & 0x7
    if move & FLAG_EP:
        victim = 1

    gain = np.zeros(32, np.int64)
    depth = 0
    gain[0] = SEE_VALUES[victim]

    occupied = bb[14] & ~_bit(source)
    if move & FLAG_EP:
        captured_square = target - 8 if side == 0 else target + 8
        occupied &= ~_bit(captured_square)

    promotion = (move >> 12) & 0x7
    if promotion:
        gain[0] += SEE_VALUES[promotion] - SEE_VALUES[1]
        mover = promotion

    attackers = _attackers_to(bb, target, occupied)
    turn = 1 - side
    on_square = mover

    while True:
        depth += 1
        gain[depth] = SEE_VALUES[on_square] - gain[depth - 1]

        base = turn * 6
        found = 0
        from_bit = np.uint64(0)
        for piece in range(1, 7):
            candidates = bb[base + piece - 1] & attackers & occupied
            if candidates:
                found = piece
                from_bit = candidates & (np.uint64(0) - candidates)
                break
        if found == 0:
            break

        occupied &= ~from_bit
        # An attacker leaving can expose a slider behind it, so recompute
        # rather than merely clearing the bit from the old set.
        attackers = _attackers_to(bb, target, occupied)
        on_square = found
        turn = 1 - turn
        if depth >= 30:
            break

    # Standard negamax unwind: at each level the side to move may stand pat
    # rather than continue the exchange, so gain[d-1] = min(gain[d-1], -gain[d]).
    while depth > 1:
        depth -= 1
        if -gain[depth] < gain[depth - 1]:
            gain[depth - 1] = -gain[depth]
    return int(gain[0])


@njit(cache=False)
def _material_kernel(bb: Any) -> int:
    """White minus black in centipawns. Drives contempt only."""
    total = 0
    for piece in range(1, 6):
        value = SEE_VALUES[piece]
        total += value * _popcount(bb[piece - 1])
        total -= value * _popcount(bb[6 + piece - 1])
    return total


def _see(board: "Board", move: int) -> int:
    return int(_see_kernel(board._bb, board._st, move))


def _material(board: "Board") -> int:
    return int(_material_kernel(board._bb))


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

_tt: dict[int, tuple[int, int, int, int, int]] = {}
_tt_old: dict[int, tuple[int, int, int, int, int]] = {}

_NO_EVAL: Final = -INFINITY - 1

_STACK: Final = MAX_SEARCH_PLY + 8

_killers: list[list[int]] = [[0, 0] for _ in range(_STACK)]
_history: list[list[int]] = [[0] * 4096, [0] * 4096]
_counters: list[list[int]] = [[0] * 4096, [0] * 4096]
_continuation: list[int] = [0] * (_CONT_SLOTS * _CONT_SLOTS)

_stack_eval: list[int] = [0] * _STACK
_stack_cont: list[int] = [-1] * _STACK

_nodes = 0
_qnodes = 0
_cutoffs = 0
_first_move_cutoffs = 0
_depth_reached = 0
_deadline = 0.0
_root_turn = True
_contempt = 0
_ply_cap = MAX_SEARCH_PLY

# Ordering buckets, spaced so the quiet history range can never collide with a
# capture or a killer whatever the tables learn.
_WINNING_CAPTURE: Final = 1 << 26
_PROMOTION: Final = 1 << 25
_KILLER_FIRST: Final = 1 << 24
_KILLER_SECOND: Final = (1 << 24) - 1
_COUNTER_MOVE: Final = 1 << 23
_LOSING_CAPTURE: Final = -(1 << 22)

# Late move reduction table, log-shaped, built once.
_LMR: list[list[int]] = [
    [
        0 if depth < 3 or index < 3 else min(depth - 1, int(0.75 + math.log(depth) * math.log(index) / 2.25))
        for index in range(64)
    ]
    for depth in range(64)
]


class _Timeout(Exception):
    pass


def _tt_probe(key: int):
    entry = _tt.get(key)
    if entry is not None:
        return entry
    entry = _tt_old.get(key)
    if entry is not None:
        _tt[key] = entry  # promote, so a hot entry survives the next age
    return entry


def _tt_store(key: int, depth: int, flag: int, score: int, move: int, static: int) -> None:
    global _tt, _tt_old
    if len(_tt) >= TT_MAX_ENTRIES:
        # Age by swapping tiers. Dropping half the table is constant time,
        # where dict.clear() mid-search throws away the whole iteration's work.
        _tt_old = _tt
        _tt = {}
    _tt[key] = (depth, flag, score, move, static)


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


def _has_pieces(board: "Board") -> bool:
    """Any non-pawn, non-king material for the side to move.

    Null move hands the opponent a free move and asks whether we are still
    winning. In a pawn ending that question has the wrong answer, because
    zugzwang makes having to move a liability rather than a right.
    """
    base = 0 if board.turn else 6
    bb = board._bb
    return bool(bb[base + 1] | bb[base + 2] | bb[base + 3] | bb[base + 4])


def _draw_score(board: "Board") -> int:
    """A draw is not always worth zero. See CONTEMPT_* above."""
    balance = _material(board)
    if not _root_turn:
        balance = -balance
    if balance > CONTEMPT_MARGIN:
        return -CONTEMPT_WINNING if board.turn == _root_turn else CONTEMPT_WINNING
    if balance < -CONTEMPT_MARGIN:
        return -CONTEMPT_LOSING if board.turn == _root_turn else CONTEMPT_LOSING
    return -CONTEMPT_LEVEL if board.turn == _root_turn else CONTEMPT_LEVEL


def clear_tables() -> None:
    """Between games, or between unrelated positions in a test."""
    global _tt, _tt_old
    _tt = {}
    _tt_old = {}
    for side in range(2):
        history = _history[side]
        counters = _counters[side]
        for index in range(4096):
            history[index] = 0
            counters[index] = 0
    for index in range(len(_continuation)):
        _continuation[index] = 0
    for ply in range(_STACK):
        _killers[ply][0] = 0
        _killers[ply][1] = 0


# --------------------------------------------------------------------------
# Move ordering
#
# The victim and the mover are packed into the move by the generator, at bits
# 18 and 21. bot4 had to call board.piece_type_at() twice per move to recover
# them, which was a large slice of its ordering cost; here MVV-LVA is three
# shifts and no board access at all.
# --------------------------------------------------------------------------


def _order_score(board: "Board", move: int, ply: int, tt_move: int, parent: int) -> int:
    if move == tt_move:
        return 1 << 30

    victim = (move >> 18) & 0x7
    promotion = (move >> 12) & 0x7

    if victim:
        mover = (move >> 21) & 0x7
        mvv_lva = victim * 16 - mover
        # SEE only chooses the bucket; MVV-LVA orders inside it. A capture that
        # loses material still gets searched, just after every quiet move that
        # history likes, which is what stops a losing sacrifice from being
        # tried first at every node of the tree.
        if victim >= mover or _see(board, move) >= 0:
            return _WINNING_CAPTURE + mvv_lva
        return _LOSING_CAPTURE + mvv_lva

    if promotion:
        return _PROMOTION + promotion

    killers = _killers[ply]
    if move == killers[0]:
        return _KILLER_FIRST
    if move == killers[1]:
        return _KILLER_SECOND

    side = 0 if board.turn else 1
    index = move & 0xFFF  # source | target << 6, a free 12-bit history index
    if _counters[side][index] == move:
        return _COUNTER_MOVE

    score = _history[side][index]
    if parent >= 0:
        mover = (move >> 21) & 0x7
        target = (move >> 6) & 0x3F
        score += _continuation[parent * _CONT_SLOTS + (mover - 1) * 64 + target]
    return score


def _record_cutoff(board: "Board", move: int, ply: int, depth: int,
                   parent: int, bad_quiets: list) -> None:
    """Reward the move that cut off, and punish the quiets that did not.

    The malus matters as much as the bonus: without it history only ever grows,
    and a move that was good once keeps its place long after it stopped being.
    """
    killers = _killers[ply]
    if killers[0] != move:
        killers[1] = killers[0]
        killers[0] = move

    side = 0 if board.turn else 1
    index = move & 0xFFF
    bonus = min(depth * depth * 4, _HISTORY_MAX)

    table = _history[side]
    value = table[index]
    table[index] = value + bonus - (value * bonus) // _HISTORY_MAX  # gravity

    if parent >= 0:
        mover = (move >> 21) & 0x7
        target = (move >> 6) & 0x3F
        slot = parent * _CONT_SLOTS + (mover - 1) * 64 + target
        value = _continuation[slot]
        _continuation[slot] = value + bonus - (value * bonus) // _HISTORY_MAX

    if ply > 0:
        previous = _stack_cont[ply - 1]
        if previous >= 0:
            _counters[side][index] = move

    for other_index, other_slot in bad_quiets:
        value = table[other_index]
        table[other_index] = value - bonus - (value * bonus) // _HISTORY_MAX
        if other_slot >= 0:
            value = _continuation[other_slot]
            _continuation[other_slot] = value - bonus - (value * bonus) // _HISTORY_MAX


# --------------------------------------------------------------------------
# Quiescence
# --------------------------------------------------------------------------


def _quiescence(board: "Board", alpha: int, beta: int, ply: int) -> int:
    global _nodes, _qnodes

    _nodes += 1
    _qnodes += 1
    if not (_nodes & 1023) and time.monotonic() >= _deadline:
        raise _Timeout

    if ply >= _ply_cap:
        return evaluate(board) + TEMPO

    in_check = board.in_check()

    if in_check:
        # bot2's regression bug #4: returning a static score while in check
        # reports a material evaluation from inside a forced mate. Every evasion
        # has to be searched, and standing pat is not an option.
        count = board.generate(ply)
        if count == 0:
            return -MATE + ply
        moves = board.buffers[ply][:count].tolist()
        best = -INFINITY
    else:
        static = evaluate(board) + TEMPO
        if static >= beta:
            return static
        if static > alpha:
            alpha = static
        best = static
        count = board.generate_captures(ply)
        if count == 0:
            return best
        moves = board.buffers[ply][:count].tolist()

    moves.sort(key=lambda move: _order_score(board, move, ply, 0, -1), reverse=True)

    for move in moves:
        if not in_check:
            victim = (move >> 18) & 0x7
            promotion = (move >> 12) & 0x7
            if not promotion:
                # Delta pruning: a capture that cannot lift the static score to
                # alpha even if it were free is not worth a node.
                if static + SEE_VALUE[victim] + DELTA_MARGIN < alpha:
                    continue
                if _see(board, move) < 0:
                    continue

        board.make(move)
        try:
            score = -_quiescence(board, -beta, -alpha, ply + 1)
        finally:
            board.unmake()

        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    return best


# --------------------------------------------------------------------------
# Main search
# --------------------------------------------------------------------------


def _negamax(board: "Board", depth: int, alpha: int, beta: int, ply: int, is_pv: bool) -> int:
    global _nodes, _cutoffs, _first_move_cutoffs

    _nodes += 1
    if not (_nodes & 1023) and time.monotonic() >= _deadline:
        raise _Timeout

    if ply > 0:
        # Two-fold is the right test inside the search, not threefold: the
        # referee claims the draw the moment it becomes available, so a
        # repetition has to be seen one visit earlier than the rules require.
        if board.is_repetition(2) or board.halfmove_clock >= 100 or board._insufficient():
            return _draw_score(board)
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

    key = board.key
    tt_move = 0
    tt_static = _NO_EVAL
    entry = _tt_probe(key)
    if entry is not None:
        stored_depth, flag, stored_score, tt_move, tt_static = entry
        if ply > 0 and not is_pv and stored_depth >= depth:
            score = _from_tt(stored_score, ply)
            if flag == TT_EXACT:
                return score
            if flag == TT_LOWER and score >= beta:
                return score
            if flag == TT_UPPER and score <= alpha:
                return score

    in_check = board.in_check()
    if in_check and ply < _ply_cap:
        depth += 1  # never drop into quiescence while in check
    if depth <= 0 or ply >= _ply_cap:
        return _quiescence(board, alpha, beta, ply)

    if in_check:
        static = _NO_EVAL
    elif tt_static != _NO_EVAL:
        static = tt_static
    else:
        static = evaluate(board) + TEMPO
    _stack_eval[ply] = static

    improving = (
        static != _NO_EVAL
        and ply >= 2
        and _stack_eval[ply - 2] != _NO_EVAL
        and static > _stack_eval[ply - 2]
    )

    # Every forward prune below is gated on the same three conditions: not in
    # check, not a PV node, no mate score in the window. Those three together
    # are what stop a margin from silently dropping a forced sequence.
    prunable = not in_check and not is_pv and abs(beta) < MATE_BOUND and abs(alpha) < MATE_BOUND

    if prunable and static != _NO_EVAL:
        if depth <= RFP_MAX_DEPTH:
            margin = RFP_MARGIN * depth - (RFP_MARGIN // 2 if improving else 0)
            if static - margin >= beta:
                return static - margin

        # Null move. Skipped with only pawns left, where zugzwang makes it
        # unsound, and never when the static score is already below beta.
        if (
            depth >= 3
            and static >= beta
            and _has_pieces(board)
        ):
            reduction = 2 + (depth > 6) + min(2, (static - beta) // 200)
            board.make_null()
            _stack_cont[ply] = -1
            try:
                score = -_negamax(board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, False)
            finally:
                board.unmake_null()
            if score >= beta:
                return beta if score >= MATE_BOUND else score

    # Internal iterative reduction: with no transposition move to lead on, this
    # node's ordering is a guess, so do not spend full depth proving the guess.
    if tt_move == 0 and depth >= IIR_MIN_DEPTH:
        depth -= 1

    count = board.generate(ply)
    if count == 0:
        return -MATE + ply if in_check else _draw_score(board)
    moves = board.buffers[ply][:count].tolist()

    parent = _stack_cont[ply - 1] if ply > 0 else -1
    moves.sort(key=lambda move: _order_score(board, move, ply, tt_move, parent), reverse=True)

    lmp_limit = 3 + depth * depth // (1 if improving else 2)
    futility_base = static + FUTILITY_BASE if static != _NO_EVAL else -INFINITY

    best_score = -INFINITY
    best_move = 0
    alpha_original = alpha
    bad_quiets: list = []

    for index, move in enumerate(moves):
        victim = (move >> 18) & 0x7
        promotion = (move >> 12) & 0x7
        quiet = victim == 0 and promotion == 0
        mover = (move >> 21) & 0x7
        target = (move >> 6) & 0x3F
        cont = parent * _CONT_SLOTS + (mover - 1) * 64 + target if parent >= 0 else -1

        # Static exchange evaluation reads the position before the move, so it
        # has to be taken here even though the prune decision happens after.
        may_prune = prunable and best_move != 0 and abs(best_score) < MATE_BOUND
        see_value = 0
        have_see = False
        if may_prune and depth <= SEE_PRUNE_MAX_DEPTH:
            see_value = _see(board, move)
            have_see = True

        _stack_cont[ply] = cont
        board.make(move)
        # A move that gives check is never pruned and never reduced. This is the
        # guard whose absence cost the selftest's mate in 3: the only move that
        # forces it, d6d1, is a *quiet* move, so late move pruning threw it away
        # and the search reported -229 in a position that is mate in three.
        # Testing after make costs 2.17us here and needs no separate
        # gives-check kernel to go wrong.
        gives_check = board.in_check()

        if may_prune and not gives_check:
            pruned = False
            if quiet:
                if depth <= LMP_MAX_DEPTH and index > lmp_limit:
                    pruned = True
                elif (
                    depth <= FUTILITY_MAX_DEPTH
                    and futility_base + FUTILITY_MARGIN * depth <= alpha
                ):
                    pruned = True
                elif have_see and see_value < SEE_QUIET_MARGIN * depth * depth:
                    pruned = True
            elif have_see and see_value < SEE_CAPTURE_MARGIN * depth:
                pruned = True
            if pruned:
                board.unmake()
                if quiet:
                    bad_quiets.append((move & 0xFFF, cont))
                continue

        try:
            reduction = 0
            if quiet and not in_check and not gives_check and depth >= 3 and index >= 3:
                reduction = _LMR[min(depth, 63)][min(index, 63)]
                if is_pv:
                    reduction -= 1
                if not improving:
                    reduction += 1
                if reduction < 0:
                    reduction = 0
                if reduction > depth - 2:
                    reduction = depth - 2

            if index == 0:
                score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1, is_pv)
            else:
                score = -_negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, False)
                if reduction and score > alpha:
                    score = -_negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, False)
                if alpha < score < beta:
                    score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1, is_pv)
        finally:
            board.unmake()

        if score > best_score:
            best_score = score
            best_move = move
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if STATS:
                        globals()["_cutoffs"] = _cutoffs + 1
                        if index == 0:
                            globals()["_first_move_cutoffs"] = _first_move_cutoffs + 1
                    if quiet:
                        _record_cutoff(board, move, ply, depth, parent, bad_quiets)
                    break
        if quiet and move != best_move:
            bad_quiets.append((move & 0xFFF, cont))

    if best_score >= beta:
        flag = TT_LOWER
    elif best_score > alpha_original:
        flag = TT_EXACT
    else:
        flag = TT_UPPER
    _tt_store(key, depth, flag, _to_tt(best_score, ply), best_move, static)
    return best_score


def _search_root(board: "Board", depth: int, alpha: int, beta: int, moves: list) -> tuple:
    global _nodes
    best_score = -INFINITY
    best_move = moves[0]
    _stack_cont[0] = -1

    for index, move in enumerate(moves):
        board.make(move)
        try:
            if index == 0:
                score = -_negamax(board, depth - 1, -beta, -alpha, 1, True)
            else:
                score = -_negamax(board, depth - 1, -alpha - 1, -alpha, 1, False)
                if alpha < score < beta:
                    score = -_negamax(board, depth - 1, -beta, -alpha, 1, True)
        finally:
            board.unmake()

        if score > best_score:
            best_score = score
            best_move = move
            if score > alpha:
                alpha = score
        if alpha >= beta:
            break
    return best_score, best_move


# --------------------------------------------------------------------------
# Time management
# --------------------------------------------------------------------------


def _budget(time_left_ms: int, plies_played: int = 0) -> tuple[float, float]:
    """Soft target and hard ceiling, both fractions of the clock we were handed.

    botB computed the same two numbers and then spent the hard one on every
    move, which made soft decorative and put eleven seconds into move one of a
    curated opening. Here soft decides whether another iteration starts and hard
    only bounds an iteration already in flight.
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


def search_position(board: "Board", soft_ms: float, hard_ms: float) -> tuple[str, int, int]:
    """Iteratively deepen. Returns (uci, score, depth reached)."""
    global _nodes, _qnodes, _deadline, _root_turn, _depth_reached
    global _cutoffs, _first_move_cutoffs

    _nodes = 0
    _qnodes = 0
    _cutoffs = 0
    _first_move_cutoffs = 0
    _root_turn = board.turn
    started = time.monotonic()
    _deadline = started + hard_ms / 1000.0

    # Second guard on the same bug: cap this search so that base + ply can
    # never reach UNDO_SLOTS however deep the extensions go. Computed once, so
    # it costs nothing per node.
    global _ply_cap
    _ply_cap = min(MAX_SEARCH_PLY, UNDO_SLOTS - int(board._st[4]) - 16)
    if _ply_cap < 2:
        return _first_legal(board), 0, 0

    count = board.generate(0)
    if count == 0:
        return "0000", 0, 0
    moves = board.buffers[0][:count].tolist()
    if count == 1:
        return move_to_uci(moves[0]), 0, 1

    for ply in range(_STACK):
        _stack_eval[ply] = _NO_EVAL
        _stack_cont[ply] = -1

    moves.sort(key=lambda move: _order_score(board, move, 0, 0, -1), reverse=True)
    best_move = moves[0]
    best_score = 0
    depth_reached = 0
    previous_score = -INFINITY
    stable = 0
    failed_low = False

    try:
        for depth in range(1, 64):
            if depth > 1:
                fraction = FAIL_LOW_FRACTION if failed_low else NEXT_DEPTH_FRACTION
                if time.monotonic() >= started + (soft_ms * fraction) / 1000.0:
                    break
            failed_low = False

            if depth <= 3:
                window_low, window_high = -INFINITY, INFINITY
            else:
                window_low, window_high = best_score - 25, best_score + 25
            delta = 25

            while True:
                score, move = _search_root(board, depth, window_low, window_high, moves)
                delta *= 3
                if score <= window_low and window_low > -INFINITY:
                    failed_low = True
                    window_high = (window_low + window_high) // 2
                    window_low = max(-INFINITY, score - delta) if delta < 800 else -INFINITY
                    continue
                if score >= window_high and window_high < INFINITY:
                    window_high = min(INFINITY, score + delta) if delta < 800 else INFINITY
                    continue
                break

            if abs(score - previous_score) <= STABLE_SCORE_DRIFT and move == best_move:
                stable += 1
            else:
                stable = 0
            previous_score = score
            best_score = score
            best_move = move
            depth_reached = depth

            moves.remove(move)
            moves.insert(0, move)

            if abs(score) >= MATE_BOUND:
                break
            if STABLE_BREAK and stable >= 3 and depth >= STABLE_MIN_DEPTH:
                break
    except _Timeout:
        pass

    _depth_reached = depth_reached
    if DEBUG:
        elapsed = (time.monotonic() - started) * 1000.0
        line = (
            f"depth {depth_reached} score {best_score} nodes {_nodes} "
            f"{elapsed:.0f}ms {_nodes / max(elapsed, 1) * 1000:.0f}nps "
            f"budget {soft_ms:.0f}/{hard_ms:.0f} move {move_to_uci(best_move)}"
        )
        if STATS and _cutoffs:
            line += f" | first-move cutoffs {_first_move_cutoffs / _cutoffs:.1%}"
        print(line)
    return move_to_uci(best_move), best_score, depth_reached


# ==========================================================================
# agent
# ==========================================================================

_board: Any = None
_tracked: Any = None
_history_keys: list[int] = []

_SCRATCH_PLY: Final = MAX_PLY + 3


def _sync(fen: str) -> Any:
    """Rebuild the game from the FEN we were handed.

    We are given a position and nothing else, but the referee claims threefold
    and fifty-move draws automatically, so the position history has to be
    reconstructed or we can hand back a won game by shuffling. Each call, find
    which legal move the opponent played to reach this FEN and push it. If the
    chain breaks, resync with a fresh history, which is the safe failure.
    """
    global _board, _tracked, _history_keys
    if _board is None:
        _tracked = None
    target = Board(fen)

    if _tracked is not None:
        if _tracked.key == target.key:
            return _tracked
        count = _tracked.generate(_SCRATCH_PLY)
        buffer = _tracked.buffers[_SCRATCH_PLY]
        for index in range(count):
            move = int(buffer[index])
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
    if _tracked is None:
        return
    move = _tracked.move_from_uci(move_uci)
    if move != NULL_MOVE:
        _tracked.make(move)
        _history_keys.append(_tracked.key)


def _first_legal(board: Any) -> str:
    count = board.generate(_SCRATCH_PLY)
    if count == 0:
        return "0000"
    return move_to_uci(int(board.buffers[_SCRATCH_PLY][0]))


def _think(fen: str, time_left_ms: int) -> str:
    global _contempt
    board = _sync(fen)
    count = board.generate(_SCRATCH_PLY)
    if count == 0:
        return "0000"
    if count == 1:
        uci = move_to_uci(int(board.buffers[_SCRATCH_PLY][0]))
        _record(uci)
        return uci

    soft_ms, hard_ms = _budget(time_left_ms, len(_history_keys))
    if hard_ms <= 0:
        uci = _first_legal(board)
        _record(uci)
        return uci

    uci, _score, _depth = search_position(board, soft_ms, hard_ms)
    if uci == "0000":
        uci = _first_legal(board)
    _record(uci)
    return uci


def _fallback(fen: str) -> str:
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
    """Fixed-node search, harness only. Kept for A/B runs between our builds."""
    try:
        board = _sync(fen)
        if board.generate(_SCRATCH_PLY) == 0:
            return "0000"
        uci, _score, _depth = search_position(board, 1e9, 1e9)
        _record(uci)
        return uci
    except Exception:
        return _fallback(fen)


def _warmup() -> None:
    """Compile every kernel at import, inside the 60 second budget.

    numba compiles per signature, so each kernel has to be called here with the
    argument types the real search uses. A kernel first compiled on the clock is
    a kernel compiled during a rated game.
    """
    global _board, _tracked, _history_keys
    with contextlib.suppress(Exception):
        _warmup_bitboard()
        _warmup_eval()
    with contextlib.suppress(Exception):
        probe = Board()
        probe.generate(0)
        _see_kernel(probe._bb, probe._st, int(probe.buffers[0][0]))
        _material_kernel(probe._bb)
    with contextlib.suppress(Exception):
        get_move(chess.STARTING_FEN, 3_000)
    _board = None
    _tracked = None
    _history_keys = []
    clear_tables()


_warmup()
