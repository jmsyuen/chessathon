"""AI Chessathon submission entrypoint.

A classical engine: iterative deepening negamax with alpha-beta, a transposition
table, quiescence search, and a tapered material + piece-square evaluation.
Everything runs on python-chess. No model, no numba, no torch.

The design priority, in order, is: never crash, never flag, never play an illegal
move, then play well. The first three are free losses and they are the ones that
sink an otherwise decent agent.

Layout of this file:
    1. Constants and tunables
    2. Evaluation tables, built at import
    3. Evaluation
    4. Move ordering
    5. Search
    6. Time management
    7. Game-state tracking (repetition history from FEN alone)
    8. get_move
"""

from __future__ import annotations

import contextlib
import time
from typing import Final

import chess

# --------------------------------------------------------------------------
# 1. Constants and tunables
# --------------------------------------------------------------------------

DEBUG: Final = False  # stdout is discarded in rated games; flip on to read validation logs

INFINITY: Final = 10_000_000
MATE: Final = 30_000
MATE_BOUND: Final = 29_000  # scores past this are mate scores and need ply adjustment
MAX_PLY: Final = 64

# The referee times the whole request/response round trip, not just get_move, and
# flags the moment the clock goes negative. Hold this much back for the IPC hop,
# JSON encoding, and the odd garbage collection pause.
OVERHEAD_MS: Final = 40
INCREMENT_MS: Final = 500

# A transposition entry is a tuple plus a tuple key, so roughly half a kilobyte
# all in. This cap keeps the table near 250 MB against a 2 GB container.
TT_MAX_ENTRIES: Final = 500_000
PAWN_CACHE_MAX: Final = 100_000

TT_EXACT: Final = 0
TT_LOWER: Final = 1
TT_UPPER: Final = 2

# Draw handling matters more here than in a normal engine. The referee calls
# board.outcome(claim_draw=True), so threefold and fifty-move draws are claimed
# against us automatically; we can hand back a won game by shuffling.
CONTEMPT_WINNING: Final = 35  # ahead: treat a draw as a small loss
CONTEMPT_LEVEL: Final = 10  # level: mild preference to keep playing
CONTEMPT_LOSING: Final = -35  # behind: a draw is a good outcome, take it
CONTEMPT_MARGIN: Final = 150  # centipawns of material that counts as "ahead"

TEMPO: Final = 12

# --------------------------------------------------------------------------
# 2. Evaluation tables
# --------------------------------------------------------------------------

# Piece values in centipawns, midgame and endgame. Textbook values: pawns and
# rooks gain in the endgame, minor pieces stay flat.
PIECE_MG: Final = (0, 100, 320, 330, 500, 900, 0)
PIECE_EG: Final = (0, 120, 320, 340, 550, 950, 0)

# Phase weights: 24 at a full board, 0 in a bare pawn endgame.
PHASE_WEIGHT: Final = (0, 0, 1, 1, 2, 4, 0)
TOTAL_PHASE: Final = 24


def _flatten(rows: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Turn a table written rank 8 first into one indexed by python-chess square."""
    flat: list[int] = []
    for rank in range(8):
        flat.extend(rows[7 - rank])
    return tuple(flat)


# These tables are hand-authored from ordinary positional principles: centralise
# knights, keep bishops on long diagonals, rooks on the seventh, king tucked away
# in the midgame and active in the endgame. They are deliberately plain, because
# the point of a baseline is to be a reference the next version is measured
# against. Tuning them is the single highest-value improvement from here.

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
    (18, 30, 8, 0, 0, 8, 34, 18),
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

_EMPTY_TABLE: Final = (0,) * 64
# Indexed by piece type, so index 0 is unused padding.
MG_TABLE: Final = (
    _EMPTY_TABLE, _PAWN_MG, _KNIGHT_MG, _BISHOP_MG, _ROOK_MG, _QUEEN_MG, _KING_MG,
)
EG_TABLE: Final = (
    _EMPTY_TABLE, _PAWN_EG, _KNIGHT_EG, _BISHOP_EG, _ROOK_EG, _QUEEN_EG, _KING_EG,
)

# Pawn-structure masks, built once at import.
_FILE_BB: Final = tuple(chess.BB_FILES)
_ADJACENT_FILES: Final = tuple(
    (chess.BB_FILES[file - 1] if file > 0 else 0)
    | (chess.BB_FILES[file + 1] if file < 7 else 0)
    for file in range(8)
)


def _build_passed_masks(color: chess.Color) -> tuple[int, ...]:
    """Squares an enemy pawn must occupy to stop this pawn being passed."""
    masks: list[int] = []
    for square in range(64):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        span = _FILE_BB[file] | _ADJACENT_FILES[file]
        ahead = 0
        ranks = range(rank + 1, 8) if color == chess.WHITE else range(0, rank)
        for other in ranks:
            ahead |= chess.BB_RANKS[other]
        masks.append(span & ahead)
    return tuple(masks)


def _build_shield_masks(color: chess.Color) -> tuple[int, ...]:
    """The three files around the king, on the two ranks in front of it."""
    masks: list[int] = []
    for square in range(64):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        span = _FILE_BB[file] | _ADJACENT_FILES[file]
        ahead = 0
        steps = (1, 2) if color == chess.WHITE else (-1, -2)
        for step in steps:
            target = rank + step
            if 0 <= target <= 7:
                ahead |= chess.BB_RANKS[target]
        masks.append(span & ahead)
    return tuple(masks)


# Indexed by int(color), so index 1 is White.
_PASSED_MASK: Final = (_build_passed_masks(chess.BLACK), _build_passed_masks(chess.WHITE))
_SHIELD_MASK: Final = (_build_shield_masks(chess.BLACK), _build_shield_masks(chess.WHITE))

# Bonus for a passed pawn by rank, counted from the pawn's own side of the board.
_PASSED_MG: Final = (0, 4, 8, 16, 30, 55, 90, 0)
_PASSED_EG: Final = (0, 10, 18, 34, 60, 100, 160, 0)

_DOUBLED_MG: Final = -12
_DOUBLED_EG: Final = -24
_ISOLATED_MG: Final = -14
_ISOLATED_EG: Final = -18
_BISHOP_PAIR_MG: Final = 28
_BISHOP_PAIR_EG: Final = 45
_ROOK_OPEN_FILE: Final = 22
_ROOK_SEMI_OPEN_FILE: Final = 11
_SHIELD_MISSING: Final = -14

# Promotion generation masks for quiescence, indexed by int(color).
_SEVENTH_RANK: Final = (chess.BB_RANK_2, chess.BB_RANK_7)
_EIGHTH_RANK: Final = (chess.BB_RANK_1, chess.BB_RANK_8)


# --------------------------------------------------------------------------
# 3. Evaluation
# --------------------------------------------------------------------------

_pawn_cache: dict[tuple[int, int], tuple[int, int]] = {}


def _pawn_structure(white_pawns: int, black_pawns: int) -> tuple[int, int]:
    """Doubled, isolated and passed pawns, from White's perspective.

    Pawn structure changes rarely, so this is cached on the pair of pawn
    bitboards. It is the most expensive part of the evaluation and the cache hit
    rate inside a search is very high.
    """
    key = (white_pawns, black_pawns)
    cached = _pawn_cache.get(key)
    if cached is not None:
        return cached

    midgame = 0
    endgame = 0
    sides = ((black_pawns, white_pawns), (white_pawns, black_pawns))
    for color_index, (own, enemy) in enumerate(sides):
        sign = 1 if color_index else -1
        passed_masks = _PASSED_MASK[color_index]
        for file in range(8):
            count = chess.popcount(own & _FILE_BB[file])
            if count > 1:
                midgame += sign * _DOUBLED_MG * (count - 1)
                endgame += sign * _DOUBLED_EG * (count - 1)
        bits = own
        while bits:
            square = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            if not own & _ADJACENT_FILES[square & 7]:
                midgame += sign * _ISOLATED_MG
                endgame += sign * _ISOLATED_EG
            if not enemy & passed_masks[square]:
                rank = square >> 3
                relative = rank if color_index else 7 - rank
                midgame += sign * _PASSED_MG[relative]
                endgame += sign * _PASSED_EG[relative]

    if len(_pawn_cache) >= PAWN_CACHE_MAX:
        _pawn_cache.clear()
    result = (midgame, endgame)
    _pawn_cache[key] = result
    return result


_MATE_DRIVE_MARGIN: Final = 400  # a rook up is enough to be trying to mate
_MATE_DRIVE_PHASE: Final = 8  # only once the board has emptied out


def _mating_drive(
    board: chess.Board,
    phase: int,
    material: list[int],
    white_pawns: int,
    black_pawns: int,
) -> int:
    """Push a bare enemy king to the edge and walk our own king towards it.

    Material and piece-square tables alone will happily shuffle a won K+R ending
    forever. Since the referee claims threefold and fifty-move draws against us
    automatically, failing to convert is not a slow win, it is a lost half point.
    This term exists to stop that.

    Returns an endgame-side score from White's perspective.
    """
    difference = material[1] - material[0]
    if phase > _MATE_DRIVE_PHASE or abs(difference) < _MATE_DRIVE_MARGIN:
        return 0
    strong_is_white = difference > 0
    if (black_pawns if strong_is_white else white_pawns):
        return 0  # the losing side still has counterplay; leave this alone

    strong_king = board.king(strong_is_white)
    weak_king = board.king(not strong_is_white)
    if strong_king is None or weak_king is None:
        return 0

    weak_file, weak_rank = weak_king & 7, weak_king >> 3
    strong_file, strong_rank = strong_king & 7, strong_king >> 3

    # 0 when the weak king sits in the middle, 6 when it is in a corner.
    cornered = (3 - min(weak_file, 7 - weak_file)) + (3 - min(weak_rank, 7 - weak_rank))
    # 0 when the kings are far apart, 6 when they are nearly touching.
    approach = 7 - max(abs(weak_file - strong_file), abs(weak_rank - strong_rank))

    drive = cornered * 12 + approach * 10
    return drive if strong_is_white else -drive


def evaluate(board: chess.Board) -> int:
    """Static evaluation in centipawns, from the perspective of the side to move."""
    midgame = 0
    endgame = 0
    phase = 0

    white_pawns = board.pawns & board.occupied_co[chess.WHITE]
    black_pawns = board.pawns & board.occupied_co[chess.BLACK]
    all_pawns = board.pawns

    material = [0, 0]
    for color_index in (1, 0):
        color = bool(color_index)
        sign = 1 if color_index else -1
        own_pawns = white_pawns if color_index else black_pawns

        for piece_type in range(1, 7):
            bits = board.pieces_mask(piece_type, color)
            if not bits:
                continue
            mg_table = MG_TABLE[piece_type]
            eg_table = EG_TABLE[piece_type]
            mg_value = PIECE_MG[piece_type]
            eg_value = PIECE_EG[piece_type]
            weight = PHASE_WEIGHT[piece_type]
            while bits:
                square = (bits & -bits).bit_length() - 1
                bits &= bits - 1
                index = square if color_index else square ^ 56
                midgame += sign * (mg_value + mg_table[index])
                endgame += sign * (eg_value + eg_table[index])
                phase += weight
                material[color_index] += eg_value

                if piece_type == chess.ROOK:
                    file_bb = _FILE_BB[square & 7]
                    if not all_pawns & file_bb:
                        midgame += sign * _ROOK_OPEN_FILE
                    elif not own_pawns & file_bb:
                        midgame += sign * _ROOK_SEMI_OPEN_FILE
                elif piece_type == chess.KING:
                    shield = _SHIELD_MASK[color_index][square]
                    missing = 3 - min(3, chess.popcount(own_pawns & shield))
                    midgame += sign * _SHIELD_MISSING * missing

        if chess.popcount(board.bishops & board.occupied_co[color]) >= 2:
            midgame += sign * _BISHOP_PAIR_MG
            endgame += sign * _BISHOP_PAIR_EG

    pawn_mg, pawn_eg = _pawn_structure(white_pawns, black_pawns)
    midgame += pawn_mg
    endgame += pawn_eg
    endgame += _mating_drive(board, phase, material, white_pawns, black_pawns)

    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    score = (midgame * phase + endgame * (TOTAL_PHASE - phase)) // TOTAL_PHASE
    if board.turn != chess.WHITE:
        score = -score
    return score + TEMPO


# --------------------------------------------------------------------------
# 4. Move ordering
# --------------------------------------------------------------------------

# Alpha-beta only pays for itself when good moves come first, so after time
# management this is the highest-leverage code in the file.

_TT_MOVE_SCORE: Final = 1 << 30
_CAPTURE_BASE: Final = 1 << 24
_KILLER_BASE: Final = 1 << 23

_killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 4)]
_history: list[list[int]] = [[0] * 4096, [0] * 4096]


def _captured_type(board: chess.Board, move: chess.Move) -> int | None:
    """Piece type this move captures, handling en passant. None for a quiet move."""
    victim = board.piece_type_at(move.to_square)
    if victim is not None:
        return victim
    if move.to_square == board.ep_square and board.piece_type_at(move.from_square) == chess.PAWN:
        return chess.PAWN
    return None


def _order_score(board: chess.Board, move: chess.Move, ply: int, tt_move: chess.Move | None) -> int:
    if move == tt_move:
        return _TT_MOVE_SCORE
    victim = _captured_type(board, move)
    if victim is not None:
        attacker = board.piece_type_at(move.from_square) or chess.PAWN
        # MVV-LVA: most valuable victim first, cheapest attacker as the tiebreak.
        score = _CAPTURE_BASE + victim * 100 - attacker
        if move.promotion:
            score += move.promotion * 1000
        return score
    if move.promotion:
        return _CAPTURE_BASE + move.promotion * 1000
    killers = _killers[ply]
    if move == killers[0]:
        return _KILLER_BASE + 1
    if move == killers[1]:
        return _KILLER_BASE
    return _history[board.turn][(move.from_square << 6) | move.to_square]


def _quiescence_moves(board: chess.Board) -> list[chess.Move]:
    """Captures plus non-capture promotions, ordered by MVV-LVA."""
    moves = list(board.generate_legal_captures())
    color_index = int(board.turn)
    pawns_on_seventh = board.pawns & board.occupied_co[board.turn] & _SEVENTH_RANK[color_index]
    if pawns_on_seventh:
        empty_eighth = _EIGHTH_RANK[color_index] & ~board.occupied
        if empty_eighth:
            moves.extend(board.generate_legal_moves(pawns_on_seventh, empty_eighth))
    moves.sort(key=lambda move: _order_score(board, move, 0, None), reverse=True)
    return moves


# --------------------------------------------------------------------------
# 5. Search
# --------------------------------------------------------------------------


class _Timeout(Exception):
    """Raised inside the search when the hard deadline passes."""


_tt: dict[object, tuple[int, int, int, chess.Move | None]] = {}
_repetitions: dict[object, int] = {}

_nodes = 0
_deadline = 0.0
_root_turn: chess.Color = chess.WHITE
_contempt = CONTEMPT_LEVEL


def _draw_score(board: chess.Board) -> int:
    """A draw, seen from the side to move at this node."""
    return -_contempt if board.turn == _root_turn else _contempt


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


def _has_pieces(board: chess.Board, color: chess.Color) -> bool:
    """True if this side has more than pawns and a king. Guards the null move."""
    return bool(board.occupied_co[color] & ~(board.pawns | board.kings))


def _quiescence(board: chess.Board, alpha: int, beta: int, ply: int) -> int:
    """Search captures until the position is quiet.

    Without this, the evaluation gets measured mid-exchange and is simply wrong.
    """
    global _nodes
    _nodes += 1
    if not _nodes & 511 and time.monotonic() >= _deadline:
        raise _Timeout

    stand_pat = evaluate(board)
    if ply >= MAX_PLY:
        return stand_pat
    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat

    in_check = board.is_check()
    best = stand_pat
    for move in _quiescence_moves(board):
        if not in_check:
            victim = _captured_type(board, move)
            if victim is not None:
                # Delta pruning: winning this piece outright still would not reach alpha.
                gain = PIECE_MG[victim] + (900 if move.promotion else 0)
                if stand_pat + gain + 200 < alpha:
                    continue
        board.push(move)
        score = -_quiescence(board, -beta, -alpha, ply + 1)
        board.pop()
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    return best


def _negamax(board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
    global _nodes
    _nodes += 1
    if not _nodes & 511 and time.monotonic() >= _deadline:
        raise _Timeout

    key = board._transposition_key()

    if ply > 0:
        # Draw detection. Two-fold inside the search is the right test, not
        # threefold: the referee claims the draw the moment it becomes available,
        # so we want to see a repetition coming one visit earlier.
        if _repetitions.get(key) or board.halfmove_clock >= 100:
            return _draw_score(board)
        if board.is_insufficient_material():
            return _draw_score(board)
        # Mate distance pruning: never go looking for a slower mate than one we have.
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

    tt_move: chess.Move | None = None
    entry = _tt.get(key)
    if entry is not None:
        stored_depth, flag, stored_score, tt_move = entry
        if ply > 0 and stored_depth >= depth:
            score = _from_tt(stored_score, ply)
            if flag == TT_EXACT:
                return score
            if flag == TT_LOWER and score >= beta:
                return score
            if flag == TT_UPPER and score <= alpha:
                return score

    in_check = board.is_check()
    if in_check and ply < MAX_PLY:
        depth += 1  # never drop into quiescence while in check
    if depth <= 0 or ply >= MAX_PLY:
        return _quiescence(board, alpha, beta, ply)

    _repetitions[key] = _repetitions.get(key, 0) + 1
    try:
        # Null move: hand the opponent a free move. If the position still causes a
        # cutoff, it is good enough to stop searching. Skipped in check and with
        # only pawns left, where zugzwang makes it unsound.
        if depth >= 3 and not in_check and beta < MATE_BOUND and _has_pieces(board, board.turn):
            reduction = 2 + (depth > 6)
            board.push(chess.Move.null())
            score = -_negamax(board, depth - 1 - reduction, -beta, -beta + 1, ply + 1)
            board.pop()
            if score >= beta:
                return beta

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else _draw_score(board)
        moves.sort(key=lambda move: _order_score(board, move, ply, tt_move), reverse=True)

        best_score = -INFINITY
        best_move: chess.Move | None = None
        alpha_original = alpha
        color_index = int(board.turn)

        for index, move in enumerate(moves):
            quiet = _captured_type(board, move) is None and move.promotion is None
            reduction = 0
            if quiet and depth >= 3 and index >= 3 and not in_check:
                # Late move reduction: moves this far down the ordering rarely
                # deserve full depth. Anything that beats alpha is re-searched.
                reduction = 1 + (index >= 8 and depth >= 6)

            board.push(move)
            try:
                if index == 0:
                    score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -_negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
                    if score > alpha and reduction:
                        score = -_negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                    if alpha < score < beta:
                        score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        if quiet:
                            killers = _killers[ply]
                            if killers[0] != move:
                                killers[1] = killers[0]
                                killers[0] = move
                            slot = (move.from_square << 6) | move.to_square
                            _history[color_index][slot] += depth * depth
                        break

        if best_score <= alpha_original:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        if len(_tt) >= TT_MAX_ENTRIES:
            _tt.clear()
        _tt[key] = (depth, flag, _to_tt(best_score, ply), best_move)
        return best_score
    finally:
        remaining = _repetitions[key] - 1
        if remaining:
            _repetitions[key] = remaining
        else:
            del _repetitions[key]


def _search_root(
    board: chess.Board, depth: int, alpha: int, beta: int, moves: list[chess.Move]
) -> tuple[int, chess.Move, bool]:
    """One iterative-deepening pass. The bool reports whether any move finished."""
    best_score = -INFINITY
    best_move = moves[0]
    completed = False
    for index, move in enumerate(moves):
        board.push(move)
        try:
            if index == 0:
                score = -_negamax(board, depth - 1, -beta, -alpha, 1)
            else:
                score = -_negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                if alpha < score < beta:
                    score = -_negamax(board, depth - 1, -beta, -alpha, 1)
        finally:
            board.pop()
        completed = True
        if score > best_score:
            best_score = score
            best_move = move
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    return best_score, best_move, completed


# --------------------------------------------------------------------------
# 6. Time management
# --------------------------------------------------------------------------


def _budget(time_left_ms: int, plies_played: int) -> tuple[float, float]:
    """Soft and hard limits in milliseconds.

    Soft is the point past which we stop starting a new depth. Hard is the point
    at which we abandon the search wherever it is. A flag is a loss, so the hard
    limit is enforced inside the search, not only between iterations.
    """
    usable = time_left_ms - OVERHEAD_MS
    if usable <= 0:
        return 0.0, 0.0
    expected_moves = max(20, 50 - plies_played // 2)
    soft = usable / expected_moves + INCREMENT_MS * 0.7
    hard = min(usable * 0.20, max(soft * 2.0, INCREMENT_MS * 0.5))
    soft = min(soft, hard)
    return soft, hard


# Each extra depth costs roughly two to four times the one before, so starting an
# iteration once most of the soft budget is gone means finishing it on the hard
# limit instead. Only start a depth we have a realistic chance of completing.
NEXT_DEPTH_FRACTION: Final = 0.55


# --------------------------------------------------------------------------
# 7. Game-state tracking
# --------------------------------------------------------------------------

# We are handed a FEN and nothing else, but the referee claims threefold and
# fifty-move draws automatically. So we rebuild the game ourselves: each call we
# work out which legal move the opponent played to reach the FEN we were given,
# push it, and keep the running list of position keys. If the chain ever breaks we
# resync from the FEN and start a fresh history, which is the safe failure.

_board: chess.Board | None = None
_history_keys: list[object] = []


def _sync(fen: str) -> chess.Board:
    global _board, _history_keys
    target = chess.Board(fen)
    target_key = target._transposition_key()

    if _board is not None:
        if _board._transposition_key() == target_key:
            return _board
        for move in _board.legal_moves:
            _board.push(move)
            if _board._transposition_key() == target_key:
                _history_keys.append(target_key)
                return _board
            _board.pop()

    _board = target
    _history_keys = [target_key]
    return _board


def _record(move: chess.Move) -> None:
    """Push our own move so the next call only has one opponent move to find."""
    if _board is None:
        return
    _board.push(move)
    _history_keys.append(_board._transposition_key())


# --------------------------------------------------------------------------
# 8. get_move
# --------------------------------------------------------------------------


def _material_balance(board: chess.Board) -> int:
    """Rough centipawn balance for the side to move, used to pick a contempt."""
    total = 0
    for piece_type in range(1, 6):
        value = PIECE_MG[piece_type]
        total += value * chess.popcount(board.pieces_mask(piece_type, chess.WHITE))
        total -= value * chess.popcount(board.pieces_mask(piece_type, chess.BLACK))
    return total if board.turn == chess.WHITE else -total


def _think(fen: str, time_left_ms: int) -> str:
    global _nodes, _deadline, _root_turn, _contempt

    started = time.monotonic()
    board = _sync(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"

    soft_ms, hard_ms = _budget(time_left_ms, len(_history_keys))
    if len(legal) == 1 or hard_ms <= 0:
        _record(legal[0])
        return legal[0].uci()

    _deadline = started + hard_ms / 1000.0
    next_depth_deadline = started + (soft_ms * NEXT_DEPTH_FRACTION) / 1000.0
    _root_turn = board.turn
    _nodes = 0

    balance = _material_balance(board)
    if balance >= CONTEMPT_MARGIN:
        _contempt = CONTEMPT_WINNING
    elif balance <= -CONTEMPT_MARGIN:
        _contempt = CONTEMPT_LOSING
    else:
        _contempt = CONTEMPT_LEVEL

    _repetitions.clear()
    for key in _history_keys:
        _repetitions[key] = _repetitions.get(key, 0) + 1
    for killers in _killers:
        killers[0] = killers[1] = None
    # Age the history rather than discarding it: ordering learned last move is
    # still worth something this move, just less.
    _history[0] = [value >> 3 for value in _history[0]]
    _history[1] = [value >> 3 for value in _history[1]]

    search_board = board.copy(stack=False)
    root_moves = sorted(
        legal, key=lambda move: _order_score(search_board, move, 0, None), reverse=True
    )
    best_move = root_moves[0]
    best_score = -INFINITY
    depth_reached = 0

    try:
        for depth in range(1, MAX_PLY):
            if depth > 1 and time.monotonic() >= next_depth_deadline:
                break

            if depth <= 3 or best_score <= -INFINITY // 2:
                window_low, window_high = -INFINITY, INFINITY
            else:
                window_low, window_high = best_score - 40, best_score + 40

            while True:
                score, move, completed = _search_root(
                    search_board, depth, window_low, window_high, root_moves
                )
                if not completed:
                    break
                if score <= window_low and window_low > -INFINITY:
                    window_low = -INFINITY  # failed low, research on a wide window
                    continue
                if score >= window_high and window_high < INFINITY:
                    window_high = INFINITY  # failed high, research on a wide window
                    continue
                break

            best_score = score
            best_move = move
            depth_reached = depth
            # Keep the best move first next iteration; it is usually best again.
            root_moves.remove(move)
            root_moves.insert(0, move)

            if abs(best_score) >= MATE_BOUND:
                break  # mate found, nothing left to look for
    except _Timeout:
        pass

    if DEBUG:
        elapsed = (time.monotonic() - started) * 1000.0
        print(
            f"depth {depth_reached} score {best_score} nodes {_nodes} "
            f"{elapsed:.0f}ms budget {soft_ms:.0f}/{hard_ms:.0f} move {best_move.uci()}"
        )

    if best_move not in legal:  # paranoia; should be unreachable
        best_move = legal[0]
    _record(best_move)
    return best_move.uci()


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


def _warmup() -> None:
    """Run a short search at import so the first real move is not the cold one.

    Import happens inside a 60 second budget before the clock starts, so this is
    free. It also means a broken build fails during validation rather than on move
    one of a rated game.
    """
    global _board, _history_keys
    with contextlib.suppress(Exception):
        get_move(chess.STARTING_FEN, 3_000)
    _board = None
    _history_keys = []
    _tt.clear()
    _pawn_cache.clear()
    for killers in _killers:
        killers[0] = killers[1] = None
    _history[0] = [0] * 4096
    _history[1] = [0] * 4096


_warmup()
