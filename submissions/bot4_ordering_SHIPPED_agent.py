"""AI Chessathon submission entrypoint — bot4, ordering and pruning.

Same evaluation as bot1_baseline, deliberately: this iteration attacks the
effective branching factor and nothing else, so that a bench result reads as
"search quality" rather than as the sum of two unrelated changes. The only
evaluation edits are two fixes that belong to any build (a colour bias from
floor division, and a K+2B ending that would not convert).

What is new against bot1:
    * static exchange evaluation, used to split winning from losing captures,
      to prune in quiescence, and to prune shallow captures in the main search
    * countermove and 1-ply continuation history alongside butterfly history,
      with gravity and a malus for quiets that failed to cut off
    * staged move generation: the transposition move is tried before
      list(board.legal_moves) is ever called, which at 33 us a generation is the
      single largest saving available
    * internal iterative reduction, reverse futility, razoring, late move
      pruning, SEE pruning, and a log-table late move reduction
    * progressive aspiration widening and a two-tier transposition table that
      ages out instead of clearing mid-search
    * instrumentation for first-move cutoff rate and TT hit rate, which the
      iteration log records as never having been measured

The design priority, in order, is still: never crash, never flag, never play an
illegal move, then play well.

Layout of this file:
    1. Constants and tunables
    2. Evaluation tables, built at import
    3. Evaluation
    4. Static exchange evaluation
    5. Move ordering
    6. Search
    7. Time management
    8. Game-state tracking (repetition history from FEN alone)
    9. get_move
"""

from __future__ import annotations

import contextlib
import math
import time
from typing import Final

import chess

# --------------------------------------------------------------------------
# 1. Constants and tunables
# --------------------------------------------------------------------------

DEBUG: Final = False  # stdout is discarded in rated games; flip on to read validation logs
STATS: Final = False  # ordering and TT counters; costs a little nps, so off by default

INFINITY: Final = 10_000_000
MATE: Final = 30_000
MATE_BOUND: Final = 29_000  # scores past this are mate scores and need ply adjustment
MAX_PLY: Final = 64
_STACK_SIZE: Final = MAX_PLY + 8  # check extensions can push a little past MAX_PLY

# The referee times the whole request/response round trip, not just get_move, and
# flags the moment the clock goes negative. Hold this much back for the IPC hop,
# JSON encoding, and the odd garbage collection pause.
OVERHEAD_MS: Final = 40
INCREMENT_MS: Final = 500

# Measured, not guessed: a key tuple plus a five-tuple value costs about 268
# bytes, so two tiers of this size is roughly 270 MB against a 2 GB container.
# bot1's comment claimed half a kilobyte an entry, which was 2x pessimistic.
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

# Pruning margins. Deliberately conservative: the published numbers are tuned
# for engines seeing millions of nodes a move, and at fifteen thousand nodes a
# second an over-eager margin throws away the only good move near the root.
RFP_MARGIN: Final = 80  # reverse futility, per ply of remaining depth
RFP_MAX_DEPTH: Final = 7
FUTILITY_BASE: Final = 110
FUTILITY_MARGIN: Final = 95  # per ply
FUTILITY_MAX_DEPTH: Final = 6
LMP_MAX_DEPTH: Final = 6
SEE_QUIET_MARGIN: Final = -70  # per ply squared; deeper searches may speculate more
SEE_CAPTURE_MARGIN: Final = -90  # per ply
SEE_PRUNE_MAX_DEPTH: Final = 6
IIR_MIN_DEPTH: Final = 4
DELTA_MARGIN: Final = 190  # quiescence delta pruning

# The referee adjudicates the 300-ply cap on pure material with no positional
# terms at all, so in the last stretch of a long game our own evaluation is
# measuring the wrong thing. Ramp the referee's own formula in over these plies.
ADJUDICATION_START: Final = 240
ADJUDICATION_FULL: Final = 300
ADJUDICATION_WEIGHT: Final = 0.6  # never all the way; positional play still matters

_HISTORY_MAX: Final = 16_384
_CONT_SLOTS: Final = 384  # (piece type - 1) * 64 + square

# --------------------------------------------------------------------------
# 2. Evaluation tables
# --------------------------------------------------------------------------

# Piece values in centipawns, midgame and endgame. Textbook values: pawns and
# rooks gain in the endgame, minor pieces stay flat.
PIECE_MG: Final = (0, 100, 320, 330, 500, 900, 0)
PIECE_EG: Final = (0, 120, 320, 340, 550, 950, 0)

# The referee's adjudication values, in centipawns. P1 N3 B3 R5 Q9, king zero.
REFEREE_VALUE: Final = (0, 100, 300, 300, 500, 900, 0)

# Phase weights: 24 at a full board, 0 in a bare pawn endgame.
PHASE_WEIGHT: Final = (0, 0, 1, 1, 2, 4, 0)
TOTAL_PHASE: Final = 24


def _flatten(rows: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Turn a table written rank 8 first into one indexed by python-chess square."""
    flat: list[int] = []
    for rank in range(8):
        flat.extend(rows[7 - rank])
    return tuple(flat)


# These tables are bot1's, unchanged. Tuning them is a separate iteration and
# mixing it into this one would make the bench result uninterpretable.

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

# Late move reduction table, indexed [depth][move index]. Logarithmic, which
# reduces late moves hard at high depth without touching a shallow search where
# there is no depth left to give away.
_LMR: Final = tuple(
    tuple(
        0 if depth == 0 or index == 0 else int(0.80 + math.log(depth) * math.log(index) / 2.30)
        for index in range(64)
    )
    for depth in range(64)
)


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

    The minor-piece attraction term is the fix for K+2B, which the iteration log
    records as the one elementary mate bot1 could not convert. Cornering the king
    is not enough when the mating net needs two bishops brought up alongside it;
    a bare corner drive leaves them centralised where their own tables want them.

    Returns an endgame-side score from White's perspective.
    """
    difference = material[1] - material[0]
    if phase > _MATE_DRIVE_PHASE or abs(difference) < _MATE_DRIVE_MARGIN:
        return 0
    strong_is_white = difference > 0
    if black_pawns if strong_is_white else white_pawns:
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

    drive = cornered * 14 + approach * 12

    minors = board.occupied_co[strong_is_white] & (board.bishops | board.knights)
    while minors:
        square = (minors & -minors).bit_length() - 1
        minors &= minors - 1
        distance = max(abs((square & 7) - weak_file), abs((square >> 3) - weak_rank))
        drive += (7 - distance) * 4

    return drive if strong_is_white else -drive


_adjudication_blend = 0.0


def _referee_material(board: chess.Board) -> int:
    """The referee's own adjudication score, from White's perspective.

    Ten popcounts, and only paid in the last stretch of a long game, so this
    never shows up in the node rate of a normal position.
    """
    total = 0
    for piece_type in range(1, 6):
        value = REFEREE_VALUE[piece_type]
        total += value * chess.popcount(board.pieces_mask(piece_type, chess.WHITE))
        total -= value * chess.popcount(board.pieces_mask(piece_type, chess.BLACK))
    return total


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
    # Truncate toward zero, not toward negative infinity. Floor division here is
    # bot1's regression bug #5: a position and its mirror differ by one
    # centipawn, which is a colour bias, and it fired on a third of positions.
    blended = midgame * phase + endgame * (TOTAL_PHASE - phase)
    score = blended // TOTAL_PHASE if blended >= 0 else -(-blended // TOTAL_PHASE)

    if _adjudication_blend:
        # Near the 300-ply cap the referee stops caring about position entirely,
        # so being a bishop up for "compensation" is a loss on the count.
        pure = _referee_material(board)
        score = int(score * (1.0 - _adjudication_blend) + pure * _adjudication_blend)

    if board.turn != chess.WHITE:
        score = -score
    return score + TEMPO


# --------------------------------------------------------------------------
# 4. Static exchange evaluation
# --------------------------------------------------------------------------

# The king is priced far above a queen so that a sequence is never credited with
# winning it; illegal king recaptures are cut out explicitly below.
_SEE_VALUES: Final = (0, 100, 320, 330, 500, 900, 20_000)


def _attackers_to(board: chess.Board, color: chess.Color, square: int, occupied: int) -> int:
    """Pieces of `color` attacking `square` under a hypothetical occupancy.

    python-chess only exposes attackers against the board's real occupancy, and
    SEE needs x-rays to open as pieces come off the square, so the attack tables
    are queried directly here. A piece standing on `square` is never returned,
    because no piece attacks the square it occupies, which is what lets the
    swap-off below leave captured pieces nominally in place.
    """
    queens_rooks = board.queens | board.rooks
    queens_bishops = board.queens | board.bishops
    attackers = (
        (chess.BB_KING_ATTACKS[square] & board.kings)
        | (chess.BB_KNIGHT_ATTACKS[square] & board.knights)
        | (chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & occupied] & queens_rooks)
        | (chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & occupied] & queens_rooks)
        | (chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & occupied] & queens_bishops)
        | (chess.BB_PAWN_ATTACKS[not color][square] & board.pawns)
    )
    return attackers & board.occupied_co[color] & occupied


def _see(board: chess.Board, move: chess.Move) -> int:
    """Static exchange evaluation of a capture, in centipawns for the mover.

    Pins are ignored, as they are in essentially every engine: the error rate is
    a couple of signs in a thousand captures and always conservative. Promotion
    on a recapture is also ignored, being both rare and small.
    """
    to_square = move.to_square
    from_square = move.from_square
    mover = board.turn
    occupied = board.occupied

    victim = board.piece_type_at(to_square)
    if victim is not None:
        balance = _SEE_VALUES[victim]
    elif to_square == board.ep_square and board.piece_type_at(from_square) == chess.PAWN:
        captured = to_square - 8 if mover == chess.WHITE else to_square + 8
        occupied &= ~chess.BB_SQUARES[captured]
        balance = _SEE_VALUES[chess.PAWN]
    else:
        balance = 0

    attacker = board.piece_type_at(from_square)
    if attacker is None:
        return 0
    if move.promotion:
        balance += _SEE_VALUES[move.promotion] - _SEE_VALUES[chess.PAWN]
        attacker = move.promotion

    occupied = (occupied & ~chess.BB_SQUARES[from_square]) | chess.BB_SQUARES[to_square]
    by_type = (0, board.pawns, board.knights, board.bishops, board.rooks, board.queens,
               board.kings)
    gains = [balance]
    standing = attacker
    side = not mover

    while True:
        attackers = _attackers_to(board, side, to_square, occupied)
        if not attackers:
            break
        chosen = 0
        chosen_square = 0
        for piece_type in range(1, 7):
            subset = attackers & by_type[piece_type]
            if subset:
                chosen = piece_type
                chosen_square = (subset & -subset).bit_length() - 1
                break
        if not chosen:
            break
        if chosen == chess.KING and _attackers_to(
            board, not side, to_square, occupied & ~chess.BB_SQUARES[chosen_square]
        ):
            # bot2's regression bug #4: a king cannot recapture onto a square the
            # other side still guards, and crediting that recapture makes winning
            # captures score as losing ones.
            break
        gains.append(_SEE_VALUES[standing] - gains[-1])
        standing = chosen
        occupied &= ~chess.BB_SQUARES[chosen_square]
        side = not side

    for index in range(len(gains) - 1, 0, -1):
        if gains[index - 1] > -gains[index]:
            gains[index - 1] = -gains[index]
    return gains[0]


# --------------------------------------------------------------------------
# 5. Move ordering
# --------------------------------------------------------------------------

# Alpha-beta only pays for itself when good moves come first, so after time
# management this is the highest-leverage code in the file. The buckets below are
# spaced so that the quiet-move history range can never collide with a capture
# or a killer, whatever the tables learn.

_WINNING_CAPTURE: Final = 1 << 26
_PROMOTION: Final = 1 << 25
_KILLER_FIRST: Final = 1 << 24
_KILLER_SECOND: Final = (1 << 24) - 1
_COUNTER_MOVE: Final = 1 << 23
_LOSING_CAPTURE: Final = -(1 << 22)

_killers: list[list[chess.Move | None]] = [[None, None] for _ in range(_STACK_SIZE)]
_history: list[list[int]] = [[0] * 4096, [0] * 4096]
_counters: list[list[chess.Move | None]] = [[None] * 4096, [None] * 4096]
_continuation: list[int] = [0] * (_CONT_SLOTS * _CONT_SLOTS)

# Per-ply search stack. Static evals drive the improving flag and the futility
# margins; continuation indices are how a quiet move is scored by what preceded
# it, which is the largest ordering gain left after killers and history.
_stack_eval: list[int] = [0] * _STACK_SIZE
_stack_cont: list[int] = [-1] * _STACK_SIZE
_stack_move: list[chess.Move | None] = [None] * _STACK_SIZE


def _captured_type(board: chess.Board, move: chess.Move) -> int | None:
    """Piece type this move captures, handling en passant. None for a quiet move.

    En passant returning None was bot2's regression bug #6: the capture then gets
    ordered by history and exposed to reduction and futility pruning as a quiet.
    """
    victim = board.piece_type_at(move.to_square)
    if victim is not None:
        return victim
    if move.to_square == board.ep_square and board.piece_type_at(move.from_square) == chess.PAWN:
        return chess.PAWN
    return None


def _cont_index(piece_type: int, square: int) -> int:
    return (piece_type - 1) * 64 + square


def _order_score(board: chess.Board, move: chess.Move, ply: int) -> int:
    """Score one move for ordering. Higher is tried earlier."""
    victim = _captured_type(board, move)
    attacker = board.piece_type_at(move.from_square) or chess.PAWN

    if victim is not None:
        # MVV-LVA inside the bucket; SEE only decides which bucket. Running SEE
        # on every capture would cost more than it returns, so it is skipped
        # whenever the victim already outranks the attacker and the exchange
        # cannot possibly lose material.
        base = victim * 100 - attacker + (move.promotion or 0) * 1000
        if _SEE_VALUES[victim] >= _SEE_VALUES[attacker]:
            return _WINNING_CAPTURE + base
        gain = _see(board, move)
        if gain >= 0:
            return _WINNING_CAPTURE + base
        return _LOSING_CAPTURE + gain

    if move.promotion:
        return _PROMOTION + move.promotion * 1000

    killers = _killers[ply]
    if move == killers[0]:
        return _KILLER_FIRST
    if move == killers[1]:
        return _KILLER_SECOND

    previous = _stack_move[ply]
    if previous is not None:
        counter = _counters[board.turn][(previous.from_square << 6) | previous.to_square]
        if counter is not None and counter == move:
            return _COUNTER_MOVE

    slot = (move.from_square << 6) | move.to_square
    score = _history[board.turn][slot]
    parent = _stack_cont[ply]
    if parent >= 0:
        score += _continuation[parent * _CONT_SLOTS + _cont_index(attacker, move.to_square)]
    return score


def _record_good_quiet(
    board: chess.Board, move: chess.Move, ply: int, depth: int, parent: int
) -> None:
    """Credit a quiet move that caused a cutoff, in every table that saw it."""
    killers = _killers[ply]
    if killers[0] != move:
        killers[1] = killers[0]
        killers[0] = move

    color = board.turn
    bonus = min(depth * depth * 5 + depth * 40, 1_400)
    slot = (move.from_square << 6) | move.to_square
    table = _history[color]
    table[slot] += bonus - table[slot] * bonus // _HISTORY_MAX

    if parent >= 0:
        attacker = board.piece_type_at(move.from_square) or chess.PAWN
        index = parent * _CONT_SLOTS + _cont_index(attacker, move.to_square)
        _continuation[index] += bonus - _continuation[index] * bonus // _HISTORY_MAX

    previous = _stack_move[ply]
    if previous is not None:
        _counters[color][(previous.from_square << 6) | previous.to_square] = move


def _record_bad_quiet(slots: list[tuple[int, int]], color: chess.Color, depth: int) -> None:
    """Penalise the quiets that were searched and failed before the cutoff.

    Without the malus, history only ever learns which moves are good and never
    which are merely plausible, and plausible-but-wrong moves keep being ordered
    ahead of the rest of the list.
    """
    malus = min(depth * depth * 4 + depth * 30, 1_000)
    table = _history[color]
    for slot, cont in slots:
        table[slot] -= malus + table[slot] * malus // _HISTORY_MAX
        if cont >= 0:
            _continuation[cont] -= malus + _continuation[cont] * malus // _HISTORY_MAX


def _quiescence_moves(board: chess.Board, in_check: bool, ply: int) -> list[chess.Move]:
    """Captures plus non-capture promotions, or every legal move when in check."""
    if in_check:
        moves = list(board.legal_moves)
    else:
        moves = list(board.generate_legal_captures())
        color_index = int(board.turn)
        seventh = board.pawns & board.occupied_co[board.turn] & _SEVENTH_RANK[color_index]
        if seventh:
            empty_eighth = _EIGHTH_RANK[color_index] & ~board.occupied
            if empty_eighth:
                moves.extend(board.generate_legal_moves(seventh, empty_eighth))
    moves.sort(key=lambda move: _order_score(board, move, ply), reverse=True)
    return moves


# --------------------------------------------------------------------------
# 6. Search
# --------------------------------------------------------------------------


class _Timeout(Exception):
    """Raised inside the search when the hard deadline passes."""


# Two tiers. When the live table fills it becomes the old tier and a fresh dict
# takes over, which ages half the table out in constant time. bot1 called
# dict.clear() on half a million entries in the middle of a search instead.
_tt: dict[object, tuple[int, int, int, chess.Move | None, int]] = {}
_tt_old: dict[object, tuple[int, int, int, chess.Move | None, int]] = {}
_repetitions: dict[object, int] = {}

_nodes = 0
_qnodes = 0
_deadline = 0.0
_root_turn: chess.Color = chess.WHITE
_contempt = CONTEMPT_LEVEL
_seldepth = 0

# Instrumentation. The iteration log records first-move cutoff rate as never
# having been measured, which means every ordering change so far was made blind.
_tt_probes = 0
_tt_hits = 0
_cutoffs = 0
_first_move_cutoffs = 0

_NO_EVAL: Final = -INFINITY


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


def _tt_probe(key: object) -> tuple[int, int, int, chess.Move | None, int] | None:
    global _tt_probes, _tt_hits
    if STATS:
        _tt_probes += 1
    entry = _tt.get(key)
    if entry is None:
        entry = _tt_old.get(key)
        if entry is None:
            return None
        _tt[key] = entry  # promote, so a live line stops depending on the old tier
    if STATS:
        _tt_hits += 1
    return entry


def _tt_store(
    key: object, depth: int, flag: int, score: int, move: chess.Move | None, static: int
) -> None:
    global _tt, _tt_old
    existing = _tt.get(key)
    if existing is not None and existing[0] > depth + 2 and flag != TT_EXACT:
        return  # depth-preferred: do not overwrite a deeper result with a bound
    if len(_tt) >= TT_MAX_ENTRIES:
        _tt_old = _tt
        _tt = {}
    _tt[key] = (depth, flag, score, move, static)


def _has_pieces(board: chess.Board, color: chess.Color) -> bool:
    """True if this side has more than pawns and a king. Guards the null move."""
    return bool(board.occupied_co[color] & ~(board.pawns | board.kings))


def _quiescence(board: chess.Board, alpha: int, beta: int, ply: int) -> int:
    """Search captures until the position is quiet.

    Without this, the evaluation gets measured mid-exchange and is simply wrong.
    """
    global _nodes, _qnodes, _seldepth
    _nodes += 1
    if STATS:
        _qnodes += 1
        if ply > _seldepth:
            _seldepth = ply
    if not _nodes & 255 and time.monotonic() >= _deadline:
        raise _Timeout

    in_check = board.is_check()

    if in_check:
        # bot2's regression bug #3: standing pat while in check hands back a
        # material score for a position that is actually mate. A capture that
        # gives check drops straight into this path, so it is not a rare case.
        if ply >= MAX_PLY:
            return evaluate(board)
        best = -INFINITY
    else:
        stand_pat = evaluate(board)
        if ply >= MAX_PLY:
            return stand_pat
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
        best = stand_pat

    moves = _quiescence_moves(board, in_check, ply)
    if in_check and not moves:
        return -MATE + ply

    for move in moves:
        if not in_check:
            victim = _captured_type(board, move)
            if victim is not None:
                # Delta pruning: winning this piece outright still would not
                # reach alpha, so the capture cannot matter.
                gain = PIECE_MG[victim] + (900 if move.promotion else 0)
                if best + gain + DELTA_MARGIN < alpha:
                    continue
                if _SEE_VALUES[victim] < _SEE_VALUES[
                    board.piece_type_at(move.from_square) or chess.PAWN
                ] and _see(board, move) < 0:
                    continue  # a losing exchange never quietens the position
        board.push(move)
        _stack_move[ply + 1] = move
        _stack_cont[ply + 1] = -1
        try:
            score = -_quiescence(board, -beta, -alpha, ply + 1)
        finally:
            board.pop()
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    return best


def _negamax(
    board: chess.Board, depth: int, alpha: int, beta: int, ply: int, can_null: bool
) -> int:
    global _nodes, _cutoffs, _first_move_cutoffs, _seldepth
    _nodes += 1
    if STATS and ply > _seldepth:
        _seldepth = ply
    if not _nodes & 255 and time.monotonic() >= _deadline:
        raise _Timeout

    is_pv = beta > alpha + 1
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

    in_check = board.is_check()
    if in_check and ply < MAX_PLY:
        depth += 1  # never drop into quiescence while in check
    if depth <= 0 or ply >= MAX_PLY:
        return _quiescence(board, alpha, beta, ply)

    # Static eval, once per node, reused by every margin below.
    if in_check:
        static = _NO_EVAL
    elif tt_static != _NO_EVAL:
        static = tt_static
    else:
        static = evaluate(board)
    _stack_eval[ply] = static

    improving = (
        static != _NO_EVAL
        and ply >= 2
        and _stack_eval[ply - 2] != _NO_EVAL
        and static > _stack_eval[ply - 2]
    )

    # Forward pruning. Every branch here is gated on not being in check, not
    # being a PV node, and not having a mate score in the window: those three
    # together are what stop a pruning margin from silently dropping a mate or
    # pruning the only defence. It presents as a slightly weaker evaluation
    # rather than as a bug, which is why the gates are explicit and shared.
    prunable = (
        not in_check and not is_pv and abs(beta) < MATE_BOUND and abs(alpha) < MATE_BOUND
    )

    if prunable and static != _NO_EVAL:
        if depth <= RFP_MAX_DEPTH:
            margin = RFP_MARGIN * depth - (RFP_MARGIN // 2 if improving else 0)
            if static - margin >= beta:
                return static - margin

        # Razoring was tried here and removed. It drops straight to quiescence
        # when the static score is far below alpha, which is precisely the
        # position you are in three plies into a queen sacrifice: it returned a
        # material score from inside a forced mate and cost the selftest's
        # mate-in-3. It returns before a single move is generated, so no
        # check-giving guard can save it. Worth a few Elo of speed at best.

        if (
            can_null
            and depth >= 3
            and static >= beta
            and _has_pieces(board, board.turn)
        ):
            # Hand the opponent a free move. If the position still cuts off, it is
            # good enough to stop searching. Unsound in check and with only pawns
            # left, where zugzwang makes a free move a real concession.
            reduction = 3 + depth // 5 + min((static - beta) // 200, 3)
            board.push(chess.Move.null())
            _stack_move[ply + 1] = None
            _stack_cont[ply + 1] = -1
            try:
                score = -_negamax(
                    board, max(0, depth - 1 - reduction), -beta, -beta + 1, ply + 1, False
                )
            finally:
                board.pop()
            if score >= beta:
                if depth < 10:
                    return beta if score >= MATE_BOUND else score
                # Deep enough that a zugzwang miss is expensive: verify without
                # the null move before trusting it.
                verified = _negamax(board, depth - 4, beta - 1, beta, ply, False)
                if verified >= beta:
                    return verified

    # Internal iterative reduction. With no transposition move the ordering at
    # this node is guesswork, so a full-depth search of it is poor value; give up
    # a ply rather than pay for a separate reduced search to find a move.
    if tt_move is None and depth >= IIR_MIN_DEPTH and not in_check:
        depth -= 1

    _repetitions[key] = _repetitions.get(key, 0) + 1
    try:
        parent = _stack_cont[ply]
        color = board.turn
        futility_base = (
            static + FUTILITY_BASE + FUTILITY_MARGIN * depth if static != _NO_EVAL else INFINITY
        )
        lmp_limit = (3 + depth * depth) // (1 if improving else 2)

        # Staged generation. The transposition move is tried before
        # list(board.legal_moves) is ever called; at 33 us a generation against
        # 1 us for a hash key, a cutoff on the hash move that skips generation
        # entirely is the largest single saving available in this engine.
        #
        # The root never takes this path. Returning a depth-1 move while
        # apparently searching deep was bot2's regression bug #1, and it looked
        # like an evaluation problem rather than an ordering one.
        moves: list[chess.Move] = []
        if tt_move is not None and board.is_legal(tt_move):
            moves.append(tt_move)
        expanded = False
        index = 0
        legal_count = len(moves)

        best_score = -INFINITY
        best_move: chess.Move | None = None
        alpha_original = alpha
        bad_quiets: list[tuple[int, int]] | None = None

        while True:
            if index == len(moves):
                if expanded:
                    break
                expanded = True
                rest = list(board.legal_moves)
                if moves:
                    rest.remove(tt_move)  # already searched as stage one
                legal_count = len(moves) + len(rest)
                if legal_count == 0:
                    return -MATE + ply if in_check else _draw_score(board)
                rest.sort(key=lambda move: _order_score(board, move, ply), reverse=True)
                moves.extend(rest)
                if index == len(moves):
                    break

            move = moves[index]
            index += 1

            victim = _captured_type(board, move)
            quiet = victim is None and move.promotion is None
            attacker = board.piece_type_at(move.from_square) or chess.PAWN
            cont = (
                parent * _CONT_SLOTS + _cont_index(attacker, move.to_square)
                if parent >= 0
                else -1
            )

            if prunable and best_move is not None and abs(best_score) < MATE_BOUND:
                pruned = False
                if quiet:
                    if depth <= LMP_MAX_DEPTH and index > lmp_limit:
                        pruned = True  # late move pruning
                    elif depth <= FUTILITY_MAX_DEPTH and futility_base <= alpha:
                        pruned = True  # frontier futility
                    elif depth <= SEE_PRUNE_MAX_DEPTH and _see(
                        board, move
                    ) < SEE_QUIET_MARGIN * depth * depth:
                        pruned = True
                elif (
                    victim is not None
                    and depth <= SEE_PRUNE_MAX_DEPTH
                    and _see(board, move) < SEE_CAPTURE_MARGIN * depth
                ):
                    pruned = True
                # A move that gives check is never pruned here. After a
                # sacrifice the move that saves the position is almost always a
                # quiet check, and pruning it is exactly how an engine misses a
                # forced mate while still reporting a healthy cutoff rate. The
                # test is only paid when a prune would otherwise have fired.
                if pruned and not board.gives_check(move):
                    continue

            reduction = 0
            if quiet and depth >= 3 and index > 2 and not in_check:
                reduction = _LMR[min(depth, 63)][min(index - 1, 63)]
                if is_pv:
                    reduction -= 1
                if not improving:
                    reduction += 1
                if _history[color][(move.from_square << 6) | move.to_square] > _HISTORY_MAX // 4:
                    reduction -= 1
                reduction = max(0, min(reduction, depth - 2))

            board.push(move)
            _stack_move[ply + 1] = move
            _stack_cont[ply + 1] = _cont_index(attacker, move.to_square)
            try:
                if index == 1:
                    score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1, True)
                else:
                    score = -_negamax(
                        board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, True
                    )
                    if score > alpha and reduction:
                        score = -_negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, True)
                    if alpha < score < beta:
                        score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1, True)
            finally:
                # bot2's regression bug #2. An unbalanced pop on a clock abort
                # returns a move for the opponent, and it only ever fires under
                # time pressure. bot1 survived it by accident, because it
                # searched a stack-free copy; this does not rely on that.
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        if STATS:
                            _cutoffs += 1
                            if index == 1:
                                _first_move_cutoffs += 1
                        if quiet:
                            _record_good_quiet(board, move, ply, depth, parent)
                            if bad_quiets:
                                _record_bad_quiet(bad_quiets, color, depth)
                        break
            if quiet:
                if bad_quiets is None:
                    bad_quiets = []
                if len(bad_quiets) < 32:
                    bad_quiets.append(((move.from_square << 6) | move.to_square, cont))

        if best_score <= alpha_original:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        _tt_store(key, depth, flag, _to_tt(best_score, ply), best_move, static)
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
    """One iterative-deepening pass. The bool reports whether any move finished.

    Kept deliberately plain: full legal list, no staging, no reductions. The root
    is where a subtle ordering bug turns into a played blunder rather than a lost
    centipawn, and the saving from reducing thirty root moves is small.
    """
    best_score = -INFINITY
    best_move = moves[0]
    completed = False
    for index, move in enumerate(moves):
        attacker = board.piece_type_at(move.from_square) or chess.PAWN
        board.push(move)
        _stack_move[1] = move
        _stack_cont[1] = _cont_index(attacker, move.to_square)
        try:
            if index == 0:
                score = -_negamax(board, depth - 1, -beta, -alpha, 1, True)
            else:
                score = -_negamax(board, depth - 1, -alpha - 1, -alpha, 1, True)
                if alpha < score < beta:
                    score = -_negamax(board, depth - 1, -beta, -alpha, 1, True)
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
# 7. Time management
# --------------------------------------------------------------------------

# Carried over from bot1 unchanged. It is the one part of that engine with a
# clean bill of health from the selftest's clock discipline check, and changing
# the budget in the same iteration as the search would make a flag impossible to
# attribute.


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

# A root fail-low means the move we were about to play is worse than we thought,
# which is exactly when the extra time is worth spending.
FAIL_LOW_FRACTION: Final = 0.85

# Once the best move has survived this many iterations without the score moving,
# more depth is unlikely to change the decision. Banking the time is worth more.
STABLE_ITERATIONS: Final = 3
STABLE_SCORE_DRIFT: Final = 20


# --------------------------------------------------------------------------
# 8. Game-state tracking
# --------------------------------------------------------------------------

# Carried over from bot1 unchanged. We are handed a FEN and nothing else, but the
# referee claims threefold and fifty-move draws automatically. So we rebuild the
# game ourselves: each call we work out which legal move the opponent played to
# reach the FEN we were given, push it, and keep the running list of position
# keys. If the chain ever breaks we resync from the FEN and start a fresh
# history, which is the safe failure.

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
# 9. get_move
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
    global _nodes, _qnodes, _deadline, _root_turn, _contempt, _seldepth
    global _adjudication_blend, _tt_probes, _tt_hits, _cutoffs, _first_move_cutoffs

    started = time.monotonic()
    board = _sync(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"

    plies_played = len(_history_keys)
    soft_ms, hard_ms = _budget(time_left_ms, plies_played)
    if len(legal) == 1 or hard_ms <= 0:
        _record(legal[0])
        return legal[0].uci()

    _deadline = started + hard_ms / 1000.0
    _root_turn = board.turn
    _nodes = 0
    _qnodes = 0
    _seldepth = 0
    _tt_probes = _tt_hits = _cutoffs = _first_move_cutoffs = 0

    if plies_played >= ADJUDICATION_START:
        span = ADJUDICATION_FULL - ADJUDICATION_START
        ramp = min(1.0, (plies_played - ADJUDICATION_START) / span)
        _adjudication_blend = ramp * ADJUDICATION_WEIGHT
    else:
        _adjudication_blend = 0.0

    balance = _material_balance(board)
    if balance >= CONTEMPT_MARGIN:
        _contempt = CONTEMPT_WINNING
    elif balance <= -CONTEMPT_MARGIN:
        _contempt = CONTEMPT_LOSING
    else:
        _contempt = CONTEMPT_LEVEL

    _repetitions.clear()
    for history_key in _history_keys:
        _repetitions[history_key] = _repetitions.get(history_key, 0) + 1
    for killers in _killers:
        killers[0] = killers[1] = None
    # History and continuation tables are not aged between moves. Gravity keeps
    # them bounded, retaining them across moves is a real ordering gain, and
    # rescaling 150k entries every move costs more than it returns.
    _stack_move[0] = None
    _stack_cont[0] = -1
    _stack_eval[0] = _NO_EVAL
    _stack_eval[1] = _NO_EVAL

    search_board = board.copy(stack=False)
    root_moves = sorted(
        legal, key=lambda move: _order_score(search_board, move, 0), reverse=True
    )
    best_move = root_moves[0]
    best_score = -INFINITY
    depth_reached = 0
    stable = 0
    previous_score = -INFINITY
    failed_low = False

    try:
        for depth in range(1, MAX_PLY):
            if depth > 1:
                fraction = FAIL_LOW_FRACTION if failed_low else NEXT_DEPTH_FRACTION
                if time.monotonic() >= started + (soft_ms * fraction) / 1000.0:
                    break
            failed_low = False

            # Progressive widening. bot1 jumped straight to a full window on any
            # failure, which throws away the whole point of the narrow window.
            if depth <= 3 or best_score <= -INFINITY // 2:
                window_low, window_high = -INFINITY, INFINITY
            else:
                window_low, window_high = best_score - 25, best_score + 25
            delta = 25

            while True:
                score, move, completed = _search_root(
                    search_board, depth, window_low, window_high, root_moves
                )
                if not completed:
                    break
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
            # Keep the best move first next iteration; it is usually best again.
            root_moves.remove(move)
            root_moves.insert(0, move)

            if abs(best_score) >= MATE_BOUND:
                break  # mate found, nothing left to look for
            if stable >= STABLE_ITERATIONS and depth >= 6:
                break  # the decision has stopped moving; bank the rest
    except _Timeout:
        pass

    if DEBUG:
        elapsed = (time.monotonic() - started) * 1000.0
        line = (
            f"depth {depth_reached} seldepth {_seldepth} score {best_score} nodes {_nodes} "
            f"{elapsed:.0f}ms {_nodes / max(elapsed, 1) * 1000:.0f}nps "
            f"budget {soft_ms:.0f}/{hard_ms:.0f} move {best_move.uci()}"
        )
        if STATS and _cutoffs:
            line += (
                f" | first-move cutoffs {_first_move_cutoffs / _cutoffs:.1%}"
                f" tt hit {_tt_hits / max(_tt_probes, 1):.1%}"
                f" qnodes {_qnodes / max(_nodes, 1):.1%}"
            )
        print(line)

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
    global _board, _history_keys, _adjudication_blend
    with contextlib.suppress(Exception):
        get_move(chess.STARTING_FEN, 3_000)
    _board = None
    _history_keys = []
    _adjudication_blend = 0.0
    _tt.clear()
    _tt_old.clear()
    _pawn_cache.clear()
    for killers in _killers:
        killers[0] = killers[1] = None
    _history[0] = [0] * 4096
    _history[1] = [0] * 4096
    _counters[0] = [None] * 4096
    _counters[1] = [None] * 4096
    for index in range(len(_continuation)):
        _continuation[index] = 0


_warmup()
