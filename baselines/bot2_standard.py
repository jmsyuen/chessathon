"""
AI Chessathon agent - classical alpha-beta engine.

Design notes for a reader:

  Search      Principal variation search with iterative deepening and aspiration
              windows.  Null-move pruning, reverse futility, razoring, futility,
              late-move pruning and late-move reductions.  Quiescence on captures
              and promotions with stand-pat, delta pruning and a static-exchange
              filter.  Staged move generation: the transposition move is tried on
              its own before the move list is built, because generating all legal
              moves costs ~33us and most cut nodes never need the rest.

  Evaluation  Tapered midgame/endgame.  Everything is computed on raw bitboard
              integers with int.bit_count(); no SquareSet or piece_map objects are
              allocated anywhere in the hot path.  Material and piece-square terms
              are computed first and the expensive terms (mobility, king safety,
              pawn structure) are skipped when the fast score is already far
              outside the search window.  Pawn structure is cached in a pawn hash.

  Clock       Hard deadline checked inside the node loop.  A best move from the
              last completed iteration is always available, so an abort is safe.
              The referee deducts wall clock and flags the instant it goes
              negative, so the watchdog grace is not treated as usable slack.

  Draws       The referee claims threefold and fifty-move draws automatically and
              we only ever receive a FEN, so the game history is reconstructed by
              matching the incoming FEN against our last known position.  Draw
              scores are offset by a contempt term derived from material, which
              also handles the 300-ply material adjudication: ahead on material we
              avoid repetitions, behind we seek them.

Piece-square tables are generated from stated positional principles in
_build_piece_square_tables() rather than transcribed, so every number in them
traces back to a rule you can read.
"""

import time

import chess

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

MATE = 30000
MATE_IN_MAX = MATE - 256
INFINITE = 31000
MAX_PLY = 64

# Midgame and endgame piece values, in centipawns.
VALUE_MG = {
    chess.PAWN: 100,
    chess.KNIGHT: 325,
    chess.BISHOP: 340,
    chess.ROOK: 500,
    chess.QUEEN: 960,
    chess.KING: 0,
}
VALUE_EG = {
    chess.PAWN: 118,
    chess.KNIGHT: 320,
    chess.BISHOP: 348,
    chess.ROOK: 540,
    chess.QUEEN: 1010,
    chess.KING: 0,
}

# Game phase: 24 at the full opening array, 0 in a bare pawn ending.
PHASE_WEIGHT = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}
PHASE_TOTAL = 24

TEMPO_MG = 12
BISHOP_PAIR_MG = 34
BISHOP_PAIR_EG = 52

DOUBLED_MG, DOUBLED_EG = -11, -26
ISOLATED_MG, ISOLATED_EG = -14, -17
BACKWARD_MG, BACKWARD_EG = -9, -11
PASSED_MG = (0, 5, 9, 18, 36, 62, 98, 0)
PASSED_EG = (0, 13, 21, 37, 66, 112, 172, 0)
PROTECTED_PASSER_EG = 14

ROOK_OPEN_MG, ROOK_OPEN_EG = 26, 12
ROOK_SEMI_MG, ROOK_SEMI_EG = 12, 7
ROOK_SEVENTH_MG, ROOK_SEVENTH_EG = 16, 22

SHIELD_MISSING_MG = -17
KING_OPEN_FILE_MG = -23
# Indexed by weighted count of attackers on the king zone, saturating.
KING_DANGER = (0, 0, 8, 22, 44, 74, 108, 146, 186, 226, 264, 300, 330, 356, 378, 396, 410)

# Mobility is scored as a linear term about a neutral count per piece type.
MOB_MG = {chess.KNIGHT: 5, chess.BISHOP: 4, chess.ROOK: 2, chess.QUEEN: 1}
MOB_EG = {chess.KNIGHT: 4, chess.BISHOP: 4, chess.ROOK: 4, chess.QUEEN: 2}
MOB_NEUTRAL = {chess.KNIGHT: 4, chess.BISHOP: 6, chess.ROOK: 7, chess.QUEEN: 14}

LAZY_MARGIN = 240

# Search margins.
RFP_MARGIN = 78          # reverse futility, per ply of depth
RAZOR_MARGIN = 300       # razoring, per ply of depth
FUTILITY_MARGIN = 110    # frontier futility, per ply of depth
DELTA_MARGIN = 190       # quiescence delta pruning
NULL_MIN_DEPTH = 3
LMR_MIN_DEPTH = 3
LMR_MIN_MOVES = 3
LMP_TABLE = (0, 5, 8, 13, 20, 29, 40)

# --------------------------------------------------------------------------
# Piece-square tables, generated from positional principles
# --------------------------------------------------------------------------


def _centre_distance(square):
    """0 for the four central squares, 3 at the rim."""
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    return max(abs(2 * file_index - 7), abs(2 * rank_index - 7)) // 2


def _build_piece_square_tables():
    """Every table below is a rule, written out.  Indexed from White's side."""
    mg = {piece: [0] * 64 for piece in chess.PIECE_TYPES}
    eg = {piece: [0] * 64 for piece in chess.PIECE_TYPES}

    long_diagonals = chess.BB_A1 | chess.BB_B2 | chess.BB_C3 | chess.BB_D4 | chess.BB_E5 \
        | chess.BB_F6 | chess.BB_G7 | chess.BB_H8 | chess.BB_H1 | chess.BB_G2 | chess.BB_F3 \
        | chess.BB_E4 | chess.BB_D5 | chess.BB_C6 | chess.BB_B7 | chess.BB_A8

    # Pawns: advancement matters a little in the midgame and a great deal in the
    # endgame; central pawns are worth more; the d2/e2 pawns are slightly
    # discouraged from sitting still because they block the bishops.
    advance_mg = (0, 0, 4, 10, 21, 41, 72, 0)
    advance_eg = (0, 8, 15, 27, 51, 96, 162, 0)
    file_weight = (0.55, 0.65, 0.80, 1.00, 1.00, 0.80, 0.65, 0.55)
    for square in chess.SQUARES:
        rank_index = chess.square_rank(square)
        file_index = chess.square_file(square)
        mg[chess.PAWN][square] = round(advance_mg[rank_index] * file_weight[file_index])
        eg[chess.PAWN][square] = advance_eg[rank_index]
        if rank_index == 1 and file_index in (3, 4):
            mg[chess.PAWN][square] -= 11

    # Knights: centralisation dominates, plus a bonus for the outpost ranks and a
    # penalty for sitting on the back rank undeveloped.
    for square in chess.SQUARES:
        distance = _centre_distance(square)
        rank_index = chess.square_rank(square)
        score = -34 + 17 * (3 - distance)
        if rank_index in (3, 4, 5):
            score += (6, 10, 8)[rank_index - 3]
        if rank_index == 0:
            score -= 13
        mg[chess.KNIGHT][square] = score
        eg[chess.KNIGHT][square] = -30 + 14 * (3 - distance)

    # Bishops: mild centralisation, a bonus for the long diagonals, a penalty for
    # the back rank.
    for square in chess.SQUARES:
        distance = _centre_distance(square)
        score = -14 + 7 * (3 - distance)
        if chess.BB_SQUARES[square] & long_diagonals:
            score += 9
        if chess.square_rank(square) == 0:
            score -= 9
        mg[chess.BISHOP][square] = score
        eg[chess.BISHOP][square] = -12 + 6 * (3 - distance)

    # Rooks: central files in the midgame, and the seventh rank.
    rook_file = (-4, 0, 3, 6, 6, 3, 0, -4)
    for square in chess.SQUARES:
        rank_index = chess.square_rank(square)
        score = rook_file[chess.square_file(square)]
        if rank_index == 6:
            score += 22
        elif rank_index == 7:
            score += 6
        mg[chess.ROOK][square] = score
        eg[chess.ROOK][square] = 4 if rank_index >= 4 else 0

    # Queens: slight centralisation, stronger in the endgame.
    for square in chess.SQUARES:
        distance = _centre_distance(square)
        mg[chess.QUEEN][square] = -8 + 4 * (3 - distance)
        eg[chess.QUEEN][square] = -18 + 8 * (3 - distance)

    # King: shelter in the midgame, centralisation in the endgame.
    shelter_rank0 = (24, 32, 14, -12, -12, 10, 32, 22)
    shelter_rank1 = (6, 8, -6, -22, -22, -8, 8, 4)
    shelter_rank2 = (-16, -20, -28, -34, -34, -28, -20, -16)
    for square in chess.SQUARES:
        rank_index = chess.square_rank(square)
        file_index = chess.square_file(square)
        if rank_index == 0:
            score = shelter_rank0[file_index]
        elif rank_index == 1:
            score = shelter_rank1[file_index]
        elif rank_index == 2:
            score = shelter_rank2[file_index]
        else:
            score = -40 - 6 * (rank_index - 3)
        mg[chess.KING][square] = score
        eg[chess.KING][square] = -46 + 16 * (3 - _centre_distance(square))

    return mg, eg


_MG_TABLE, _EG_TABLE = _build_piece_square_tables()

# Flatten into per-colour lookup lists so evaluation is a single index, with the
# piece value already folded in.  Black's tables are vertically mirrored.
PST_MG = {chess.WHITE: {}, chess.BLACK: {}}
PST_EG = {chess.WHITE: {}, chess.BLACK: {}}
for _piece in chess.PIECE_TYPES:
    PST_MG[chess.WHITE][_piece] = tuple(
        _MG_TABLE[_piece][s] + VALUE_MG[_piece] for s in chess.SQUARES
    )
    PST_EG[chess.WHITE][_piece] = tuple(
        _EG_TABLE[_piece][s] + VALUE_EG[_piece] for s in chess.SQUARES
    )
    PST_MG[chess.BLACK][_piece] = tuple(
        _MG_TABLE[_piece][s ^ 56] + VALUE_MG[_piece] for s in chess.SQUARES
    )
    PST_EG[chess.BLACK][_piece] = tuple(
        _EG_TABLE[_piece][s ^ 56] + VALUE_EG[_piece] for s in chess.SQUARES
    )

# --------------------------------------------------------------------------
# Bitboard helpers
# --------------------------------------------------------------------------

BB_DIAG_MASKS = chess.BB_DIAG_MASKS
BB_DIAG_ATTACKS = chess.BB_DIAG_ATTACKS
BB_RANK_MASKS = chess.BB_RANK_MASKS
BB_RANK_ATTACKS = chess.BB_RANK_ATTACKS
BB_FILE_MASKS = chess.BB_FILE_MASKS
BB_FILE_ATTACKS = chess.BB_FILE_ATTACKS
BB_KNIGHT_ATTACKS = chess.BB_KNIGHT_ATTACKS
BB_KING_ATTACKS = chess.BB_KING_ATTACKS
BB_PAWN_ATTACKS = chess.BB_PAWN_ATTACKS
BB_SQUARES = chess.BB_SQUARES
BB_FILES = chess.BB_FILES
BB_RANKS = chess.BB_RANKS
BB_ALL = chess.BB_ALL


def bishop_attacks(square, occupied):
    return BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & occupied]


def rook_attacks(square, occupied):
    return (
        BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied]
        | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied]
    )


def queen_attacks(square, occupied):
    return bishop_attacks(square, occupied) | rook_attacks(square, occupied)


# Squares in front of a pawn on each square, on its own file and the two
# adjacent files: the passed-pawn test.
PASSED_MASK = {chess.WHITE: [0] * 64, chess.BLACK: [0] * 64}
# Squares in front on the pawn's own file only: the doubled/backward test.
FRONT_MASK = {chess.WHITE: [0] * 64, chess.BLACK: [0] * 64}
# The three files around a file, for isolated pawns and king shelter.
NEIGHBOUR_FILES = [0] * 8
for _file in range(8):
    mask = BB_FILES[_file]
    if _file > 0:
        mask |= BB_FILES[_file - 1]
    if _file < 7:
        mask |= BB_FILES[_file + 1]
    NEIGHBOUR_FILES[_file] = mask
for _square in chess.SQUARES:
    _f = chess.square_file(_square)
    _r = chess.square_rank(_square)
    ahead_white = 0
    ahead_black = 0
    for _rank in range(_r + 1, 8):
        ahead_white |= BB_RANKS[_rank]
    for _rank in range(0, _r):
        ahead_black |= BB_RANKS[_rank]
    FRONT_MASK[chess.WHITE][_square] = ahead_white & BB_FILES[_f]
    FRONT_MASK[chess.BLACK][_square] = ahead_black & BB_FILES[_f]
    PASSED_MASK[chess.WHITE][_square] = ahead_white & NEIGHBOUR_FILES[_f]
    PASSED_MASK[chess.BLACK][_square] = ahead_black & NEIGHBOUR_FILES[_f]

# King zone: the king's square plus its neighbours, widened one rank forward.
KING_ZONE = {chess.WHITE: [0] * 64, chess.BLACK: [0] * 64}
for _square in chess.SQUARES:
    zone = BB_KING_ATTACKS[_square] | BB_SQUARES[_square]
    KING_ZONE[chess.WHITE][_square] = zone | (zone << 8) & BB_ALL
    KING_ZONE[chess.BLACK][_square] = zone | (zone >> 8)

ATTACK_WEIGHT = {chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 5}


def _div(value, divisor):
    """
    Divide truncating toward zero.  Python's // floors toward minus infinity,
    which would make the evaluation of a position and its mirror differ by one
    centipawn and give the engine a systematic bias by colour.
    """
    if value >= 0:
        return value // divisor
    return -((-value) // divisor)


def _squares(bitboard):
    """Iterate set bits without allocating a SquareSet."""
    while bitboard:
        low = bitboard & -bitboard
        yield low.bit_length() - 1
        bitboard ^= low


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


class Evaluator:
    def __init__(self):
        self.pawn_hash = {}

    def evaluate(self, board, alpha=-INFINITE, beta=INFINITE):
        """Score in centipawns from the side to move's point of view."""
        white = board.occupied_co[chess.WHITE]
        black = board.occupied_co[chess.BLACK]
        pawns = board.pawns
        knights = board.knights
        bishops = board.bishops
        rooks = board.rooks
        queens = board.queens
        kings = board.kings

        mg = 0
        eg = 0
        phase = 0

        pst_mg_w = PST_MG[chess.WHITE]
        pst_eg_w = PST_EG[chess.WHITE]
        pst_mg_b = PST_MG[chess.BLACK]
        pst_eg_b = PST_EG[chess.BLACK]

        for piece_type, bitboard in (
            (chess.PAWN, pawns),
            (chess.KNIGHT, knights),
            (chess.BISHOP, bishops),
            (chess.ROOK, rooks),
            (chess.QUEEN, queens),
            (chess.KING, kings),
        ):
            weight = PHASE_WEIGHT[piece_type]
            table_mg = pst_mg_w[piece_type]
            table_eg = pst_eg_w[piece_type]
            own = bitboard & white
            while own:
                low = own & -own
                square = low.bit_length() - 1
                mg += table_mg[square]
                eg += table_eg[square]
                phase += weight
                own ^= low
            table_mg = pst_mg_b[piece_type]
            table_eg = pst_eg_b[piece_type]
            other = bitboard & black
            while other:
                low = other & -other
                square = low.bit_length() - 1
                mg -= table_mg[square]
                eg -= table_eg[square]
                phase += weight
                other ^= low

        # Bishop pair.
        if (bishops & white).bit_count() >= 2:
            mg += BISHOP_PAIR_MG
            eg += BISHOP_PAIR_EG
        if (bishops & black).bit_count() >= 2:
            mg -= BISHOP_PAIR_MG
            eg -= BISHOP_PAIR_EG

        if phase > PHASE_TOTAL:
            phase = PHASE_TOTAL

        if board.turn == chess.WHITE:
            mg += TEMPO_MG
        else:
            mg -= TEMPO_MG

        fast = _div(mg * phase + eg * (PHASE_TOTAL - phase), PHASE_TOTAL)
        if board.turn == chess.BLACK:
            fast = -fast

        # Lazy exit: the slow terms are bounded in practice, so if the cheap score
        # is already far outside the window they cannot change the outcome.
        if fast - LAZY_MARGIN > beta or fast + LAZY_MARGIN < alpha:
            return fast

        occupied = board.occupied

        # Pawn structure, cached on the pawn skeleton alone.
        key = (pawns & white, pawns & black)
        cached = self.pawn_hash.get(key)
        if cached is None:
            cached = self._pawn_structure(pawns & white, pawns & black)
            if len(self.pawn_hash) > 120000:
                self.pawn_hash.clear()
            self.pawn_hash[key] = cached
        pawn_mg, pawn_eg, passed_white, passed_black = cached
        mg += pawn_mg
        eg += pawn_eg

        # Protected passers are worth more in the endgame.
        white_pawn_attacks = ((pawns & white) << 9) & ~BB_FILES[0] & BB_ALL
        white_pawn_attacks |= ((pawns & white) << 7) & ~BB_FILES[7] & BB_ALL
        black_pawn_attacks = ((pawns & black) >> 7) & ~BB_FILES[0]
        black_pawn_attacks |= ((pawns & black) >> 9) & ~BB_FILES[7]
        eg += PROTECTED_PASSER_EG * (passed_white & white_pawn_attacks).bit_count()
        eg -= PROTECTED_PASSER_EG * (passed_black & black_pawn_attacks).bit_count()

        white_king = (kings & white).bit_length() - 1
        black_king = (kings & black).bit_length() - 1

        # Mobility, rook files and king attacks, in one pass per colour.
        for colour, own, enemy, enemy_king, sign in (
            (chess.WHITE, white, black, black_king, 1),
            (chess.BLACK, black, white, white_king, -1),
        ):
            enemy_pawn_attacks = black_pawn_attacks if colour == chess.WHITE else white_pawn_attacks
            safe = ~(own | enemy_pawn_attacks)
            zone = KING_ZONE[chess.BLACK if colour == chess.WHITE else chess.WHITE][enemy_king]
            danger = 0

            for square in _squares(knights & own):
                attacks = BB_KNIGHT_ATTACKS[square]
                count = (attacks & safe).bit_count()
                mg += sign * MOB_MG[chess.KNIGHT] * (count - MOB_NEUTRAL[chess.KNIGHT])
                eg += sign * MOB_EG[chess.KNIGHT] * (count - MOB_NEUTRAL[chess.KNIGHT])
                if attacks & zone:
                    danger += ATTACK_WEIGHT[chess.KNIGHT]

            for square in _squares(bishops & own):
                attacks = bishop_attacks(square, occupied)
                count = (attacks & safe).bit_count()
                mg += sign * MOB_MG[chess.BISHOP] * (count - MOB_NEUTRAL[chess.BISHOP])
                eg += sign * MOB_EG[chess.BISHOP] * (count - MOB_NEUTRAL[chess.BISHOP])
                if attacks & zone:
                    danger += ATTACK_WEIGHT[chess.BISHOP]

            for square in _squares(rooks & own):
                attacks = rook_attacks(square, occupied)
                count = (attacks & safe).bit_count()
                mg += sign * MOB_MG[chess.ROOK] * (count - MOB_NEUTRAL[chess.ROOK])
                eg += sign * MOB_EG[chess.ROOK] * (count - MOB_NEUTRAL[chess.ROOK])
                if attacks & zone:
                    danger += ATTACK_WEIGHT[chess.ROOK]
                file_bb = BB_FILES[square & 7]
                if not (file_bb & pawns & own):
                    if file_bb & pawns & enemy:
                        mg += sign * ROOK_SEMI_MG
                        eg += sign * ROOK_SEMI_EG
                    else:
                        mg += sign * ROOK_OPEN_MG
                        eg += sign * ROOK_OPEN_EG
                relative_rank = (square >> 3) if colour == chess.WHITE else 7 - (square >> 3)
                if relative_rank == 6:
                    mg += sign * ROOK_SEVENTH_MG
                    eg += sign * ROOK_SEVENTH_EG

            for square in _squares(queens & own):
                attacks = queen_attacks(square, occupied)
                count = (attacks & safe).bit_count()
                mg += sign * MOB_MG[chess.QUEEN] * (count - MOB_NEUTRAL[chess.QUEEN])
                eg += sign * MOB_EG[chess.QUEEN] * (count - MOB_NEUTRAL[chess.QUEEN])
                if attacks & zone:
                    danger += ATTACK_WEIGHT[chess.QUEEN]

            if danger >= len(KING_DANGER):
                danger = len(KING_DANGER) - 1
            mg += sign * KING_DANGER[danger]

        # King shelter: missing pawns in front of the king, and open files beside it.
        for colour, own, king_square, sign in (
            (chess.WHITE, white, white_king, 1),
            (chess.BLACK, black, black_king, -1),
        ):
            king_file = king_square & 7
            king_rank = king_square >> 3
            shelter = NEIGHBOUR_FILES[king_file] & pawns & own
            # Only pawns strictly in front of the king shelter it.
            if colour == chess.WHITE:
                ahead = ~((1 << ((king_rank + 1) * 8)) - 1) & BB_ALL
            else:
                ahead = (1 << (king_rank * 8)) - 1
            # A king on the a- or h-file only has two files to be sheltered on.
            files_available = 3 if 0 < king_file < 7 else 2
            missing = files_available - (shelter & ahead).bit_count()
            if missing > 0:
                mg += sign * SHIELD_MISSING_MG * missing
            for file_index in range(max(0, king_file - 1), min(8, king_file + 2)):
                if not (BB_FILES[file_index] & pawns & own):
                    mg += sign * KING_OPEN_FILE_MG

        score = _div(mg * phase + eg * (PHASE_TOTAL - phase), PHASE_TOTAL)

        # Scale down drawish endings: opposite bishops, and no pawns.
        if phase <= 6:
            score = self._scale_endgame(score, board, white, black, pawns, bishops, knights)

        return score if board.turn == chess.WHITE else -score

    @staticmethod
    def _scale_endgame(score, board, white, black, pawns, bishops, knights):
        """Pull scores toward zero in endings the leader cannot actually win."""
        if score == 0:
            return 0
        leader = white if score > 0 else black
        if not (pawns & leader):
            majors = (board.rooks | board.queens) & leader
            minors = (bishops | knights) & leader
            if not majors:
                # A lone minor cannot mate; nor can two knights against a bare king.
                if minors.bit_count() <= 1:
                    return _div(score, 8)
                if not (bishops & leader):
                    return _div(score, 4)
            return _div(score, 2)
        # Opposite-coloured bishops with nothing else on the board are drawish.
        if (bishops & white).bit_count() == 1 and (bishops & black).bit_count() == 1 \
                and not (board.rooks | board.queens | knights):
            light = chess.BB_LIGHT_SQUARES
            if bool(bishops & white & light) != bool(bishops & black & light):
                return _div(score * 3, 5)
        return score

    @staticmethod
    def _pawn_structure(white_pawns, black_pawns):
        mg = 0
        eg = 0
        passed_white = 0
        passed_black = 0

        for square in _squares(white_pawns):
            file_index = square & 7
            if FRONT_MASK[chess.WHITE][square] & white_pawns:
                mg += DOUBLED_MG
                eg += DOUBLED_EG
            if not (NEIGHBOUR_FILES[file_index] & ~BB_FILES[file_index] & white_pawns):
                mg += ISOLATED_MG
                eg += ISOLATED_EG
            elif not (PASSED_MASK[chess.BLACK][square] & NEIGHBOUR_FILES[file_index] & white_pawns):
                mg += BACKWARD_MG
                eg += BACKWARD_EG
            if not (PASSED_MASK[chess.WHITE][square] & black_pawns):
                rank_index = square >> 3
                mg += PASSED_MG[rank_index]
                eg += PASSED_EG[rank_index]
                passed_white |= BB_SQUARES[square]

        for square in _squares(black_pawns):
            file_index = square & 7
            if FRONT_MASK[chess.BLACK][square] & black_pawns:
                mg -= DOUBLED_MG
                eg -= DOUBLED_EG
            if not (NEIGHBOUR_FILES[file_index] & ~BB_FILES[file_index] & black_pawns):
                mg -= ISOLATED_MG
                eg -= ISOLATED_EG
            elif not (PASSED_MASK[chess.WHITE][square] & NEIGHBOUR_FILES[file_index] & black_pawns):
                mg -= BACKWARD_MG
                eg -= BACKWARD_EG
            if not (PASSED_MASK[chess.BLACK][square] & white_pawns):
                rank_index = 7 - (square >> 3)
                mg -= PASSED_MG[rank_index]
                eg -= PASSED_EG[rank_index]
                passed_black |= BB_SQUARES[square]

        return mg, eg, passed_white, passed_black


# --------------------------------------------------------------------------
# Static exchange evaluation
# --------------------------------------------------------------------------

SEE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 325,
    chess.BISHOP: 340,
    chess.ROOK: 500,
    chess.QUEEN: 960,
    chess.KING: 10000,
    None: 0,
}


def _attackers_to(board, square, occupied):
    return (
        (BB_PAWN_ATTACKS[chess.BLACK][square] & board.pawns & board.occupied_co[chess.WHITE])
        | (BB_PAWN_ATTACKS[chess.WHITE][square] & board.pawns & board.occupied_co[chess.BLACK])
        | (BB_KNIGHT_ATTACKS[square] & board.knights)
        | (BB_KING_ATTACKS[square] & board.kings)
        | (bishop_attacks(square, occupied) & (board.bishops | board.queens))
        | (rook_attacks(square, occupied) & (board.rooks | board.queens))
    ) & occupied


def see(board, move, victim_type):
    """Signed value of the capture sequence on move.to_square, in centipawns."""
    to_square = move.to_square
    from_square = move.from_square
    attacker_type = board.piece_type_at(from_square)
    if attacker_type is None:
        return 0

    gains = [SEE_VALUE[victim_type]]
    if move.promotion:
        gains[0] += SEE_VALUE[move.promotion] - SEE_VALUE[chess.PAWN]
        attacker_type = move.promotion

    occupied = board.occupied & ~BB_SQUARES[from_square]
    if victim_type is not None and board.piece_type_at(to_square) is None:
        # En passant: the captured pawn is not on the destination square.
        captured_square = to_square - 8 if board.turn == chess.WHITE else to_square + 8
        occupied &= ~BB_SQUARES[captured_square]

    attackers = _attackers_to(board, to_square, occupied)
    colour = not board.turn
    depth = 0

    while True:
        side_attackers = attackers & board.occupied_co[colour]
        if not side_attackers:
            break
        # Cheapest attacker first.
        for piece_type in (
            chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
        ):
            candidates = side_attackers & board.pieces_mask(piece_type, colour)
            if candidates:
                break
        else:
            break

        # A king may only recapture onto a square the other side no longer
        # defends. Checking this before committing the gain matters: counting an
        # illegal king recapture makes winning captures look like losing ones.
        if piece_type == chess.KING and (attackers & board.occupied_co[not colour]):
            break

        square = (candidates & -candidates).bit_length() - 1
        depth += 1
        gains.append(SEE_VALUE[attacker_type] - gains[depth - 1])
        attacker_type = piece_type
        occupied &= ~BB_SQUARES[square]
        # Recompute to pick up x-ray attackers revealed by the capture.
        attackers = _attackers_to(board, to_square, occupied)
        colour = not colour

    while depth:
        gains[depth - 1] = -max(-gains[depth - 1], gains[depth])
        depth -= 1
    return gains[0]


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


class TimeUp(Exception):
    pass


class Engine:
    def __init__(self):
        self.evaluator = Evaluator()
        self.tt = {}
        self.killers = [[None, None] for _ in range(MAX_PLY + 8)]
        self.history = [[0] * 4096, [0] * 4096]
        self.counter = {}
        self.nodes = 0
        self.deadline = 0.0
        self.key_stack = []
        self.contempt = 0
        self.root_colour = chess.WHITE
        self.seldepth = 0

    # -- utilities ---------------------------------------------------------

    def draw_score(self, board):
        """Contempt-adjusted draw value from the side to move's point of view."""
        return -self.contempt if board.turn == self.root_colour else self.contempt

    def is_repetition(self, board, key):
        stack = self.key_stack
        limit = min(board.halfmove_clock, len(stack) - 1)
        index = len(stack) - 3
        stop = len(stack) - 1 - limit
        while index >= stop:
            if stack[index] == key:
                return True
            index -= 2
        return False

    # -- quiescence --------------------------------------------------------

    def quiescence(self, board, alpha, beta, ply):
        self.nodes += 1
        if not (self.nodes & 511) and time.monotonic() > self.deadline:
            raise TimeUp
        if ply > self.seldepth:
            self.seldepth = ply

        in_check = board.is_check()

        if in_check:
            # Standing pat while in check is unsound: the side to move may be
            # mated and has no right to decline. Search every evasion instead.
            evasions = list(board.generate_legal_moves())
            if not evasions:
                return -MATE + ply
            if ply >= MAX_PLY:
                return self.evaluator.evaluate(board, alpha, beta)
            stand_pat = -INFINITE
            scored = []
            for move in evasions:
                victim = board.piece_type_at(move.to_square)
                gain = SEE_VALUE[victim] if victim is not None else 0
                if move.promotion:
                    gain += SEE_VALUE[move.promotion]
                scored.append((gain, move))
        else:
            stand_pat = self.evaluator.evaluate(board, alpha, beta)
            if stand_pat >= beta:
                return stand_pat
            if ply >= MAX_PLY:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat

            ep_square = board.ep_square
            scored = []
            for move in board.generate_legal_captures():
                to_square = move.to_square
                victim = board.piece_type_at(to_square)
                if victim is None:
                    victim = chess.PAWN  # en passant
                gain = SEE_VALUE[victim]
                if move.promotion:
                    gain += SEE_VALUE[move.promotion] - SEE_VALUE[chess.PAWN]
                # Delta pruning: even winning this material would not reach alpha.
                if stand_pat + gain + DELTA_MARGIN < alpha:
                    continue
                attacker = board.piece_type_at(move.from_square)
                order = gain * 16 - SEE_VALUE[attacker]
                # Only pay for SEE when the capture looks like it might lose material.
                if SEE_VALUE[attacker] > gain and see(board, move, victim) < 0:
                    continue
                scored.append((order, move))
            del ep_square

        scored.sort(key=lambda item: -item[0])

        best = stand_pat
        for _, move in scored:
            board.push(move)
            try:
                score = -self.quiescence(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    # -- move ordering -----------------------------------------------------

    def order_moves(self, board, moves, tt_move, ply, previous):
        killer_a, killer_b = self.killers[ply]
        counter_move = self.counter.get(previous) if previous else None
        history = self.history[board.turn]
        ep_square = board.ep_square
        pawns = board.pawns
        scored = []
        append = scored.append
        for move in moves:
            if move == tt_move:
                continue
            to_square = move.to_square
            victim = board.piece_type_at(to_square)
            if victim is None and to_square == ep_square and (pawns & BB_SQUARES[move.from_square]):
                victim = chess.PAWN  # en passant captures a pawn off the target square
            if victim is None and move.promotion is None:
                # Quiet move.
                if move == killer_a:
                    append((400000, move))
                elif move == killer_b:
                    append((390000, move))
                elif move == counter_move:
                    append((380000, move))
                else:
                    append((history[(move.from_square << 6) | to_square], move))
                continue
            gain = SEE_VALUE[victim]
            if move.promotion:
                gain += SEE_VALUE[move.promotion] - SEE_VALUE[chess.PAWN]
            attacker = SEE_VALUE[board.piece_type_at(move.from_square)]
            order = gain * 16 - attacker
            if attacker > gain and see(board, move, victim) < 0:
                append((order - 900000, move))  # losing capture, search last
            else:
                append((order + 500000, move))
        scored.sort(key=lambda item: -item[0])
        return scored

    # -- main search -------------------------------------------------------

    def negamax(self, board, depth, alpha, beta, ply, previous=None, can_null=True):
        self.nodes += 1
        if not (self.nodes & 511) and time.monotonic() > self.deadline:
            raise TimeUp

        is_pv = beta - alpha > 1
        key = board._transposition_key()

        if ply > 0:
            if board.halfmove_clock >= 100 or self.is_repetition(board, key):
                return self.draw_score(board)
            # Mate distance pruning.
            if alpha < -MATE + ply:
                alpha = -MATE + ply
            if beta > MATE - ply - 1:
                beta = MATE - ply - 1
            if alpha >= beta:
                return alpha

        entry = self.tt.get(key)
        tt_move = None
        if entry is not None:
            tt_depth, tt_flag, tt_score, tt_move = entry
            if tt_depth >= depth and ply > 0:
                if tt_score > MATE_IN_MAX:
                    tt_score -= ply
                elif tt_score < -MATE_IN_MAX:
                    tt_score += ply
                if tt_flag == 0:
                    return tt_score
                if tt_flag == 1 and tt_score >= beta:
                    return tt_score
                if tt_flag == 2 and tt_score <= alpha:
                    return tt_score

        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)

        in_check = board.is_check()
        if in_check:
            depth += 1  # check extension

        static = None
        if not in_check:
            static = self.evaluator.evaluate(board, alpha, beta)

            if not is_pv and depth <= 6 and static - RFP_MARGIN * depth >= beta \
                    and abs(beta) < MATE_IN_MAX:
                return static

            if not is_pv and depth <= 3 and static + RAZOR_MARGIN * depth < alpha:
                score = self.quiescence(board, alpha, beta, ply)
                if score < alpha:
                    return score

            # Null move: give the opponent a free move; if we are still winning,
            # this node is very likely a cut node.  Skipped in pawn endings,
            # where zugzwang makes the assumption false.
            if (
                can_null
                and not is_pv
                and depth >= NULL_MIN_DEPTH
                and static >= beta
                and (board.occupied_co[board.turn] & ~board.pawns & ~board.kings)
            ):
                reduction = 2 + depth // 6
                board.push(chess.Move.null())
                self.key_stack.append(board._transposition_key())
                try:
                    score = -self.negamax(
                        board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, None, False
                    )
                finally:
                    self.key_stack.pop()
                    board.pop()
                if score >= beta:
                    return beta if score > MATE_IN_MAX else score

        # Staged generation: try the transposition move before building the list.
        best_score = -INFINITE
        best_move = None
        moves_searched = 0
        quiets_searched = 0
        original_alpha = alpha
        quiet_tried = []

        if tt_move is not None and board.is_legal(tt_move):
            board.push(tt_move)
            self.key_stack.append(board._transposition_key())
            try:
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, tt_move)
            finally:
                self.key_stack.pop()
                board.pop()
            moves_searched = 1
            best_score = score
            best_move = tt_move
            # The transposition move is the first move searched, so it is the
            # best so far by definition. Without this the root only ever
            # reported a move that BEAT the transposition move, and when the
            # transposition move stayed best - the usual case from depth two
            # onwards - the engine returned the move from the previous
            # iteration instead of the one it had just searched deeper.
            if ply == 0:
                self.root_best = tt_move
                self.root_score = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    self._on_cutoff(board, tt_move, depth, ply, previous, quiet_tried)
                    self._store(key, depth, 1, score, tt_move, ply)
                    return score
        else:
            tt_move = None

        moves = list(board.generate_legal_moves())
        if not moves:
            if moves_searched:
                pass  # the transposition move was legal, so this cannot happen
            return -MATE + ply if in_check else self.draw_score(board)

        lmp_limit = LMP_TABLE[depth] if depth < len(LMP_TABLE) else 999
        futile = (
            not in_check
            and not is_pv
            and depth <= 3
            and static is not None
            and static + FUTILITY_MARGIN * depth <= alpha
        )

        ep_square = board.ep_square
        pawns = board.pawns
        for order, move in self.order_moves(board, moves, tt_move, ply, previous):
            to_square = move.to_square
            is_quiet = (
                move.promotion is None
                and board.piece_type_at(to_square) is None
                and not (to_square == ep_square and (pawns & BB_SQUARES[move.from_square]))
            )

            if is_quiet and best_score > -MATE_IN_MAX:
                # Late move pruning.
                if not is_pv and not in_check and quiets_searched >= lmp_limit:
                    continue
                # Frontier futility.
                if futile and quiets_searched:
                    continue

            board.push(move)
            gives_check = board.is_check()
            self.key_stack.append(board._transposition_key())

            try:
                reduction = 0
                if (
                    is_quiet
                    and depth >= LMR_MIN_DEPTH
                    and moves_searched >= LMR_MIN_MOVES
                    and not in_check
                    and not gives_check
                ):
                    reduction = 1 + (moves_searched > 6) + (depth > 6)
                    if is_pv:
                        reduction -= 1
                    if order >= 380000:  # killer or counter move
                        reduction -= 1
                    if reduction < 0:
                        reduction = 0
                    if reduction > depth - 2:
                        reduction = depth - 2

                if moves_searched == 0:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, move)
                else:
                    score = -self.negamax(
                        board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, move
                    )
                    if score > alpha and reduction:
                        score = -self.negamax(
                            board, depth - 1, -alpha - 1, -alpha, ply + 1, move
                        )
                    if alpha < score < beta:
                        score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, move)
            finally:
                self.key_stack.pop()
                board.pop()

            moves_searched += 1
            if is_quiet:
                quiets_searched += 1
                quiet_tried.append(move)

            if score > best_score:
                best_score = score
                best_move = move
                if ply == 0:
                    self.root_best = move
                    self.root_score = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        self._on_cutoff(board, move, depth, ply, previous, quiet_tried)
                        break

        flag = 0
        if best_score <= original_alpha:
            flag = 2
        elif best_score >= beta:
            flag = 1
        self._store(key, depth, flag, best_score, best_move, ply)
        return best_score

    def _on_cutoff(self, board, move, depth, ply, previous, quiet_tried):
        if board.piece_type_at(move.to_square) is not None or move.promotion:
            return
        killers = self.killers[ply]
        if killers[0] != move:
            killers[1] = killers[0]
            killers[0] = move
        if previous is not None:
            self.counter[previous] = move
        history = self.history[board.turn]
        bonus = depth * depth
        index = (move.from_square << 6) | move.to_square
        history[index] += bonus
        if history[index] > 250000:
            for i in range(4096):
                history[i] >>= 1
        # Penalise the quiet moves that were tried and failed.
        for other in quiet_tried:
            if other != move:
                other_index = (other.from_square << 6) | other.to_square
                history[other_index] -= bonus
                if history[other_index] < -250000:
                    history[other_index] = -250000

    def _store(self, key, depth, flag, score, move, ply):
        if score > MATE_IN_MAX:
            score += ply
        elif score < -MATE_IN_MAX:
            score -= ply
        entry = self.tt.get(key)
        if entry is None or entry[0] <= depth:
            if len(self.tt) > 900000:
                self.tt.clear()
            self.tt[key] = (depth, flag, score, move)

    # -- iterative deepening ----------------------------------------------

    def search(self, board, history_keys, soft_limit, hard_limit):
        start = time.monotonic()
        self.deadline = start + hard_limit
        self.nodes = 0
        self.seldepth = 0
        self.root_colour = board.turn
        self.key_stack = list(history_keys)
        self.counter.clear()
        for killers in self.killers:
            killers[0] = killers[1] = None

        # Contempt: avoid draws when ahead on material, accept them when behind.
        # This also steers the 300-ply material adjudication the right way.
        balance = self._material_balance(board)
        self.contempt = 25 if balance > 120 else (-20 if balance < -120 else 8)

        legal = list(board.generate_legal_moves())
        if not legal:
            return None, 0, 0
        self.root_best = legal[0]
        self.root_score = 0
        best_move = legal[0]
        best_score = 0
        completed = 0

        score = 0
        iteration_start = start
        # Whatever happens inside - a clock abort unwinding through the tree, or
        # any unforeseen error - the board must come back exactly as it was
        # handed in. An unbalanced push flips the side to move and the next move
        # returned would be the opponent's, which scores as illegal.
        base_ply = len(board.move_stack)
        try:
            for depth in range(1, MAX_PLY):
                delta = 24
                if depth <= 4:
                    alpha, beta = -INFINITE, INFINITE
                else:
                    alpha, beta = score - delta, score + delta
                try:
                    while True:
                        score = self.negamax(board, depth, alpha, beta, 0)
                        if score <= alpha:
                            alpha = max(-INFINITE, alpha - delta)
                            delta *= 2
                        elif score >= beta:
                            beta = min(INFINITE, beta + delta)
                            delta *= 2
                        else:
                            break
                except TimeUp:
                    # Keep a partial improvement from this iteration if there was one.
                    if self.root_best is not None and self.root_score > best_score and completed:
                        best_move = self.root_best
                    break

                best_move = self.root_best
                best_score = score
                completed = depth

                # Decide whether the next iteration fits rather than using a fixed
                # fraction of the target. A fixed fraction is wrong in both
                # directions: it abandons the search early when iterations are cheap
                # and overshoots badly when they are not.
                now = time.monotonic()
                elapsed = now - start
                predicted = (now - iteration_start) * 2.4
                iteration_start = now
                if depth >= 4 and elapsed + predicted > soft_limit * 1.15:
                    break
                if abs(score) > MATE_IN_MAX and depth >= 4:
                    break
        finally:
            while len(board.move_stack) > base_ply:
                board.pop()

        return best_move, best_score, completed

    @staticmethod
    def _material_balance(board):
        total = 0
        for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            value = SEE_VALUE[piece_type]
            bitboard = board.pieces_mask(piece_type, chess.WHITE)
            total += value * bitboard.bit_count()
            bitboard = board.pieces_mask(piece_type, chess.BLACK)
            total -= value * bitboard.bit_count()
        return total if board.turn == chess.WHITE else -total


# --------------------------------------------------------------------------
# Game state tracking and the clock
# --------------------------------------------------------------------------

BASE_MS = 120000
INCREMENT_MS = 500
OVERHEAD_MS = 60
PANIC_MS = 3000


class Agent:
    def __init__(self):
        self.engine = Engine()
        self.board = None
        self.history_keys = []

    def sync(self, fen):
        """
        Rebuild the game history from the incoming FEN.  We are only handed a
        position, but the referee claims threefold and fifty-move draws, so the
        move stack has to be reconstructed to see repetitions coming.
        """
        target = chess.Board(fen)
        if self.board is not None:
            if self.board.board_fen() == target.board_fen() \
                    and self.board.turn == target.turn \
                    and self.board.castling_rights == target.castling_rights \
                    and self.board.ep_square == target.ep_square:
                return
            # The opponent has played exactly one move since our last position.
            for move in self.board.generate_legal_moves():
                self.board.push(move)
                if self.board.board_fen() == target.board_fen() \
                        and self.board.turn == target.turn:
                    self.board.halfmove_clock = target.halfmove_clock
                    self.history_keys.append(self.board._transposition_key())
                    return
                self.board.pop()
        # First move of the game, or the chain broke: start clean from the FEN.
        self.board = target
        self.history_keys = [target._transposition_key()]

    def budget(self, time_left_ms, board):
        """Soft and hard limits in seconds."""
        usable = time_left_ms - OVERHEAD_MS
        if usable <= 0:
            return 0.0, 0.0
        if time_left_ms < PANIC_MS:
            slice_ms = usable / 12.0
            return slice_ms / 1000.0, min(usable * 0.5, slice_ms * 1.5) / 1000.0

        # Expect more moves left early, fewer later. The increment is banked in
        # full: every move we make earns it back, so it is spendable each move.
        fullmove = board.fullmove_number
        expected = 32 if fullmove < 16 else (26 if fullmove < 32 else 20)
        soft_ms = usable / expected + INCREMENT_MS * 0.9
        # Never commit a large slice of the clock to one move. The referee flags
        # the instant the clock goes negative, so the watchdog grace is not slack
        # we can spend. Aborting mid-iteration is cheap because the move from the
        # last completed depth is kept.
        hard_ms = min(usable * 0.25, soft_ms * 2.2)
        if soft_ms > hard_ms:
            soft_ms = hard_ms
        return soft_ms / 1000.0, hard_ms / 1000.0

    def get_move(self, fen, time_left_ms):
        self.sync(fen)
        board = self.board
        soft_limit, hard_limit = self.budget(time_left_ms, board)

        if hard_limit <= 0.0:
            return next(iter(board.legal_moves))

        move, _, _ = self.engine.search(board, self.history_keys, soft_limit, hard_limit)
        if move is None or not board.is_legal(move):
            move = next(iter(board.legal_moves))
        board.push(move)
        self.history_keys.append(board._transposition_key())
        return move


_AGENT = Agent()


def get_move(fen: str, time_left_ms: int) -> str:
    """
    Entry point.  Wrapped so that no exception can reach the runner: the runner
    does not catch anything, and an uncaught exception is a lost game.
    """
    try:
        move = _AGENT.get_move(fen, time_left_ms)
        return move.uci()
    except Exception:
        try:
            # Rebuild from scratch and play something legal.
            board = chess.Board(fen)
            best = None
            for candidate in board.legal_moves:
                if best is None:
                    best = candidate
                if board.is_capture(candidate):
                    best = candidate
                    break
            _AGENT.board = None
            return best.uci() if best is not None else "0000"
        except Exception:
            return "0000"


def _warmup():
    """
    Spend a little of the 60s init budget priming code paths and caches so the
    first real move is not the one paying for them.
    """
    try:
        engine = Engine()
        board = chess.Board()
        keys = [board._transposition_key()]
        engine.search(board, keys, 0.20, 0.30)
    except Exception:
        pass


_warmup()
