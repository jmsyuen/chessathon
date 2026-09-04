# Results — 2026-09-04 08:05 UTC

- machine: yinmint (12 cores, x86_64, Linux)
- commit: `af37e11` on `main` (uncommitted changes present)

## Failure gate

| run | games | failures | detail |
|---|---|---|---|
| bot4_ordering_vs_bot1_baseline | 198 | 0 | clean |
| nobreak_vs_bot4_ordering | 300 | 0 | clean |

**Gate clean: zero failures across all runs.**

## Match results

| run | control | +W =D -L | score | Elo (95%) | LLR | verdict |
|---|---|---|---|---|---|---|
| bot4_ordering_vs_bot1_baseline | 8000ms+500ms | +118 =36 -44 | 68.7% | +136 (+92..+186) | +1.53 | undecided |
| nobreak_vs_bot4_ordering | 8000ms+500ms | +149 =64 -87 | 60.3% | +73 (+38..+109) | +1.15 | undecided |

Elo intervals are wide at small samples by nature. A verdict of *undecided* means exactly that — not 'probably fine'.

## Selftest

### bot5_nnue2.w256.selftest.txt

```
tactics
  ok   tactics/mate in 1, back rank: a1a8
  ok   tactics/mate in 1, scholar's: f3f7
  ok   tactics/mate in 2, smothered: g5f7
  ok   tactics/hangs nothing, must recapture: d8d5
  ok   tactics/takes a free queen on the file: d1d5
  ok   tactics/takes a free queen next door: b1b2
  ok   tactics/mate in 3, white: f6a6
  ok   tactics/mate in 3, black: d6d1
  ok   tactics/promotes: a7a8q
  ok   tactics/takes the draw when lost: a1b2
edge cases
  ok   edge/start position: c2c4
  ok   edge/only one legal move: h1g1
  ok   edge/en passant available: d2d4
  ok   edge/black to move, promotion race: g2g1q
  ok   edge/bare kings plus pawn: e3f4
  ok   edge/fifty-move clock nearly up: e3d4
  ok   edge/castling rights all round: d2d4
  ok   edge/very low clock: g1f3
  ok   edge/absurdly low clock: g1h3
  ok   edge/zero clock: g1h3
  ok   edge/negative clock: g1h3
malformed input
  ok   malformed/'': 0000
  ok   malformed/'not a fen': 0000
  ok   malformed/'8/8/8/8/8/8/8/8 w - - 0 ': 0000
  ok   malformed/'rnbqkbnr/pppppppp/8/8/8/': 0000
endgame conversion
  ok   conversion/K+R vs K: mate in 15 plies
  ok   conversion/K+Q vs K: mate in 65 plies
  ok   conversion/K+2B vs K: mate in 27 plies
repetition tracking
  ok   repetition: reconstructed 12 positions from FENs alone
  ok   repetition: winning line played ['b1b8']
clock discipline
  ok   clock: 110 plies, worst ply 14, 4691ms against a 5960ms limit
       white 59.8s left, black 51.1s left

1 failure(s):
  FAIL conversion/K+R+B vs K: no mate after 70 plies
```

