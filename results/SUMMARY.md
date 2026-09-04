# Results — 2026-09-04 09:45 UTC

- machine: yinmint (12 cores, x86_64, Linux)
- commit: `6a3e788` on `main` (uncommitted changes present)

## Failure gate

| run | games | failures | detail |
|---|---|---|---|
| bot4_ordering_vs_bot1_baseline | 198 | 0 | clean |
| bot5_nnue2_w128_vs_bot4_ordering | 59 | 0 | clean |
| bot5_nnue2_w256_vs_bot4_ordering | 300 | 0 | clean |
| nobreak_vs_bot4_ordering | 300 | 0 | clean |

**Gate clean: zero failures across all runs.**

## Match results

| run | control | +W =D -L | score | Elo (95%) | LLR | verdict |
|---|---|---|---|---|---|---|
| bot4_ordering_vs_bot1_baseline | 8000ms+500ms | +118 =36 -44 | 68.7% | +136 (+92..+186) | +1.53 | undecided |
| bot5_nnue2_w128_vs_bot4_ordering | 8000ms+500ms | +47 =7 -5 | 85.6% | +310 (+218..+460) | +1.50 | undecided |
| bot5_nnue2_w256_vs_bot4_ordering | 8000ms+500ms | +165 =30 -105 | 60.0% | +70 (+33..+109) | +0.96 | undecided |
| nobreak_vs_bot4_ordering | 8000ms+500ms | +149 =64 -87 | 60.3% | +73 (+38..+109) | +1.15 | undecided |

Elo intervals are wide at small samples by nature. A verdict of *undecided* means exactly that — not 'probably fine'.

## Selftest

### bot5_nnue2.w256.selftest.txt

```
tactics
  ok   tactics/mate in 1, back rank: a1a8
  ok   tactics/mate in 1, scholar's: f3f7
  ok   tactics/mate in 2, smothered: g5f7
  ok   tactics/takes a free queen on the file: d1d5
  ok   tactics/takes a free queen next door: b1b2
  ok   tactics/mate in 3, white: f6a6
  ok   tactics/mate in 3, black: d6d1
  ok   tactics/promotes: a7a8q
  ok   tactics/takes the draw when lost: a1b2
edge cases
  ok   edge/start position: d2d4
  ok   edge/only one legal move: h1g1
  ok   edge/en passant available: d2d4
  ok   edge/black to move, promotion race: g2g1q
  ok   edge/bare kings plus pawn: e3e4
  ok   edge/fifty-move clock nearly up: e3e4
  ok   edge/castling rights all round: b2b3
  ok   edge/very low clock: c2c4
  ok   edge/absurdly low clock: g1h3
  ok   edge/zero clock: g1h3
  ok   edge/negative clock: g1h3
malformed input
  ok   malformed/'': 0000
  ok   malformed/'not a fen': 0000
  ok   malformed/'8/8/8/8/8/8/8/8 w - - 0 ': 0000
  ok   malformed/'rnbqkbnr/pppppppp/8/8/8/': 0000
endgame conversion
  ok   conversion/K+R vs K: mate in 19 plies
  ok   conversion/K+Q vs K: mate in 13 plies
  ok   conversion/K+2B vs K: mate in 23 plies
  ok   conversion/K+R+B vs K: mate in 21 plies
repetition tracking
  ok   repetition: reconstructed 12 positions from FENs alone
  ok   repetition: winning line played ['b1b8']
clock discipline
  ok   clock: 120 plies, worst ply 17, 3439ms against a 5881ms limit
       white 63.8s left, black 53.6s left

1 failure(s):
  FAIL tactics/hangs nothing, must recapture: played c7c6, wanted one of ['d8d5', 'f6d5']
```

