# v6 — the merge: bot4's search on botB's bitboard kernel

All figures measured on the assistant sandbox: 1 core, 2.1 GHz Xeon, 4 GB,
python-chess 1.11.2, numba 0.67.0. That box is the fidelity proxy for the
competition container, not the PC. Sample sizes are stated next to every number
and the head-to-head is **not** a result.

---

## 1. Ledger

| Build | Movegen | Search | nps | Mean depth @2 s | Middlegame EBF | Gate |
|---|---|---|---|---|---|---|
| bot1_baseline | python-chess | basic | 27.0k | ~5 | — | pass |
| bot4_ordering | python-chess | full ordering + pruning | 17.1k | 6.4 | 4.04 | pass |
| bot5_nnue2 | python-chess | bot4 + NNUE eval | — | 6.8 | — | pass |
| botA_serena | — | teammate, untested | — | — | — | — |
| botB_adya_bitboard | **numba bitboard** | basic | **164.4k** | 5.5 | **19.48** | pass |
| **bot6_merged** | **numba bitboard** | **bot4's** | ~52k | **9.6** | **3.48** | **pass** |

**Standing conclusion: bot6 is the strongest build measured, by a wide margin
on every search metric, and it is NOT yet the ship recommendation.** It has 7
games of match evidence. bot4 stays shipped until §5 item 3 completes.

---

## 2. What bot6 is

Sections 1–3 of `bot6_merged.py` (bitboard kernel, board contract, evaluation)
are **byte-identical to botB_adya_bitboard.py**, lines 1–1594, with two
exceptions recorded in §4 bug #2. Nothing else in them is edited. That is
deliberate: perft proves those sections, and editing them would make a bench
result unreadable.

Everything below is bot4's search, ported from `chess.Move` / `chess.Board` to
int-encoded moves and the bitboard board contract.

### Ported unchanged in spirit

Two-tier transposition table with swap ageing; killers, butterfly history with
gravity, countermoves, one ply of continuation history; PVS with progressive
aspiration widening; null move, reverse futility, late move pruning, futility,
SEE pruning, internal iterative reduction, log-table LMR; quiescence that
generates evasions in check; mate distance pruning; contempt-weighted draw
scoring; bot4's time manager.

### Changed on purpose, with reasons

1. **Staged move generation dropped.** In bot4 the TT move was tried before
   `list(board.legal_moves)` was ever called, because generation cost 33 µs
   against 8 µs for a legality test — the single largest saving in that engine.
   Here generation costs **3.09 µs**. The saving is inside the noise and the
   extra state is a bug surface for nothing. The TT move is swapped to the front
   of a normally generated list, which cannot desynchronise.
2. **MVV-LVA is now free.** The generator packs victim at bit 18 and mover at
   bit 21. bot4 called `piece_type_at()` twice per move to recover them. History
   indexing is free too: `move & 0xFFF` is a unique source|target key.
3. **SEE is jitted.** bot4 ran it in Python at ~10 µs and could only afford it
   behind a victim-outranks-attacker filter. The swap-off loop with x-ray
   recompute is a handful of mask operations now, so the filter is gone and SEE
   runs on every capture.
4. **botB's time manager replaced.** See §4 bug #3.
5. **`STABLE_BREAK` defaults off.** The log already flags bot4's stability
   early-break as untuned and suspect — bot4 triggered it with most of its
   budget unspent, and bot6 is faster still. Off until measured. Constant is in
   place for the A/B.

---

## 3. Environment facts (new, do not re-derive)

Per-node cost of the bitboard kernel called from Python:

| Operation | Cost | python-chess equivalent |
|---|---|---|
| Generate legal moves | **3.09 µs** | 26.77 µs — **9× faster** |
| Make + unmake | **2.17 µs** | ~6 µs (push/pop) |
| Evaluate (jitted PST, tapered) | **0.67 µs** | 12.5 µs (bot1/bot4 Python eval) |
| **Floor** | **5.94 µs = 168k nps** | |

botB measured 164k nps against that 168k floor, i.e. **its search added
essentially nothing per node** — which is exactly why its EBF was 19.48.

Perft, all six standard positions, **ALL PASS**, at **5.3–13.5 M nps** on one
core. Verified again after the §4 bug #2 fix.

Cold start: import **15.5 s**, first move **2.47 s** (that is its budget, not a
JIT leak — the warmup covers every kernel). Inside the 60 s init budget with
room, and the shipped-numba-cache trick should take the import to ~1–2 s.

---

## 4. Regression checklist — bugs found this iteration

**#1 — Missing gives-check guard on quiet-move pruning. Mine, in the port.**
bot4 never prunes or reduces a move that gives check. The port dropped that
rule. `d6d1`, the only move that forces the selftest's mate in 3, is a *quiet*
move, so late move pruning threw it away and the search reported **−229 at
depth 13 in a position that is mate in three**. Bisecting showed both the
shallow-prune group and LMR independently masked it.
Fixed by testing `in_check()` after `make` (2.17 µs) rather than writing a
separate gives-check kernel that could go wrong in its own way. Guard now also
blocks LMR.
*Lesson, already in the log and now confirmed a third time: a search that
reports a large depth is not a search that is correct. Only positions with a
known answer measure correctness.*

**#2 — Heap corruption from `_undo` overflow. Latent in botB, not in the port.**
`st[4]` is a monotonic make/unmake stack pointer that indexes `_undo`, which had
`MAX_PLY + 8 = 136` rows and **no bounds check** — numba `@njit` writes past the
end silently. The counter counts *every move since the board was constructed*,
and `_tracked` holds the whole game, so the index is **game plies + search ply**,
not search ply alone.

botB never overflowed it because depth 4–5 kept the total under 136. bot6 at
depth 14 starts writing out of bounds around move 50. It surfaces as
`free(): invalid next size (normal)` — glibc kills the process. **`get_move`'s
try/except cannot catch it, because it is not a Python exception.** Three
crashes in six games, and without the stderr tail it reads as "the merge is
unstable" rather than "the kernel has an unguarded array".

**Confirmed at the boundary after the fix:** a 905-ply stress run over 7 games,
with a deep search forced every 20th ply, saw a worst `st[4]` of **158** between
moves — already past the old 136 limit, so the old build was certainly
overflowing and not merely unlucky. Zero failures; 15% of the new headroom used.

Fixed with `UNDO_SLOTS = 1024` (the referee adjudicates at 300 plies, so that is
the whole game plus the deepest search twice over, for 40 KB) plus a per-search
ply cap computed once from the remaining slots. Perft re-verified all-pass;
7-game rerun had zero failures.

*Lesson, new and important: **this bug class only exists in long games.** Every
short-control battery in the repo would miss it. Any future zero-failure run
must include a high `--ply-cap`.*

**#3 — botB's time manager spends its hard limit on every move.** At a 120 s
clock, soft is 4393 ms and hard is 10983 ms, and botB spent **exactly 10.98 s,
four runs out of four, to 10 ms precision**. Soft was decorative. It never
flags — the budget decays geometrically — but it front-loads absurdly: 11 s on
move one of a curated opening, down to 1.7 s by move 50. Replaced with bot4's.
bot6's worst move over a 120-ply self-game was 3534 ms against a 5661 ms hard
limit (62% of hard, where botB sat at 100% every move), and both clocks finished
with ~28 s left.

---

## 5. Metrics against the agreed authority order

| # | Metric | Target | bot6 status |
|---|---|---|---|
| 1 | Elo vs previous build, SPRT `elo0=0, elo1=5` | pass before merge | **13 games, +10 =1 −2, 80.8%, +249 Elo (95% CI +76 to +inf), LLR +0.18 of ±2.94. UNDECIDED — not a result** |
| 2 | Failures per 100 games | **exactly zero** | **PASS.** perft 6/6; 31 selftest checks; clock discipline 120 plies; long-game stress 905 plies / 41 deep searches; 13 match games. **0 failures throughout.** Still short of 100+ match games |
| 3 | nps on the fixed 12-position suite | track | ~52k (bot4 17.1k, botB 164.4k) — read with metric 4, alone it is misleading in both directions |
| 4 | First-move cutoff rate | >85% | **PASS — 90.1%** over 189,310 cutoffs (bot4: 90.6% over 10,201). Quiescence is 37.4% of nodes |
| 5 | Mean depth at real 120 s + 0.5 s | sanity check | **STILL A GAP.** Longest test was 5 s + 200 ms |

Endgame conversion: K+R **17** plies (bot4 19), K+Q 19 (13), K+2B **19**
(botB 55–67), K+R+B **29–39** (bot4 19) — the last is a regression to chase.

---

## 6. Next steps, prioritised

**P0 — evidence, not code.**

1. **Zero-failure gate at scale, with `--ply-cap 300`.** Hundreds of games
   against the weak baselines. Bug #2 lived only in long games and six games
   found it by luck; the gate has to be able to find the next one on purpose.
2. **Full SPRT vs bot4 on the PC**, ~300 games at `elo1=5`, `--workers 6`.
   Seven games is not a result and bot6 must not ship on one.
3. **Metric 5 at the real 120 s + 0.5 s control**, 20–30 games. Open across four
   iterations now. bot6 changes per-move cost by 3×, so a budget tuned at 5 s
   is tuned wrong.

**P1 — probably free Elo, cheap to test.**

4. **Re-tune the pruning margins.** They are bot4's, tuned for an engine seeing
   depth 6. `RFP_MAX_DEPTH = 7` and `LMP_MAX_DEPTH = 6` covered most of bot4's
   tree and now cover a small slice of bot6's. Sweep them; this is the most
   likely single source of remaining Elo.
5. **A/B `STABLE_BREAK`** on and off, now that the constant exists.
6. **Ship the prebuilt numba cache.** Import 15.5 s → ~1–2 s. Build with
   `NUMBA_CPU_NAME=generic` and `NUMBA_CPU_FEATURES=""`, confirm the cache hits
   on a different CPU (build on the Ryzen, load in the sandbox — that is the
   cross-CPU test, and it has not been run).

**P2 — known weaknesses.**

8. K+R+B conversion regression, 29–39 plies against bot4's 19.
9. bot5's NNUE eval on bot6's search. The kernel is 0.67 µs against bot5's
   3.2 µs, so there is budget, but this is a *second* change and must not be
   bundled with the merge.

---

## 7. Principles this iteration confirmed

- **Nodes to depth, not nodes per second.** Third confirmation. bot6 is **3×
  slower in nps than botB** and searches **4+ plies deeper**. A speed metric read
  alone scores the best build as a regression and the worst build as the winner.
- **A fast kernel inside a weak search is worth less than a slow kernel inside a
  good one.** botB at 164k nps and EBF 19.48 reached depth 5.5; bot4 at 17k nps
  and EBF 4.04 reached 6.4. The search is where the Elo is.
- **Depth is not correctness.** Bug #1 reported depth 13 and a −229 score inside
  a forced mate.
- **A crash is not always an exception.** Bug #2 killed the process below the
  Python level, where `get_move`'s try/except cannot reach. When the harness says
  "crash", read the stderr tail before theorising.
- **Take the proven half verbatim.** Keeping botB's kernel byte-identical meant
  perft stayed a valid oracle throughout, and made it possible to say with
  confidence that bug #1 was in the port and bug #2 was not.
