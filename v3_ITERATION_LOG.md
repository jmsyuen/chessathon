# Iteration log

Living document. Update it at the end of every iteration, before starting the
next one. It exists so that no future iteration re-derives something we already
paid for, and so that no bug we have already found ships twice.

**Update protocol.** Add a row to the ledger (§1). Tick or extend the regression
checklist (§4). Move anything from "next steps" (§8) into the ledger with its
measured result, including the ones that failed — a change that measured neutral
is information and must not be silently retried. Correct any fact in §3 that
turned out wrong rather than leaving both versions in.

---

## 0. How this project persists — read first

Nothing about the working environment survives a session. Concretely:

| Where | Survives? | Use it for |
|---|---|---|
| **Project files** | **Yes.** Read-only at `/mnt/project/`, always in context | **The source of truth.** Every file the next iteration needs to read or edit must be here |
| **Past chats in this project** | Yes, searchable | Rationale, measurements, why a decision was made. **Not** for recovering source code |
| Assistant sandbox (`/home/claude/...`) | **No — destroyed at session end** | Scratch only: data sets, engine binaries, checkpoints |

**This has already cost us.** §5 of an earlier version of this log listed
`h2h.py`, `run_rung.py`, `autopsy.py`, `gen_book.py`, `test_edges.py` and
`test_conversion.py` as existing tooling. **None of them exist anywhere.** They
were written in a sandbox that has since been wiped and were never uploaded. A
later iteration hit the exact problems `h2h.py` and `run_rung.py` were built to
solve — chunking long runs past the ~5 minute call limit — and re-derived worse
versions from scratch.

Past chats are a poor substitute for uploading. A file created once and then
patched five times in-session cannot be reliably reconstructed from a transcript.

**Rule: if a tool is worth a row in §5, upload it the same day.**

Scratch that is expected to die and should never be uploaded: the training csv
(~59 MB) and its parsed cache (~130 MB), float checkpoints, and above all the
Stockfish binary — shipping or calling an engine from the zip is retroactive
disqualification.

---

## 1. Iteration ledger

| # | Date | Build | Change | Measured result | Verdict |
|---|---|---|---|---|---|
| 0 | 3 Sep | bot1 | Baseline, speed-to-ship. PVS-ish negamax, TT, quiescence, null move, LMR, aspiration, tapered eval, mating drive, contempt | 100% vs SF d4; 50% vs SF d6 — **see caveat below** | On the ladder |
| 1 | 3 Sep | bot2 | From-scratch rebuild: staged movegen, bitboard eval, SEE, lazy eval, pawn hash | 60% vs bot1 over 15 games (Elo +70 ±184); 0 losses in 52 games | **Not separated. Not shipped.** |
| 2 | 3 Sep | bot2_nnue | Learned eval: material + (768→256)×2 perspectives, SCReLU, int16, numba kernel. bot1's search, ordering, TT, `_budget`, `_sync` and `get_move` wrapper carried over unchanged | 16.7% vs bot1, 12 games at 3s+100ms (+1 =2 −9, Elo −280, 95% −inf to −113). vs Stockfish at 8s+250ms: 37.5% at depth 3 (n=4), 25.0% at depth 4 (n=6), 0% at depth 5 (n=2). Selftest gate FAILS one tactic | **Not shipped. Root cause found — see §4 bug #8** |

**Caveat on row 0.** "100% vs SF d4" is not reproducible and is probably wrong or
was measured against a differently-configured opponent. bot2_nnue scored 25% vs
SF depth 4 while being 20–40% faster than bot1 at equal depth, and a separate
measurement has bot1 losing 0–6 to `nodes:1000` (SF depth 4 uses well over 1000
nodes). Treat row 0's Stockfish figures as unverified until bot1 is re-run on the
depth ladder. The paired run is one cheap call and should be done before anyone
quotes an altitude number.

**Standing conclusion:** bot1 remains the strongest build. bot2 and bot2_nnue are
both unshipped. bot2_nnue is a working NNUE pipeline with a correctly diagnosed
data fault, not a candidate.

---

## 2. Metrics we agreed on, and what has actually been measured

Authority order matters. Items 1 and 2 gate everything else.

| # | Metric | Target | Status |
|---|---|---|---|
| 1 | Self-play Elo vs previous version, SPRT `elo0=0, elo1=5, α=β=0.05` | pass before merge | **NOT DONE** — bot1 vs bot2 only 15 games; bot2_nnue vs bot1 only 12 games (but −280 Elo needs no SPRT to reject) |
| 2 | Failure rate per 100 games (flag/crash/illegal/drawn-while-winning) | exactly zero | **PASS** for all three builds — 0 flags, 0 crashes, 0 illegal moves |
| 3 | Nodes per second on fixed benchmark suite | track per build | bot1 32–52k, bot2 14–18k, **bot2_nnue 42–57k** |
| 4 | First-move cutoff rate | >85% | **NEVER MEASURED** — still a gap |
| 5 | Average depth at real 120s+0.5s control | sanity check | **PARTIAL** — bot2_nnue matches bot1 depth-for-depth at a 20s budget |

**New metric, and it is now item 0.** See §4 bug #8: *evaluation correlation
against engine labels, restricted to near-level positions*. Nothing about a
learned evaluation may be believed until this is measured, because every other
metric here passed while the net was blind.

---

## 3. Environment facts — do not re-derive

**Profiling (python-chess 1.11.2, one core).** These drive architecture:

| Operation | Cost | Consequence |
|---|---|---|
| `list(board.legal_moves)` | 33–36 us | Dominates. Avoiding generation beats optimising anything else |
| `board.push()` + `pop()` | 6 us | Cheap. Push-then-test is cheaper than `gives_check()` |
| `board._transposition_key()` | 1 us | Cheap. **Hand-rolled incremental Zobrist is not worth the bug risk** |
| bot1 `evaluate()`, pure Python | 12.5 us | ~25% of node cost |
| **NNUE forward, numba, hidden 256, 2 perspectives** | **3.2 us** | ~8% of node cost |
| Packing 8 bitboards for the kernel | 0.55 us | Included in the 3.2 us above |

**A jitted learned evaluation is cheaper than a Python piece-square evaluation
on this platform.** This is the single most useful fact from iteration 2 and it
inverts the usual assumption. bot2_nnue searched 20–40% faster than bot1 and
reached identical depths. Any future eval work should assume the numba path is
free and spend the budget on quality.

**Quantisation.**
- Scale is not a free parameter. QB=64 (the common bullet default) costs
  **35.6 cp mean / 257 cp max** error against the float model. QA=512/QB=1024
  costs 13.2/69. **QA=1024/QB=2048 costs 0.8/4.**
- int64 accumulation has ample headroom at those scales: worst case ~2.2e12
  against a 9.2e18 limit. Checked, not assumed.
- **Requantising needs no retraining.** It is a save-time transform on the float
  checkpoint. Three re-saves took seconds.
- Always measure quantised-vs-float before believing a net. A scale error does
  not crash; it just makes the search thrash.

**Toolchain.**
- `pip install chess` gives 1.11.2, exactly the competition version. `numba`
  gives 0.67.0, also exact.
- Stockfish 17.1 avx2 is obtainable directly from the GitHub release asset
  (`release-assets.githubusercontent.com` is allowlisted; the GitHub API is rate
  limited, so build the URL by hand):
  `.../official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-avx2.tar`
- Labelling throughput on one core, including game play-out and filtering:
  **404 pos/s at `nodes:1200`**, 221 pos/s at `nodes:2500`, 173 pos/s at
  `nodes:5000`. Use raw UCI pipes; `chess.engine.SimpleEngine` has asyncio
  overhead that matters at this rate.
- **torch is not needed to train an NNUE this size and its download is not worth
  the disk.** numpy plus a jitted sparse gather/scatter is enough.
- Training cost: 926k positions, hidden 256, batch 4096 — **~6 s per epoch**,
  40 epochs in 4 minutes on one core. **Data generation costs ~40x what training
  costs.** Budget accordingly: buy data, not epochs.
- Sandbox has **1 core**, same as the competition container.
- **Background processes are reaped between tool calls — `setsid` and `nohup` do
  not help.** Two runs were lost to this. Additionally **bash calls are killed at
  ~5 minutes**. Anything long must be foreground, chunked, and checkpointed to
  disk every iteration. `tools/train.py --state` does this for float weights,
  Adam moments and the epoch counter. `tools/gendata.py` is line-buffered so a
  killed run keeps every position it wrote.

**Agent contract (re-verified 3 Sep, docs change — re-check before the lock).**
- Eligibility: at least one UK university student per team. 50 London seats.
- Books and tablebases are permitted as shipped data; `chess.polyglot` and
  `chess.syzygy` are in the base image.
- Pondering is explicitly allowed: the process keeps its core after `get_move`
  returns. **No build uses this.** See §8.
- Import + warm-up: bot1 0.41s. **bot2_nnue 14 s**, almost all of it importing
  numba and llvmlite rather than compiling. Against a 60 s budget, so fine, but
  it is no longer negligible.
- numba `cache=False` is mandatory. The platform filesystem is read-only apart
  from `/tmp`, so a cache write would fail next to the source.
- Weights ship as a second artifact: `agent.py` at the zip root plus
  `weights/nnue.npz`. `harness/package.py` already includes `weights/` via
  `DEFAULT_INCLUDES`. Resolve the path from `Path(__file__).resolve().parent`,
  never from the working directory.

**Harness gotchas (confirmed in source).**
- `board.outcome(claim_draw=True)` — threefold and fifty-move are claimed
  automatically *against* us. History must be reconstructed from FENs.
- 300-ply adjudication is **pure material**, no positional terms. Exploitable
  both ways; no build plays for it deliberately.
- The 500ms watchdog grace is not usable slack — the referee flags the instant
  the clock goes negative.
- `runner.py` does not wrap `get_move`. Any uncaught exception is a lost game.
- The zip is first on `sys.path`. Never name a file `chess.py`, `types.py`.
- stdout cap 4096 bytes; over-cap counts as an illegal move.

---

## 4. Regression checklist — run against every new build

These are the bugs found so far. Most are pattern-level and will recur in any
rewrite. **Bot 1 was audited against 1–7 and failed one.**

| # | Bug | How it presents | Test |
|---|---|---|---|
| 1 | Root best move not set on the staged TT path | Engine returns its **depth-1 move** while searching deep. Scored 50% vs `greedy` | Score vs `greedy` must be ~100% |
| 2 | Unbalanced push/pop on clock abort | Returns a move for the **opponent**. Instant loss. Only fires under time pressure | Fuzz at 50ms budgets; assert legality |
| 3 | Quiescence stands pat in check | Can be mated at a leaf and return a material score | Mate-at-leaf positions |
| 4 | SEE counts illegal king recaptures | Winning captures score as losing; suppressed in ordering, pruned in qsearch | Brute-force swap-off comparison |
| 5 | Floor division colour bias | Position and its mirror differ by 1cp. Hit 45% of positions | `evaluate(b) == evaluate(b.mirror())` |
| 6 | En passant classified as quiet | Ordered by history, exposed to LMR and futility | Assert ep is treated as a capture |
| 7 | Mangled boolean (`not x == 0`) | Silently wrong branch | Read every conditional in scaling code |
| 8 | **Training distribution collapse** | Net scores r=0.95 against engine labels and beats the hand-written eval on mae, then loses 280 Elo. In level positions its correlation is **r ≈ 0**: it learned to count material and nothing else | Correlate eval against engine labels **restricted to \|material\| ≤ 120**, from near-level openings. `tools/evalcmp.py --from-openings --max-imbalance 120` |

**Bot 1 audit result:** fails #5 (198/399 positions). Fix is one line — truncate
toward zero instead of `//`. Bot 1 also has #2's missing `try/finally`, currently
harmless *only* because it searches on `board.copy(stack=False)`. That is
protection by accident, not by design. Anyone refactoring bot1 to search the
tracked board reintroduces a losing bug. **bot2_nnue inherits #2**, since it
carries bot1's search unchanged.

**bot2_nnue audit result:** **passes #5** — 0 of 400 random positions asymmetric
under `evaluate(b) == evaluate(b.mirror())`. This is structural, not luck: both
accumulators are computed relative to the side to move and the result is never
negated, so there is no white-relative-then-negate step for floor division to
bias. Any future eval that scores from White's perspective and flips the sign
reintroduces #5; a two-perspective net cannot. Do not re-audit this.

### Bug #8 in full

`tools/gendata.py` seeded games with 2–12 uniformly random plies and injected 10%
random moves during play-out. Result over 926,724 positions: median
\|material\| is **350 cp**, only **18.3%** are within 60 cp of level, only 31.5%
within 120 cp. In that distribution material counting explains nearly all the
variance, so gradient descent never had to learn a positional feature.

Measured on quiet positions from near-level openings, n=100:

| Eval | mae | **r** | sign |
|---|---|---|---|
| bot2_nnue, no skip connection | 146 | **−0.030** | 54.0% |
| bot2_nnue, material skip connection | 119 | **+0.188** | 51.0% |
| bot1 piece-square | 61 | **+0.336** | 49.0% |

Measured on the *original* lopsided distribution, n=200 — the flattering view
that hid it for most of the iteration:

| Eval | mae | r | slope |
|---|---|---|---|
| bot2_nnue | **222** | 0.944 | **1.11** |
| bot1 piece-square | 385 | **0.960** | 1.60 |

**Partial fix, implemented and measured.** Material is now a fixed skip
connection, computed inside the numba kernel — which already walks every piece,
so it is free — meaning the only thing the net can learn is the positional
residual. That moved r from −0.030 to +0.188 on identical data. **The
architecture is now right; the data is still wrong.**

**Cost of the fix.** It also broke one tactic (`hangs nothing, must recapture`
now plays `c8f5`), which passed before the change. All ten tactics passed with
the pre-skip net. That regression is unexplained and must be understood, not
patched around — see §8 P1.

**Lesson worth keeping:** keep a deliberately terrible calibration opponent
(`greedy`, one-ply material). A 50% score against it is unmistakable. The same
bug measured against a real opponent looks like "the eval needs tuning".

---

## 5. Test suite

**The status column is not decoration.** Anything marked *missing* was written in
a wiped sandbox and no longer exists; those rows are work to redo, not tools to
run. See §0.

| File | Status | Catches | Runtime |
|---|---|---|---|
| `tools/selftest.py` | in project | Tactics, clock edges (0ms→negative), promotions, ep, castling, single legal move, malformed FEN, endgame conversion, repetition tracking, clock discipline | ~6 min; split it — clock discipline alone exceeds a 5 min call |
| `tools/bench.py` | in project | Match play mirroring the referee, alternating colours, Elo with interval | ~20 s/game at 3s+100ms |
| `tools/ladder.py` | in project | Stockfish staircase, reports the 50% crossover | — |
| `baselines/stockfish/agent.py` | in project | Stockfish sparring partner speaking the agent contract | — |
| **`tools/evalcmp.py`** | **upload** | **Bug #8.** Both evals vs fresh engine labels, split by check / capture-best / king-exposed / material level | ~2 min at n=200 |
| `tools/gendata.py` | **upload** | (generator) Stockfish-labelled self-play. **Contains bug #8** — the file P0 edits first | 404 pos/s |
| `tools/train.py` | **upload** | (trainer) numpy + numba, jitted sparse gather/scatter, `--state` checkpointing, quantise-on-save | 6 s/epoch |
| `tools/checknet.py` | **upload** | Quantised kernel vs float model; material sanity | ~30 s |
| `tools/kernelbench.py` | optional | Per-position eval cost vs bot1 vs the movegen floor | ~30 s |
| `gen_book.py` + its 60-position book | **MISSING** | Near-level opening book, ±45cp at depth 13, deterministic shuffle so staircase rungs pair across engines. **Blocks P0** | ~5 min to rebuild |
| `h2h.py` | **MISSING** | Chunked head-to-head accumulating to disk — the fix for the ~5 min call limit | — |
| `run_rung.py` | **MISSING** | One checkpointed Stockfish ladder rung | — |
| `autopsy.py` | **MISSING** | Was a drawn or lost game ever winnable | ~4 min |
| `test_edges.py`, `test_conversion.py` | **MISSING** | Superseded by `tools/selftest.py` | — |

Feature encoding is verified identical between `tools/train.py` and the agent
kernel against a python-chess reference — do not skip that check on any rewrite.
A silent mismatch looks exactly like a badly trained net.

**Caveat that bites above n=24:** `bench.py` OPENINGS has 12 positions and both
engines are deterministic, so 24 distinct games is the ceiling. Rebuilding the
opening book lifts this too. Expand the set rather than adding `noise:`, which
changes a rung's strength as a side effect.

## 6. Measurement costs — read before promising a verdict

| Difference to detect | Games needed |
|---|---|
| ~200 Elo (adding quiescence) | ~40 |
| ~50 Elo (real eval improvement) | 200–400 |
| ~10 Elo (a tuning tweak) | 2,000–4,000 |
| 5% score difference at our draw rate | **~246** |

At 12s+0.5s on one core a game costs 40–95s. **One core gives roughly 40–80
games per hour.** Plan accordingly: a 15-game result is a smoke test, not a
verdict. A −280 Elo result over 12 games is an exception — the interval excludes
zero comfortably and needs no SPRT to reject.

**Constraint that will not go away:** a black-box `get_move(fen, time_left_ms)`
cannot be driven at fixed nodes. Cross-engine comparison must be fixed time. Use
fixed nodes only for A/B testing variants of an engine we control internally.

**Compressed control caveat.** 12s+0.5s keeps the increment truthful (both time
managers assume 500ms) and compresses only the base. It slightly favours lower
per-move overhead. Any final candidate must be re-checked at 120s+0.5s.

---

## 7. Known gaps and untested areas

- **The training data is not balanced.** The single blocking issue for bot2_nnue.
  See §4 bug #8 and §8 P0.
- **King safety is a shared weakness.** On quiet king-exposed positions *both*
  evals correlate **negatively** with Stockfish (bot2_nnue −0.575, bot1 −0.454).
  n=13, so this is a lead and not a finding — but if it holds at n≥100 it is a
  large gain available to whichever engine ships.
- **K+2B vs K is fixed in bot2_nnue** (31 plies), and remains broken in bot1.
  bot2_nnue also converts K+R in 17 plies vs bot1's 49 and K+Q in 17 vs 23. The
  learned eval plus `_mating_drive` converts materially better than PSTs. Worth
  porting the drive term's tuning back into bot1 regardless of what ships.
- **SEE ignores pins.** 2 sign errors in 1,611 captures, both conservative.
  Standard in every engine. Not worth fixing.
- **No SPRT has ever been run.**
- **First-move cutoff rate never instrumented.**
- **Real-control (120s+0.5s) play barely tested.**
- **Pondering unused by every build.**
- **No build plays deliberately for the 300-ply material adjudication.**
- Opening book remains a trap: rated games start from unpublished curated
  positions, so a book keyed on move one is out of book immediately.

---

## 8. Next steps, prioritised

**P0 — decides whether bot2_nnue is viable at all**
0. **Rebuild the near-level opening book and upload it the same day.** It is gone
   (§0, §5) and step 1 needs it for seeding. It also lifts `bench.py`'s
   24-distinct-game ceiling. Stockfish filter at depth 13, keep ±45cp.
1. **Regenerate the data balanced.** Seed from near-level positions, not random
   plies. Drop temperature to ~3%, or sample from engine MultiPV top-k rather
   than uniform random. Reject or stratify on \|material\| so the distribution
   over material balance is roughly flat. Target ≥1M positions with ≥50% inside
   ±120 cp. ~8–10 chunked generation runs; training after it is 4 minutes.
2. **Gate on `evalcmp --from-openings --max-imbalance 120`, not on val loss.**
   Val loss was 0.00526 for the blind net and 0.00537 for the one with positional
   signal — it moved the **wrong way** across the fix that demonstrably added the
   missing signal. It cannot see this class of failure. Require r > 0.40 on quiet
   level positions (bot1 is 0.336) before playing a single game.
3. Only then bot2_nnue vs bot1, and only then SPRT.

**P1 — cheap, high information**
4. **Understand the recapture regression.** One tactic broke when material became
   a skip connection. Diagnose it; do not patch around it. A build that fails the
   correctness gate is not a build.
5. **Fix bot1's colour bias** (bug #5). One line, zero risk, affects half of all
   positions. Applies whichever engine ships, including bot2_nnue.
6. **Re-run bot1 on the Stockfish depth ladder** to settle the row 0 caveat and
   give bot2_nnue a paired comparison point.
7. **Instrument first-move cutoff rate and TT hit rate.** Target >85% cutoffs.
8. **Re-measure king safety at n≥100.** If both evals really are anti-correlated
   there, it is the largest single gain identified so far and it argues for
   oversampling king-danger positions in training.

**P2 — real gains, moderate effort**
9. **Pondering.** Explicitly allowed, and the process keeps its core while the
   opponent thinks. Potentially close to a doubling of effective search time.
10. **Port bot2_nnue's endgame conversion behaviour into bot1** as insurance,
    since bot1 is what would ship today.
11. **Eval tuning** — only with SPRT in place.

**P3 — high ceiling, high risk, decide with data not vibes**
12. **numba bitboard movegen.** The 33–36us generation cost is now clearly the
    ceiling: with the net, eval is 8% of node cost and movegen is most of the
    rest. Must be a parallel branch validated by perft. Multi-day.
13. **Bigger net.** Hidden stays 256 until the data is fixed — 926k positions
    against 197k params is 4.7 samples/param and 512 would halve that. The speed
    budget is there (512 would still be under bot1's 12.5 us), so this is a data
    question, not a speed one.

**Competition timing (does not change with iteration)**
- Uploads close 11 Sept 11:00. Tie-break favours the **earlier** final
  submission, so submit early, not at 10:59.
- 6 uploads per team per day. The latest that passed validation is the one that
  plays, so a regression that validates *does* go live.
- Ladder Elo does not qualify anyone. The Swiss over locked builds decides.
  Treat the ladder as a free live test harness.

---

## 9. Principles that keep proving right

- **Nothing merges without SPRT.** A 60% score over 15 games is noise.
- **Profile before architecting.** The 33us/1us ratio killed a day of planned
  Zobrist work before it started. The 3.2us/12.5us ratio killed the assumption
  that a learned eval costs speed.
- **Correlation, not error.** A search evaluation is a ranking function.
  Alpha-beta cares only about ordering. bot1's slope of 1.60 means it
  over-reports by 60% and it does not matter; bot2_nnue's better mae with worse
  correlation lost 280 Elo.
- **Measure on the distribution you will play.** A metric averaged over the wrong
  distribution is worse than no metric, because it reads as success. Rated games
  start from curated near-level positions; every position in a game between
  comparable engines is near-level.
- **Give the model what you already know.** Material as a fixed skip connection
  costs nothing and forces the capacity onto what is actually unknown. It should
  have been the default from the start.
- **A loss curve cannot see a distribution fault.** Validation loss moved the
  wrong way across the change that added the missing signal.
- **Keep a terrible opponent around.** It turns invisible bugs into impossible
  scores.
- **Test adversarially, at the boundaries.** The severe bugs only appeared under
  time pressure and at 50ms budgets, never in normal play.
- **Report the sample size next to the result, always.** "60%" and "60% ±184
  Elo over 15 games" are different claims.
- **Goodhart.** Never optimise Stockfish move-match rate or raw centipawn loss.
  Use Stockfish as an opponent, a labeller and an autopsy tool, not as a target.
