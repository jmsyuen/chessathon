# Iteration log

Living document. Update it at the end of every iteration, before starting the
next one. It exists so that no future iteration re-derives something we already
paid for, and so that no bug we have already found ships twice.

**Update protocol.** Add a row to the ledger (§1). Tick or extend the regression
checklist (§4). Move anything from "next steps" (§8) into the ledger with its
measured result, including the ones that failed — a change that measured neutral
is information and must not be silently retried. Correct any fact in §3 that
turned out wrong rather than leaving both versions in.

**Naming.** The NNUE build is `bot3_nnue` and its tooling is the `v3_*` files.
Earlier revisions of this log called it `bot2_nnue`; that name is retired and
every reference below has been renamed. The families are:

| Family | Build | What it is |
|---|---|---|
| 1 | `bot1_baseline` | Classical: PVS-ish negamax, TT, quiescence, PSTs |
| 2 | `bot2_standard` | From-scratch rebuild. Unshipped, not separated from bot1 |
| 3 | `bot3_nnue` | Learned eval on bot1's search. Unshipped, data fault diagnosed |
| 4 | `bot4_ordering` | bot1's eval on a rebuilt search. Superseded; now the fallback inside bot5 |
| 5 | `bot5_nnue2` | bot4's search + the retrained net, blended. **SHIPPED at `NET_WEIGHT = 128`** |
| 6 | *next* | See §8 P0 — sweep below 96, then the phase taper |

**Where things live.** `bots/bot5_nnue2/` is a *directory* build: the agent plus
exactly one `.npz` beside it, which is how `tools/agents.py` carries weights.
A second `.npz` in that directory would silently overwrite the first at the same
target. The network is read from the file (`hidden`, `qa`, `qb`, `scale`,
`cp_scale`), so **a new network is a weights swap, not a new build** — bot6 and
after can reuse the shipped 512 net with no retraining. Retraining is forced only
by a feature-encoding change (mandatory), an architecture change, or better data.

---

## 0. How this project persists — read first

Nothing about the working environment survives a session. Concretely:

| Where | Survives? | Use it for |
|---|---|---|
| **Project files** | **Yes.** Read-only at `/mnt/project/`, always in context | **The source of truth.** Every file the next iteration needs to read or edit must be here |
| **GitHub repo** (`jmsyuen/chessathon`, public) | **Yes**, and fetchable from the sandbox | Binary artifacts project files cannot hold: `.npz` weights, opening books |
| **Past chats in this project** | Yes, searchable | Rationale, measurements, why a decision was made. **Not** for recovering source code |
| Assistant sandbox (`/home/claude/...`) | **No — destroyed at session end** | Scratch only: data sets, engine binaries, checkpoints |
| **Local PC** (Ryzen 5 3600, LMDE / WSL2) | **Yes** | **All measurement volume.** Data generation, SPRT, 120 s validation. Results go back via the repo |

**This has already cost us.** §5 of an earlier version of this log listed
`h2h.py`, `run_rung.py`, `autopsy.py`, `gen_book.py`, `test_edges.py` and
`test_conversion.py` as existing tooling. **None of them existed anywhere.** They
were written in a sandbox that has since been wiped and were never uploaded. A
later iteration hit the exact problems `h2h.py` and `run_rung.py` were built to
solve — chunking long runs past the ~5 minute call limit — and re-derived worse
versions from scratch.

Iteration 3 (bot4) re-derived `h2h.py` for the third time. It is now uploaded.

Past chats are a poor substitute for uploading. A file created once and then
patched five times in-session cannot be reliably reconstructed from a transcript.

**Rule: if a tool is worth a row in §5, upload it the same day.**

**Clone the repo first thing, do not ask for re-uploads.** Verified fetchable
from the sandbox — `codeload.github.com` and `raw.githubusercontent.com` are
allowlisted, a private repo would not work because there are no credentials:

```
curl -sL https://codeload.github.com/jmsyuen/chessathon/tar.gz/refs/heads/main | tar -xz
```

`bots/bot3_nnue/bot3_nnue.npz` is there and loads correctly (QA=1024, QB=2048,
hidden=256). `tools/` now holds `bench.py`, `gen_openings.py`, `h2h.py`,
`kernelbench.py`, `openings.py` and `selftest.py` — **the missing-test-tools gap
recorded here previously is closed; both gates are in the repo.** Verified
against a fresh clone, not against this file.

**The remaining clone-only blocker is different, and it bites harder.**
`versions/` and `baselines/` **do not exist in the repo**, and no Makefile target
creates them. The harness loads an agent as a *directory* containing `agent.py`,
but builds are stored as flat files in `bots/`. So every match tool defaults to a
path that is not there — `h2h.py` to `versions/bot4` and `baselines/bot1fix`,
`ladder.py` to `baselines/stockfish`. A fresh clone can run `selftest.py` and
`kernelbench.py` and cannot play a single game. **This is the thing to fix before
the PC runs anything — and it is now fixed.** `tools/agents.py` (`make agents`)
materialises `bots/<name>.py` into `versions/<name>/agent.py`, the `bot0_*` and
`bot_stockfish_spar` builds into `baselines/<name>/agent.py`, and copies
`bot3_nnue.npz` to `weights/nnue.npz` inside its agent directory, which is the
path the agent actually opens rather than the name it is stored under.
`--check` verifies without writing; `--ship BUILD` points the root `agent.py` at
a build. Run it after every clone. `bots/ladder.py` is excluded: it is a tool,
and materialising it as an agent produced a `versions/ladder` that dies on the
first `get_move`.
`bota_serena.py` is a teammate's bot held for later testing, deliberately
outside the `bot<N>` numbering; it is not one of our iterations and owes no
ledger row. **Standing cost:** the repo is public, so the engine is visible
to every other team before the locked-build Swiss. Not a rules breach, but it
should be a deliberate choice rather than a side effect.

Scratch that is expected to die and should never be uploaded: the training csv
(~59 MB) and its parsed cache (~130 MB), float checkpoints, and above all the
Stockfish binary — shipping or calling an engine from the zip is retroactive
disqualification.

---

## 1. Iteration ledger

| # | Date | Build | Change | Measured result | Verdict |
|---|---|---|---|---|---|
| 0 | 3 Sep | bot1 | Baseline, speed-to-ship. PVS-ish negamax, TT, quiescence, null move, LMR, aspiration, tapered eval, mating drive, contempt | 100% vs SF d4; 50% vs SF d6 — **withdrawn, see caveat** | Superseded by bot4 |
| 1 | 3 Sep | bot2_standard | From-scratch rebuild: staged movegen, bitboard eval, SEE, lazy eval, pawn hash | 60% vs bot1 over 15 games (Elo +70 ±184); 0 losses in 52 games | **Not separated. Not shipped.** |
| 2 | 3 Sep | bot3_nnue | Learned eval: material + (768→256)×2 perspectives, SCReLU, int16, numba kernel. bot1's search carried over unchanged | 16.7% vs bot1, 12 games at 3s+100ms (Elo −280). Selftest FAILS one tactic | **Not shipped. Root cause: §4 bug #8** |
| 3 | 3 Sep | bot4_ordering | bot1's evaluation unchanged except two fixes; search rebuilt for branching factor | **+118 =36 −44 over 198 games vs bot1_baseline = 68.7%, +136 Elo (+92..+186).** LLR +1.53, undecided. Selftest clean. Equal-or-deeper than bot1 on 6/6 suite positions using 5.5× fewer nodes | Superseded by bot5. Now the **reference baseline** |
| 4 | 4 Sep | **bot5_nnue2** | bot4's search **byte-identical**; `evaluate()` replaced by the int16 kernel with material as a fixed skip connection, retrained on 10.2M stratified positions. `NET_WEIGHT` blends network with the hand-crafted eval. Bare-king endings hand off to the classical eval entirely | **Blend, `NET_WEIGHT=128`, hidden 512: 85.6% over 59 games vs bot4, +310 Elo (+218..+460).** Pure net (256): 61.1% over 427 games, +79 (+47..+111). Selftest **clean at 128**; **FAILS `must recapture` at 256**. First-move cutoff 86.9%. Zero failures in 486 games | **SHIPPED at `NET_WEIGHT = 128`. The reference for everything after.** |
| 5 | 4 Sep | bot4_nobreak | `STABLE_ITERATIONS` 3 → 99, disabling the stability early-break. One line | **+149 =64 −87 over 300 games vs bot4 = 60.3%, +73 Elo (+38..+109).** But only **+20 Elo (−30..+70) on top of bot5-blend**, 156 games | Kept. **Not the win it looked against bot4** |
| 6 | 4 Sep | `NET_WEIGHT` sweep | 96 / 160 / 192 against the shipped 128, all with nobreak, 156 games each | 96 **+22** (−25..+70); 160 **−18** (−66..+29); 192 **−79** (−132..−30). Paired on identical slots: 96 beats 192 at **t=2.91**. Zero failures in 624 games | **128 retained.** Curve peaks at or below 96 — see §8 P0 |

**Caveat on row 0, now settled.** "100% vs SF d4" never reproduced and is
withdrawn. Both recorded crossovers came from 6-game rungs and neither is
trustworthy; see §3 on rung sample size.

**Standing conclusion (changed this iteration).** **bot5_nnue2 at
`NET_WEIGHT = 128` with the hidden-512 network is the shipped build and the
reference every later claim is measured against.** bot4 is superseded but stays
as the fallback: bot5 falls back to bot4's evaluation verbatim if the weights
file is missing or malformed, so the worst case is that it plays as bot4.

**The headline finding of this iteration is not the one we set out to test.**
The hypothesis was "replace the evaluation with a network". The answer is
"average the network with the hand-crafted evaluation" — worth **~240 Elo over
the pure network**, verified paired on identical opening and colour slots
(t = 4.26). The network is better than the HCE where it was trained
(quiet level r **0.535** against bot4's 0.378) and wrong outside it. The blend
cancels errors that do not correlate. **Bug #8 is fixed and closed.**

## 2. Metrics we agreed on, and what has actually been measured

Authority order matters. Items 1 and 2 gate everything else.

| # | Metric | Target | Status |
|---|---|---|---|
| 1 | Self-play Elo vs previous version, SPRT `elo0=0, elo1=5, α=β=0.05` | pass before merge | **PARTIAL, and the gap is on the shipped arm.** bot5 w128 has **59 games** (+310 Elo, +218..+460); w256 has 427; the sweep runs 156 each. `tools/h2h.py --workers 6` resumes from its `--out` file. `--elo1 10` halves the cost and is the deliberate default in `tools/run.sh` |
| 2 | Failure rate per 100 games (flag/crash/illegal/drawn-while-winning) | exactly zero | **PASS. Zero failures across 1,481 games** on 4 Sep, every build and every arm |
| 3 | Nodes per second on fixed benchmark suite | track per build | On one sandbox core at 300 ms: bot4 **25.0k / 89.1% cutoff**, bot5 blend **21.7k / 86.9%**, bot5 pure net **30.7k / 83.5%**. The pure net buys speed and spends it on worse ordering |
| 4 | First-move cutoff rate | >85% | bot4 89.1%, **bot5 blend 86.9% (pass)**, bot5 pure net 83.5% (**fail**). With the search held byte-identical this is a ten-second proxy for evaluation quality — read it before any games |
| 5 | Average depth at real 120s+0.5s control | sanity check | **STILL A GAP**, but no longer a resource problem. This is a *sanity check*, not an SPRT — 20-30 games is enough to see whether depth drifted from fixed-node testing, ~1 h on the PC. Schedule it before the lock |

**Metric 3 is now actively misleading and must be read with metric 4.** bot4 runs
at **12.9k nps against bot1's 27.0k** — less than half — and is decisively
stronger. On the fixed suite at an identical 949/1898 ms budget:

| Position | bot4 | bot1fix |
|---|---|---|
| start | d6, 2,142 nodes, 137 ms | d6, 23,012 nodes, 702 ms |
| italian | **d6**, 4,460, 408 ms | d5, 33,841, 1,454 ms |
| sicilian | d6, **2,944**, 193 ms | d6, 43,607, 1,473 ms |
| french | **d5**, 8,443, 637 ms | d4, 12,854, 564 ms |
| queens gambit | **d7**, 8,700, 827 ms | d4, 11,614, 541 ms |
| kings indian | **d6**, 4,331, 308 ms | d5, 40,090, 1,690 ms |

Equal or greater depth in every position, on 5.5× fewer nodes and 2.6× less wall
time, with an identical evaluation on both sides. **Track nodes-to-depth, not
nodes per second.** A build that halves its node rate and quarters its node count
is a large win, and metric 3 alone scores it as a regression.

---

## 3. Environment facts — do not re-derive

**Profiling (python-chess 1.11.2, one core).** These drive architecture:

| Operation | Cost | Consequence |
|---|---|---|
| `list(board.legal_moves)` | 33–36 us | Dominates. Avoiding generation beats optimising anything else |
| `board.push()` + `pop()` | 6 us | Cheap. Push-then-test is cheaper than `gives_check()` |
| `board.is_legal(move)` | ~8 us | **Cheaper than generating.** This is what makes staged movegen pay |
| `board._transposition_key()` | 1 us | Cheap. **Hand-rolled incremental Zobrist is not worth the bug risk** |
| bot1 `evaluate()`, pure Python | 12.5 us | ~25% of node cost |
| **NNUE forward, numba, hidden 256, 2 perspectives** | **3.2 us** | ~8% of node cost |
| Packing 8 bitboards for the kernel | 0.55 us | Included in the 3.2 us above |
| `_see()` on one capture, pure Python | ~10 us | Only run when victim value < attacker value; that filter removes most calls |

**A jitted learned evaluation is cheaper than a Python piece-square evaluation
on this platform.** This inverts the usual assumption. bot3_nnue searched 20–40%
faster than bot1 and reached identical depths. Any future eval work should assume
the numba path is free and spend the budget on quality.

**Transposition table sizing — corrected.** bot1's comment claims an entry costs
"roughly half a kilobyte". Measured with `tracemalloc`, including freshly built
key tuples: **268 bytes per entry**. bot4 therefore runs two tiers of 500k
entries for about **270 MB** against a 2 GB container, and never calls
`dict.clear()` mid-search. Ageing is `_tt_old = _tt; _tt = {}`, which drops half
the table in constant time; probes check the live tier then the old one and
promote on a hit.

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
- **Stockfish is one command: `apt-get download stockfish` then
  `dpkg-deb -x`.** `archive.ubuntu.com` is allowlisted and it fetched 33.5 MB in
  2 s, giving Stockfish 16 at 866k nps. This is far simpler than the GitHub
  release-asset route and should be the default. The GitHub route still works if
  a specific version is needed: `release-assets.githubusercontent.com` is
  allowlisted, the API is rate limited, so build the URL by hand —
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
**Compute topology — changed this iteration, and it changes the loop.** Local
hardware is now available and it is roughly **10× the sandbox** for anything that
parallelises:

| | Assistant sandbox | Local PC (Ryzen 5 3600) | M4 Air |
|---|---|---|---|
| Cores | **1** @ 2.1 GHz Xeon | 6C/12T @ ~4.0 GHz | 4P + 6E, **fanless** |
| RAM | 3 GB | 16 GB | 16 GB |
| GPU | none | RTX 2060 6 GB | — |
| Job length | **~5 min per call, processes reaped between calls** | unlimited | unlimited |
| Best for | **Fidelity** | **Volume** | Editing code, short runs |

**The sandbox is a better proxy for the competition container than the PC is** —
1 core, x86, ~2 GHz against the platform's 1 core / 2 GB. Measure nps,
nodes-to-depth, import budget and clock discipline *there*. A depth measured on a
4 GHz Ryzen is a depth we will not reach in the Swiss, and `_budget` tuned
against it is tuned wrong. If a timed run must happen on the PC, calibrate the
ratio once with `tools/kernelbench.py` and scale the control.

**The GPU is close to irrelevant.** Training is 6 s/epoch on *one* core and the
numpy trainer already works; a GPU cannot improve on four minutes. The bottleneck
is data generation, which is CPU-bound. Do not rewrite the trainer for CUDA. It
would only pay above ~50M positions with an architecture sweep, and that is not
the problem we have.

**Use LMDE natively for timed runs, WSL2 is acceptable for data generation.**
SPRT and the 120 s validation are measuring wall-clock behaviour, so hypervisor
jitter contaminates the thing being measured and can produce flags that look like
time-management bugs. Data generation is throughput-bound and does not care.

**Cap game concurrency at 6, not 12.** Each game runs two agent processes but
only one thinks at a time, so six games saturates six physical cores. SMT
oversubscription distorts every timing number. Pin BLAS/torch to one thread per
process or each will spawn six and thrash.

**Parallelising the existing tools needs almost no code.** `v3_gendata.py`
already takes `--out` and `--seed`, so six processes with distinct seeds and
output files then `cat` together — no change at all. **`tools/h2h.py` is the one
gap — now closed.** It had `--out` and `--games` but no offset, so six copies
would have played the same six games and read as a completed SPRT. `--workers N`
now shards by **opening pair**, not by game, so each worker stays internally
colour-balanced and no two touch the same opening; the parent pools the shards
and stops on the **pooled** LLR. Two further bugs surfaced in testing: a clean
sweep gave zero variance and an LLR of ~1e10 (harmless when a human read it at
the end of a serial run, not harmless when it drives process termination — a
4-0 start would have killed a 300-game run and reported a merge), and resuming
double-counted because the pooled result was written back into the base file.
Both fixed; LLR is now regularised with half-win/half-loss pseudo-counts and no
early stop is allowed under 20 games.
Its `--deadline-s 250` default is a sandbox artefact and should be raised.

- Sandbox has **1 core**, same as the competition container. No parallel games
  *in-session* — that is what the PC is for.
- **Background processes are reaped between tool calls — `setsid` and `nohup` do
  not help.** Two runs were lost to this. Additionally **bash calls are killed at
  ~5 minutes**. Anything long must be foreground, chunked, and checkpointed to
  disk every iteration. `v3_train.py --state` does this for float weights, Adam
  moments and the epoch counter. `v3_gendata.py` is line-buffered so a killed run
  keeps every position it wrote. `tools/h2h.py` checkpoints after every game.

**Stockfish ladder — the crossover number is far noisier than it looks.**
Paired runs, both builds, same session, same control (6 s + 0.06 s):

| Rung | bot4 | bot1fix |
|---|---|---|
| nodes:100 | 62.5% (8 games) | 50.0% (8 games) |
| nodes:300 | 25.0% (6 games) | 25.0% (6 games) |
| nodes:1000 | **33.3%** (6 games) | **8.3%** (6 games) |

Interpolated 50% crossover: **bot4 ≈ 144 nodes, bot1fix ≈ 100 nodes.**

Two things follow. First, the rig is sound: bot1fix's 8.3% at `nodes:1000`
reproduces the historically recorded figure exactly. Second, **the earlier
~340-node crossover for bot1 does not reproduce** — same build, same control,
this run gives ~100. Neither is wrong; both are 6-game samples. The crossover
estimate **swings by 3× at 6–8 games per rung**, which promotes the standing
"use 40+ games per rung" note from advice to a measured requirement. Do not quote
a crossover from fewer than 40 games per rung, and do not compare a new
crossover against an old one measured at a different sample size.

**Agent contract (re-verified at source 3 Sep). It does not change once the
qualifier starts on 4 Sep, so it is now frozen — no need to re-check before the
lock.**
- Eligibility: at least one UK university student per team. 50 London seats.
- Books and tablebases are permitted as shipped data; `chess.polyglot` and
  `chess.syzygy` are in the base image.
- Pondering is explicitly allowed: the process keeps its core after `get_move`
  returns. **No build uses this.** See §8 P2 for the mechanism and the conditions.
- Import + warm-up: bot1 0.41 s, **bot4 0.30 s**, **bot3_nnue 14 s** (almost all
  of it importing numba and llvmlite rather than compiling). Against a 60 s
  budget, so fine, but it is no longer negligible for the NNUE line.
- numba `cache=False` is mandatory. The platform filesystem is read-only apart
  from `/tmp`, so a cache write would fail next to the source. `/tmp` also starts
  empty for every game and is deleted with it, so it is never a cache between
  games.
- Weights ship as a second artifact: `agent.py` at the zip root plus
  `weights/nnue.npz`. `harness/package.py` already includes `weights/` via
  `DEFAULT_INCLUDES`. Resolve the path from `Path(__file__).resolve().parent`,
  never from the working directory.
- **Finished ladder games reveal the curated positions they were played from.**
  This is the real fix for the opening-set problem and beats guessing. Harvest
  them once the ladder has run.
- **House bots play the ladder showing their public CCRL ratings, and cannot
  qualify.** That is a free, externally-grounded absolute scale. It is worth more
  than the Stockfish node ladder, which §3 already shows is unreliable at 6-game
  rungs and which measures altitude against an opponent that blunders in places
  the field does not. Read our live ladder Elo against the house bots and treat
  that, not the crossover node count, as the headline altitude number.
- **The first FEN received is the game's starting position**, and repetition plus
  fifty-move counts begin there, not from the standard start. `_sync` is correct
  on this; anything that rebuilds history must stay correct on it.
- Validation plays **two smoke games against a house agent**, one as each colour,
  from curated positions, and publishes the verbatim log. stdout/stderr comes back
  up to **8 KB** — a separate limit from the 4 KB move payload that counts as an
  illegal move.
- Process limit is **128** on one core.

**Harness gotchas (confirmed in source).**
- `board.outcome(claim_draw=True)` — threefold and fifty-move are claimed
  automatically *against* us. History must be reconstructed from FENs.
- 300-ply adjudication is **pure material**, no positional terms. **bot4 is the
  first build to play for it**, ramping the referee's own P1/N3/B3/R5/Q9 formula
  in from ply 240 to 60% weight at ply 300. Untested in a real long game; one
  adjudication occurred in the 30-game run and bot4 won it.
- The 500 ms watchdog grace is not usable slack — the referee flags the instant
  the clock goes negative.
- `runner.py` does not wrap `get_move`. Any uncaught exception is a lost game.
- The zip is first on `sys.path`. Never name a file `chess.py`, `types.py`.
- stdout cap 4096 bytes; over-cap counts as an illegal move.

---

**Agent contract, re-read at source 4 Sep 2026. Two items are new and both are
actionable.**

- **The process keeps its *dedicated* core after `get_move` returns, and
  pondering is allowed.** "Dedicated" is the word that matters: it closes the
  open question in §7 about whether two containers share a core. They do not.
  Pondering is free. During your own move one thread is fastest — threads past
  the first share the single core and cost you time.
- **Books and tablebases are permitted as shipped data**, and `chess.polyglot`
  and `chess.syzygy` are in the base image. 4-man Syzygy WDL fits the 50 MB
  budget several times over; 5-man does not (a few hundred MB). Verify actual
  file sizes before committing to it.

---

## 4. Regression checklist — run against every new build

These are the bugs found so far. Most are pattern-level and will recur in any
rewrite. **Bugs #9 and #10 are new this iteration and are the most dangerous on
the list, because both present as a mildly weak evaluation rather than a fault.**

| # | Bug | How it presents | Test |
|---|---|---|---|
| 1 | Root best move not set on the staged TT path | Engine returns its **depth-1 move** while searching deep. Scored 50% vs `greedy` | Score vs `greedy` must be ~100% |
| 2 | Unbalanced push/pop on clock abort | Returns a move for the **opponent**. Instant loss. Only fires under time pressure | Fuzz at 50 ms budgets; assert legality |
| 3 | Quiescence stands pat in check | Can be mated at a leaf and return a material score | Mate-at-leaf positions |
| 4 | SEE counts illegal king recaptures | Winning captures score as losing; suppressed in ordering, pruned in qsearch | Brute-force swap-off comparison |
| 5 | Floor division colour bias | Position and its mirror differ by 1 cp. Hit ~34% of positions | `evaluate(b) == evaluate(b.mirror())` |
| 6 | En passant classified as quiet | Ordered by history, exposed to LMR and futility | Assert ep is treated as a capture |
| 7 | Mangled boolean (`not x == 0`) | Silently wrong branch | Read every conditional in scaling code |
| 8 | **Training distribution collapse** | Net scores r=0.95 against engine labels and beats the hand-written eval on mae, then loses 280 Elo. In level positions its correlation is **r ≈ 0**: it learned to count material and nothing else | Correlate eval against engine labels **restricted to \|material\| ≤ 120**, from near-level openings. `v3_evalcmp.py --from-openings --max-imbalance 120` |
| 9 | **Shallow pruning discards quiet checking moves** | Engine misses forced mates **while reporting a healthy 90% first-move cutoff rate**. Three plies into a sacrifice the only saving move is usually a quiet check, and futility/LMP throw it away | Selftest `mate in 3, black` (`1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1`). Guard: never prune when `board.gives_check(move)`; test only when a prune would otherwise fire |
| 10 | **Razoring returns a material score from inside a forced mate** | Same symptom as #9 but **unreachable by any move-level guard** — razoring returns before a single move is generated | Same position. bot4's fix was to **remove razoring entirely**; it is worth a few Elo of speed and cost a forced mate |

**bot1 audit result:** fails #5 (134/400 positions measured this iteration; the
earlier 198/399 figure used a different sample). Fix is one line — truncate
toward zero instead of `//`. **Now applied as `bot1fix`, verified 0/400.** bot1
also has #2's missing `try/finally`, currently harmless *only* because it
searches on `board.copy(stack=False)`. That is protection by accident, not by
design. Anyone refactoring bot1 to search the tracked board reintroduces a losing
bug. **bot3_nnue inherits #2**, since it carries bot1's search unchanged.

**bot3_nnue audit result:** **passes #5** — 0 of 400 random positions asymmetric.
This is structural, not luck: both accumulators are computed relative to the side
to move and the result is never negated, so there is no white-relative-then-negate
step for floor division to bias. Any future eval that scores from White's
perspective and flips the sign reintroduces #5; a two-perspective net cannot.
**Do not re-audit this.**

**bot4 audit result:** passes 1–7 and 9–10 by construction. #2 is fixed *by
design* — every `board.push` is inside an explicit `try/finally` rather than
relying on a stack-free copy. #4 is guarded explicitly in `_see`. #5 is fixed.
#9 and #10 were found *in bot4 during this iteration* by the selftest and fixed
before any strength measurement was taken.

### Bug #8 in full

`v3_gendata.py` seeded games with 2–12 uniformly random plies and injected 10%
random moves during play-out. Result over 926,724 positions: median
\|material\| is **350 cp**, only **18.3%** are within 60 cp of level, only 31.5%
within 120 cp. In that distribution material counting explains nearly all the
variance, so gradient descent never had to learn a positional feature.

Measured on quiet positions from near-level openings, n=100:

| Eval | mae | **r** | sign |
|---|---|---|---|
| bot3_nnue, no skip connection | 146 | **−0.030** | 54.0% |
| bot3_nnue, material skip connection | 119 | **+0.188** | 51.0% |
| bot1 piece-square | 61 | **+0.336** | 49.0% |

Measured on the *original* lopsided distribution, n=200 — the flattering view
that hid it for most of the iteration:

| Eval | mae | r | slope |
|---|---|---|---|
| bot3_nnue | **222** | 0.944 | **1.11** |
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

**New and important for bot5.** Bugs #9 and #10 mean **bot4's search amplifies
evaluation error**. RFP, futility, LMP and the improving flag all key off the
static evaluation, so a net that is blind in level positions (bug #8) will not
merely choose badly — it will *prune* badly, on margins computed from a number
that carries no signal. bot3_nnue on bot1's gentler search lost 280 Elo; the same
net on bot4's search could lose more. **The evalcmp gate is therefore a hard
precondition for bot5, not a nicety.**

**Lesson worth keeping:** keep a deliberately terrible calibration opponent
(`greedy`, one-ply material). A 50% score against it is unmistakable. The same
bug measured against a real opponent looks like "the eval needs tuning".

---

**New this iteration. Each one produced a plausible-looking wrong answer rather
than an error, which is why they are here.**

| # | Bug | How it presents | Guard |
|---|---|---|---|
| 11 | **Learned eval has no basis outside its training distribution.** Resign-truncation at 600 cp removed every position where somebody is converting, so every corner king the network saw was a *castled, safe* one. Measured on K+R+B vs K: its opinion of driving the weak king centre→corner is **−67 cp** against the mating drive's **+31** and the HCE's **+103** | The ending is never won. Fifty-move claimed against you. No error, no crash | Hand drive-active positions (low phase, decisive edge, no pawns for the loser) wholly to the classical eval. Conversion tests in `selftest` are the detector |
| 12 | **`tools/agents.py` `WEIGHT_TARGETS` had no entry for a new NNUE build.** The `.npz` lands beside the agent instead of at `weights/nnue.npz` | Network silently does not load, the build runs the *fallback* evaluation, selftest passes, every measurement is of the wrong engine | `run.sh check` greps for the entry and prints `network loaded:` before anything else runs |
| 13 | **Environment knobs reach both agents.** The harness spawns both as children of one `h2h` process, so `CHESSATHON_NET_WEIGHT=96` sets it on the *opponent* too | A build plays itself at 50% and looks like "no effect" | Compare variants of the same build by baking constants into separate `versions/` directories — `tools/sweep.sh` does this, with a `grep` that aborts if a `sed` stops matching |
| 14 | **`best_hidden()` took `head -1` of the pass list, not the best.** Both nets cleared the gate; it shipped the weaker one | Everything works; you measure the wrong network | Fixed to read the r values out of `results/evalcmp.*.txt` and print which it chose |
| 15 | **`v3_checknet.py`'s float reference omitted the material skip connection.** The kernel adds material inside the forward pass and the trainer adds it as a fixed offset; the reference did neither | "Quantisation error" was mostly the material balance | `tools/checknet.py` supersedes it |
| 16 | **`selftest.py`'s `_reset()` left `_tt_old` alive.** Two-tier TT builds leaked search state between cases meant to be isolated | Order-dependent, intermittent tactic failures | Patched; any future build with more module state needs the same |
| 17 | **`bot4_ordering.py` as `agent.py` fails mypy** on `rest.remove(tt_move)` with a `Move \| None` | Style-gate only, no behaviour | One-line narrowing. Fixed in bot5, still open in bot4 |

## 5. Test suite

**The status column is not decoration.** Anything marked *missing* was written in
a wiped sandbox and no longer exists; those rows are work to redo, not tools to
run. See §0. Status names a **location**: `in project` = project files only,
`in repo` = clonable from GitHub only, `both` = safe. Anything sitting in one
place only is one accident away from the paragraph above.

| File | Status | Catches | Runtime |
|---|---|---|---|
| `tools/selftest.py` | both | Tactics, clock edges (0 ms→negative), promotions, ep, castling, single legal move, malformed FEN, endgame conversion, repetition tracking, clock discipline | ~4 min for bot4; split it if it grows |
| `tools/bench.py` | both | Match play mirroring the referee, alternating colours, Elo with interval | ~20 s/game at 3 s+100 ms |
| `tools/ladder.py` | both (repo: `bots/ladder.py`) | Stockfish staircase, reports the 50% crossover | ~1 min per 6-game rung at 6 s |
| `tools/agents.py` | in repo | Materialises `versions/` and `baselines/` from `bots/`. `--check`, `--ship BUILD`. **Run after every clone** | instant |
| `tools/collect.py` | in repo | Pools all `results/*.json` (folding shards), writes `results/SUMMARY.md`. The return channel to a session | seconds |
| `RUNBOOK.md` | in repo | How to run the PC battery and get results back | — |
| `baselines/stockfish/agent.py` | in project (`bot_stockfish_spar.py`) | Stockfish sparring partner speaking the agent contract | — |
| **`tools/h2h.py`** | **in repo** | **Chunked head-to-head with SPRT, checkpointed after every game.** The fix for the ~5 min call limit, re-derived three times now | 20–130 s/game at 8 s+0.5 s |
| **`tools/gen_openings.py`** | **in repo** | Generates a varied near-level opening set. **Replaces `gen_book.py` and needs no Stockfish** | ~30 s for 64 positions |
| **`tools/openings.py`** | **in repo** | 64 generated positions = **128 distinct games**, lifting bench.py's 24-game ceiling | (data) |
| **`tools/kernelbench.py`** | **in repo** | Fixed 12-position suite: nodes, nps, depth, first-move cutoff rate. Metric 3 and 4 | ~10 s/build |
| `v3_evalcmp.py` | both (repo: `bots/bot3_nnue/`) | **Bug #8.** Both evals vs fresh engine labels, split by check / capture-best / king-exposed / material level | ~2 min at n=200 |
| `v3_gendata.py` | both (repo: `bots/bot3_nnue/`) | (generator) Stockfish-labelled self-play. **Contains bug #8** — the file P0 edits first | 404 pos/s |
| `v3_train.py` | both (repo: `bots/bot3_nnue/`) | (trainer) numpy + numba, jitted sparse gather/scatter, `--state` checkpointing, quantise-on-save | 6 s/epoch |
| `v3_checknet.py` | both (repo: `bots/bot3_nnue/`) | Quantised kernel vs float model; material sanity | ~30 s |
| `autopsy.py` | **MISSING** | Was a drawn or lost game ever winnable. **Now wanted: bot4 lost 6 of 30** | ~4 min |
| `test_edges.py`, `test_conversion.py` | **MISSING** | Superseded by `tools/selftest.py` | — |

**Which of these run where.** `selftest.py`, `kernelbench.py`, `v3_checknet.py`
and `v3_evalcmp.py` are **fidelity** tools — short, single-core, and they belong
in the sandbox where the hardware matches the platform. `v3_gendata.py`,
`h2h.py`, `bench.py` and `ladder.py` are **volume** tools and belong on the PC,
parallelised. `v3_train.py` runs anywhere; it is four minutes either way.

**Status rows go stale silently. Re-verify against a clone, not against this
file.** These two rows read "in project only — add to repo" for an iteration
after both files had been committed, and that stale claim was repeated as fact.
The check is one command: `curl -sL .../main | tar -xz && ls tools/`.

Feature encoding is verified identical between `v3_train.py` and the agent kernel
against a python-chess reference — do not skip that check on any rewrite. A
silent mismatch looks exactly like a badly trained net.

**The 24-game ceiling is lifted.** `bench.py` OPENINGS has 12 positions and both
engines are deterministic, so 24 distinct games was the ceiling and every game
past it was a replay. `tools/openings.py` has 64 → 128 distinct games. **Note
`tools/ladder.py` still imports OPENINGS from `bench.py`**, which is fine at
≤24 games per rung but must be repointed before running the 40-game rungs §3
now requires.

## 6. Measurement costs — read before promising a verdict

| Difference to detect | Games needed |
|---|---|
| ~200 Elo (adding quiescence) | ~40 |
| ~50 Elo (real eval improvement) | 200–400 |
| ~10 Elo (a tuning tweak) | 2,000–4,000 |
| 5% score difference at our draw rate | **~246** |

**SPRT termination is slower than the effect size suggests, and this surprised
us.** With `elo0=0, elo1=5` the two hypotheses differ by a score of 0.500 vs
0.507, so each game carries very little evidence about *which* is true even when
the true difference is 176 Elo. **The rate recorded here previously, ~0.02 per
game, was wrong by half.** Recomputed from bot4's actual +20 =4 -6: LLR +0.29
over 30 games is **~0.0099 per game**, so the ±2.94 boundary needs **roughly 300
games, not 130**. Every "~130 games" figure elsewhere in this log traced to that
one bad rate and has been corrected. A large effect does **not** buy early
termination here.

**`--elo1` is the lever, and it is now a flag.** The hypotheses are what make
this slow: at `elo1=5` the test asks "is this at least 5 Elo better", and 5 Elo
is a score difference of 0.007, so each game carries almost no evidence. At
`elo1=20` the same 73% result decides in **~90 games** instead of ~300 — at the
cost of rejecting a genuine +10 Elo change. elo1=5 is a Fishtest setting for
banking tiny gains over months. With the lock close, pick deliberately rather
than inheriting the default.

At 8 s+0.5 s on one core a game costs 20–130 s. **One core gives roughly 40–80
games per hour.** A 15-game result is a smoke test, not a verdict. A −280 Elo
result over 12 games is an exception — the interval excludes zero comfortably.

**Constraint that will not go away:** a black-box `get_move(fen, time_left_ms)`
cannot be driven at fixed nodes. Cross-engine comparison must be fixed time. Use
fixed nodes only for A/B testing variants of an engine we control internally.

**Compressed control caveat.** 8–12 s + 0.5 s keeps the increment truthful (both
time managers assume 500 ms) and compresses only the base. It slightly favours
lower per-move overhead. Any final candidate must be re-checked at 120 s + 0.5 s.

---

## 7. Known gaps and untested areas

- **bot5's own SPRT is the thinnest evidence in the repo.** The shipped arm
  (`NET_WEIGHT=128`) has **59 games**; the arm we do not ship has 427. The
  `verdict` stage runs `ARMS="256 128"` in order and spent its budget on the
  wrong one. Fix with `ARMS=128 bash tools/run.sh verdict`.
- **`NET_WEIGHT` below 96 is unexplored**, and the curve says that is where the
  peak is. Between 0 (= bot4, ~−310 relative) and 96 (+22) the curve climbs
  ~330 Elo, so the rise is steep somewhere nobody has sampled. §8 P0.
- **`_budget` has never been tuned.** Carried unchanged through bot1, bot2, bot4
  and bot5 — five iterations. The nobreak result (+73 vs bot4) was a *time
  management* change in disguise: all it did was stop discarding budget already
  granted. That is direct evidence this area pays, and bot5 now runs at roughly
  twice bot4's node rate, so a budget model fitted to a slower engine is fitted
  to the wrong engine.
- **The pruning margins are calibrated to an evaluation we no longer run.** RFP
  80/ply, futility 110+95/ply, delta 190, aspiration ±25, null reduction
  `(static−beta)//200` were all tuned against bot1's eval at slope 1.60. bot5
  runs a blend at `cp_scale` 893 with double the node rate. Every one is now
  approximate, and this is the same cheap-constant sweep that just paid 240 Elo.
- **12.4% of games end in a referee-claimed draw** (72 threefold + 34 fifty-move
  out of 857). Some are genuinely drawn; some are thrown-away half points.
  Nobody has looked. `autopsy.py` still does not exist and this is the highest
  information-per-minute item on the list.
- **143 non-checkmate endings sit unexamined in `results/`.**
- **Real-control (120 s + 0.5 s) play still barely tested.** Metric 5 open across
  four iterations now.
- **King safety may have flipped.** bot5(512) scores r **0.880** on quiet
  king-exposed positions against bot4's 0.216 — the weakness both earlier evals
  got *backwards*. But n=21. Answering at n≥100 needs n≈10,000 sampled positions
  or a targeted sampler; king-exposed positions are ~1% of a normal draw.
- **hidden 512 beat 256 on the gate metric (0.535 vs 0.475) yet failed a tactic
  256 passed.** More capacity fitting the same distribution harder made
  out-of-distribution behaviour worse. Unexplained.
- **The training filter excludes the positions the eval is used on.**
  `tools/gendata.py` keeps a position only when no capture is available, but bot5
  calls `evaluate()` at interior nodes to drive RFP, razoring, futility and
  `improving` — at nodes where captures exist. Leading hypothesis for
  `must recapture`. §8 P2.
- **SEE ignores pins.** 2 sign errors in 2,543 captures, both conservative.
  Standard. Not worth fixing. Do not re-derive.

**Resolved this iteration, do not re-open:**

- ~~Bug #8, the training distribution~~ — fixed. 47.3% of 10.2M rows inside
  ±120 cp against v3's 31.5%; quiet level r 0.336 → 0.535.
- ~~`hangs nothing, must recapture` regression from the material skip~~ — closed
  by the data fix at hidden 256 and at `NET_WEIGHT=128`, not by a patch.
- ~~Endgame conversion under a learned eval~~ — the network believed a cornered
  king was *safer* (−67 against the mating drive's +31). Fixed by handing
  drive-active positions to the classical evaluation.
- ~~Whether the two containers share a physical core~~ — **the docs settle it.
  The process keeps its *dedicated* core after `get_move` returns. Pondering is
  free.** See §3.

## 8. Next steps, prioritised

Ordered by expected Elo per hour, with the cheap and reversible first. Items P0
and P1 need **no retraining** — the network is read from the `.npz` and any
build can reuse it.

### P0 — finish the curve that is already paying

1. **Sweep `NET_WEIGHT` below 96.** `bash tools/sweep.sh weight 48 64 80`, ~2 h,
   same reference and openings so all six points land on one curve. 96 (+22) and
   128 are within noise of each other and everything above falls off steeply, so
   the peak is at or under 96 and nobody has looked there. One constant, no
   retraining, cannot introduce a correctness risk the gate would not catch.
2. **Finish bot5's own SPRT.** `ARMS=128 bash tools/run.sh verdict`, ~1 h.
3. **Phase-tapered `NET_WEIGHT`.** ~1 h. Follows directly from the diagnosis:
   the network is measurably better in the middlegame and provably wrong in bare
   kings, and 128 is currently flat across both. **`phase` is a continuous 0–24
   scalar counting non-pawn material — it does not distinguish opening from
   middlegame** (both are 24) and pawns count zero, so this is one interpolation,
   not three buckets. Two constants:
   `NET_WEIGHT_MG` at phase 24 and `NET_WEIGHT_EG` at phase 0, linear between.
   The band is really phase 8–24, since below 8 with a decisive edge the position
   already goes wholly to the classical eval.

### P1 — cheap constants in areas now evidenced to pay

4. **Review `_budget`.** ~1 h. See §7 — untouched through five iterations, in the
   exact area nobreak just paid +73 from, and now fitted to an engine running at
   half the current node rate.
5. **Sweep the pruning margins.** ~2 h. Calibrated to bot1's eval at slope 1.60;
   we run a blend at `cp_scale` 893. Same pattern as the `NET_WEIGHT` sweep.
6. **Write `autopsy.py`.** ~1 h, no games needed. 143 non-checkmate endings are
   already sitting in `results/`. Decides whether tablebases are worth 25 Elo or
   5, and whether the 12.4% claimed-draw rate is unconvertible or thrown away.

### P2 — retrain, changing the data and not the architecture

7. **Regenerate with the filters fixed, ~4 h.** Do *not* scale the dataset up —
   at 10.2M the 512 net already has ~26 samples per parameter and its train/val
   gap (0.00850 / 0.01006) says it is fitting what it was given, not starving.
   Four changes, in order of suspicion:
   - **keep positions where a capture is available.** The eval is used at
     interior nodes to drive pruning, and those nodes have captures. This is the
     distribution mismatch, and the leading hypothesis for `must recapture`.
   - stop truncating the **winner's** conversion — that is why the network
     thought a cornered king was safe,
   - add a deliberate low-piece slice,
   - raise king-danger acceptance properly; 2× on a ~1% population barely moves it.

   Sizing, measured on the 4 Sep run: 65.2 B/row csv, 140 B/row cache,
   **604 B/row parse RAM**, ~940 kept/s across 6 cores, 71 min to train 512 for
   40 epochs. 10M ≈ 2.0 GB on disk and 6.2 GB RAM; **15M peaks at 9.1 GB and is
   the ceiling on 16 GB** until `load()` is rewritten as a two-pass
   pre-allocated fill (~20 lines).
8. **Tablebases, 4-man Syzygy WDL.** Shipped data plus a probe; `chess.syzygy` is
   in the base image and books/tablebases are permitted. 5-man does not fit 50 MB.
   **WDL says a position is won, not how to win it** — return a flat score and
   every winning move evaluates identically and the search shuffles until the
   referee claims fifty moves. Combine with `_mating_drive` for direction, or use
   it only to bound the score. Estimate 10–25 Elo; do item 6 first.

### P3 — real gains, real risk

9. **Pondering.** Now unblocked: the docs confirm the process keeps its
   **dedicated** core after `get_move` returns. Worth ~1 ply on a correct
   prediction, so ~0.6 ply averaged over a realistic hit rate — call it 20–40
   Elo. It is the only item whose failure mode is a `crash` termination, i.e. an
   outright loss. Ship tablebases first and keep a known-good fallback.
   If both ship, give the ponder thread **its own `Tablebase` instance**:
   python-chess guarantees thread safety only across *different board objects
   not modified during probing*, and a ponder thread mutates its board
   continuously.
10. **Book keyed on harvested ladder openings.** §9's "books are worthless" was
    correct given what was known, but the contract says finished games reveal the
    positions they were played from. A book keyed on the actual curated set is
    viable in a way a normal book is not.
11. **Bigger net (1024).** Only after item 7. On current data it is ~13
    samples/param and wants ~20M positions, which wants the loader rewrite.
12. **numba bitboard movegen.** Still the real ceiling at 33–36 µs a generation,
    still a multi-day correctness problem needing perft on a parallel branch,
    still wrong before a lock. The hardware does not make it safer.

### Stop doing

**The Stockfish node ladder.** §3 already records it swinging 3× at small
samples. The house bots on the live ladder carry public CCRL ratings and cannot
qualify — a grounded absolute scale, free, updating hourly. Read altitude off
those and spend the three hours on P0.

## 9. Principles that keep proving right

- **Nothing merges without SPRT.** A 60% score over 15 games is noise. And
  budget ~300 games at `elo1=5`: a large effect does not terminate the test
  early, because the hypotheses are what make it slow, not the effect size.
- **Profile before architecting.** The 33 us/1 us ratio killed a day of planned
  Zobrist work before it started. The 3.2 us/12.5 us ratio killed the assumption
  that a learned eval costs speed. The 268-bytes-per-entry measurement resized
  the TT.
- **Nodes to depth, not nodes per second.** bot4 halved its node rate and still
  searched deeper than bot1 on a fifth of the nodes. A speed metric alone scored
  the best build we have as a regression. *(New.)*
- **A high cutoff rate does not mean a correct search.** bot4 reported 90.6%
  first-move cutoffs while missing a forced mate. Ordering metrics measure
  ordering; only positions with a known answer measure correctness. *(New.)*
- **Pruning that returns before generating moves cannot be guarded at the move
  level.** Razoring, RFP and null move return early; every one of them is a place
  where a margin can silently overrule a forced sequence. Prefer prunes that skip
  a *move* over prunes that return a *score*. *(New.)*
- **Correlation, not error.** A search evaluation is a ranking function.
  Alpha-beta cares only about ordering. bot1's slope of 1.60 means it
  over-reports by 60% and it does not matter; bot3_nnue's better mae with worse
  correlation lost 280 Elo.
- **Measure on the distribution you will play.** A metric averaged over the wrong
  distribution is worse than no metric, because it reads as success.
- **Give the model what you already know.** Material as a fixed skip connection
  costs nothing and forces the capacity onto what is actually unknown.
- **A loss curve cannot see a distribution fault.** Validation loss moved the
  wrong way across the change that added the missing signal.
- **Keep a terrible opponent around.** It turns invisible bugs into impossible
  scores.
- **Test adversarially, at the boundaries.** The severe bugs only appeared under
  time pressure, at 50 ms budgets, and in forced mates — never in normal play.
- **Report the sample size next to the result, always.** "60%" and "60% ±184 Elo
  over 15 games" are different claims. A crossover from a 6-game rung moves by 3×
  on a re-run. *(Extended.)*
- **Fix the reference build before measuring against it.** bot1's colour bias
  belonged to both engines; patching it into `bot1fix` first meant bot4 was not
  credited for a fix it did not make. *(New.)*
- **Goodhart.** Never optimise Stockfish move-match rate or raw centipawn loss.
  Use Stockfish as an opponent, a labeller and an autopsy tool, not as a target.
- **The binding constraint was measurement volume, not ideas.** Every soft spot
  in this log traces to the same root: 30-game SPRTs that could not finish,
  6-game ladder rungs that swing 3×, a compromise dataset size, a metric-5 gap
  open across three iterations. None of those were analytical failures — they
  were all "we could not afford to run it." With the PC that excuse is gone, so
  **an unmeasured claim is now a choice, not a constraint.** Do not carry a
  provisional number forward again. *(New.)*
- **Design where it is cheap, measure where it is fast.** The sandbox is one
  core and cannot run volume; the PC is six cores and is the wrong hardware for
  fidelity. Splitting them is not a workaround, it is the correct assignment.
  *(New.)*

**Added 4 Sep, each paid for by this iteration.**

- **Blending two imperfectly-correlated evaluators beats either.** Worth ~240 Elo
  here. Do not treat it as a hedge to be removed once the network improves — it
  is the result. The network is better where it was trained and wrong outside it;
  the HCE covers exactly those gaps.
- **Gains do not add across baselines.** nobreak was +73 against bot4 and **+20**
  against bot5-blend. A change measured against a weaker reference is measuring
  partly the weakness. Re-measure against the current build before counting it.
- **A green gate can be testing the wrong thing.** Twice this iteration a build
  passed selftest while running the fallback evaluation. Always confirm the
  network loaded before believing any number.
- **`r` on quiet positions does not capture tactical soundness.** hidden 512 beat
  256 on the gate metric and failed a tactic 256 passed. Gate on both.
- **More capacity on the same distribution is not more coverage.** If the blend
  gap is large, the problem is what the data contains, not how many parameters
  fit it.
- **Cheap constants have paid more than architecture.** `NET_WEIGHT` 256→128 was
  240 Elo for one line; `STABLE_ITERATIONS` was 73; the network itself, with a
  data pipeline and a numba kernel behind it, delivered its value only *through*
  a blend constant. Sweep before you rewrite.
