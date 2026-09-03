# AI Chessathon — bot iteration

## This iteration

- **File:** `bot<N>_<style>.py`, `<N>` increments per family
- **Reference build:** `bot4_ordering` — the strongest measured build. Every
  strength claim is against bot4, not bot1 and never a weak baseline. also possibly measure against the stockfish ladder, what are your thoughts?
- **Idea:** read `v4_ITERATION_LOG.md` §8 and propose the next step from it
- **Carry over:** stated per iteration in §8. For bot5: bot4's search whole,
  bot3's int16 kernel, `_budget` / `_sync` / `get_move` wrapper unchanged
- **Budget:** name it in step 1 — how much of the clock, how much import time,
  how many PC-hours of measurement it will cost

---

## Step 1 — plan and ask, before any code

Read the iteration log first. Then produce this, and nothing else:

1. **Restate the idea** in three to five lines: what it buys over bot4, and where
   you expect it to be worse.
2. **The plan.** Components to write or change, in the order you would write
   them, one line of reasoning each. Say explicitly what you are leaving alone.
3. **The risks.** The two or three places most likely to flag, crash, or silently
   lose Elo, and the guard for each. Check them against §4's regression
   checklist — most bugs there are pattern-level and recur in any rewrite.
4. **The measurement plan.** New and mandatory. Name, before writing a line:
   - the **one number** that confirms or kills the hypothesis,
   - which **gate** it has to clear first (§2 metric 2 is always first),
   - **where each run happens** — sandbox for fidelity, PC for volume,
   - the **wall-clock cost** of the whole plan on the PC.
   If the plan cannot be measured in the time left before the lock, say so now
   rather than after the code is written.
5. **Questions.** Three to eight. Only things that would change the design. Where
   a question has an obvious default, state your default so I can reply
   "defaults" and move on.

Then stop.

Good question territory: what this is measured against and at what control, how
much clock it may spend, whether module state carries across moves, what happens
in the endgame, what should win when the idea and safety disagree. Bad question
territory: naming, formatting, type hints.

---

## Step 2 — the file

Output **one Python file and nothing else**, named per the slot above. No
preamble, no patch, no second file.

It must be a drop-in `agent.py`: no relative imports, no imports from sibling bot
files, no `__main__` requirement. It stands alone. Weights load from
`Path(__file__).resolve().parent`, never the working directory.

After the file, at most five lines: what changed, and the one measurement that
settles it.

---

## Step 3 — the run plan

New step. The session cannot run the measurement any more, so it has to hand it
over cleanly. Output a short block of shell commands, in order, each with its
expected wall-clock time. Every long-running command must be:

- **parallelised across at most 6 workers** (see §3 of the log — six games
  saturates six physical cores; SMT oversubscription distorts timing),
- **resumable**, writing checkpoints as it goes,
- **writing results into the repo**, not to a scratch path.

Fidelity checks stay in the session and run before the handover — there is no
point spending PC-hours on a build that fails `selftest.py`.

---

## Step 4 — after the results come back

Update `v4_ITERATION_LOG.md` per its own protocol: a ledger row in §1, the
regression checklist in §4, and move the item out of §8 with its measured result
**including if it failed**. A change that measured neutral is information and
must not be silently retried.

Report the sample size next to every number. "60%" and "60% ±184 Elo over 15
games" are different claims.

---

## Where compute runs

Three machines, and the split is an assignment, not a workaround.

| | Assistant sandbox | Local PC (Ryzen 5 3600) | M4 Air |
|---|---|---|---|
| Cores | **1** @ 2.1 GHz Xeon | 6C/12T @ ~4.0 GHz | 4P + 6E, fanless |
| RAM | 3 GB | 16 GB | 16 GB |
| Job length | **~5 min/call, processes reaped between calls** | unlimited | unlimited |
| Runs | **Fidelity** | **Volume** | Editing, short runs |

**Sandbox — fidelity.** 1 core, x86, ~2 GHz is a closer match to the competition
container (1 core, 2 GB) than a 4 GHz Ryzen. nps, nodes-to-depth, import budget,
clock discipline under 50 ms pressure, `selftest.py`, `kernelbench.py`,
`v3_checknet.py`, `v3_evalcmp.py`. A depth measured on the Ryzen is a depth we
will not reach in the Swiss, and `_budget` tuned against it is tuned wrong.

**PC — volume.** Data generation, SPRT, ladder rungs, the 120 s + 0.5 s
validation. Use LMDE natively for anything timed; hypervisor jitter contaminates
wall-clock measurement and can produce flags that look like time-management bugs.
WSL2 is fine for data generation, which is throughput-bound.

**The GPU is close to irrelevant.** Training is 6 s/epoch on one core and the
numpy trainer already works. The bottleneck is Stockfish labelling, which is
CPU-bound. Do not rewrite the trainer for CUDA.

**The repo is the only channel between them.** `jmsyuen/chessathon`, public,
clonable from the sandbox. Commit measurement outputs into it rather than pasting
them — §0 of the log explains what pasting has already cost.

---

## How this gets measured

Authority order. Items 1 and 2 gate everything else.

| # | Metric | Target | Runs on |
|---|---|---|---|
| 1 | Elo vs previous build, SPRT `elo0=0, elo1=5, α=β=0.05` | pass before merge | PC |
| 2 | Failures/100 games (flag, crash, illegal, drawn-while-winning) | **exactly zero** | PC |
| 3 | nps on the fixed 12-position suite | track per build | sandbox |
| 4 | First-move cutoff rate | >85% | sandbox |
| 5 | Average depth at real 120 s + 0.5 s | sanity check | PC |

**Nothing merges without a completed SPRT.** This used to be aspirational because
130 games was unaffordable on one core — bot4 shipped as the standing
recommendation on an undecided LLR. At ~40 minutes for a full run on the PC that
excuse is gone. Budget ~130 games; a large effect does not terminate early.

**Track nodes-to-depth, not nodes per second.** bot4 runs at 12.9k nps against
bot1's 27.0k and is decisively stronger — it reaches equal or greater depth on
5.5× fewer nodes. Metric 3 read alone scores the best build we have as a
regression.

**Weak baselines are a gate, not a score.** random/greedy/minimax carry no signal
about the field. Keep them only for surfacing promotion / en-passant /
no-legal-move crashes in volume, where they are far more efficient than
Stockfish. Metric 2 stays non-negotiable at exactly zero.

**Stockfish is altitude only.** `baselines/stockfish` + `tools/ladder.py` give a
50% crossover in nodes/move. Never tune against it — it blunders in different
places than the field, so a change can help against it and hurt in the Swiss.
Do not quote a crossover from fewer than **40 games per rung**; at 6–8 games the
estimate swings by 3×. The house bots on the live ladder show public CCRL ratings
and are a better absolute scale.

---

## Standing context

### The contract

- `agent.py` at the **root** of the zip. The platform does `import agent`.
- `get_move(fen: str, time_left_ms: int) -> str`, returning UCI (`e2e4`, `e7e8q`).
- Colour is the side to move in the FEN. **No move history is given.** The first
  FEN received is the game's starting position, and repetition plus fifty-move
  counts begin *there*, not from the standard start.
- 120 s + 0.5 s per side, wall clock. 60 s import budget before the clock starts.
- Process starts fresh per game, stays alive between your own moves. Module state
  survives to your next move in the same game, never to the next game.
- Pondering is allowed — the process keeps its core after `get_move` returns.
- One core, 2 GB RAM, no network, no GPU. Read-only FS plus `/tmp`, which starts
  empty every game and is deleted with it, so it is never a cache between games.
- 50 MB unzipped. Six uploads/day; the latest that passed validation plays.
- Weights ship as a second artifact: `agent.py` at the root plus `weights/*.npz`.
  `harness/package.py` includes `weights/` via `DEFAULT_INCLUDES`.

### The environment

Only these exist: **torch 2.13.0+cpu, numpy 2.5.2, python-chess 1.11.2,
onnxruntime 1.29.0, numba 0.67.0**. `requirements.txt` is ignored.

Native binaries are rejected and there is no compiler, so Cython is out. numba is
the only speed path, `cache=False` is mandatory, and every jitted function must
be warmed at import with the argument types the real calls use.

**A jitted learned eval is cheaper than a Python piece-square eval here** — 3.2 µs
against 12.5 µs. Assume the numba path is free and spend the budget on quality.

### Harness behaviour that is not on the docs page

- `referee.py` calls `board.outcome(claim_draw=True)`. Threefold and fifty-move
  are claimed **automatically against you**. Reconstruct history from FENs.
- The 300-ply cap adjudicates on **pure material** — P1 N3 B3 R5 Q9, king 0.
  Exploitable in both directions; bot4 plays for it from ply 240.
- The watchdog's 500 ms grace is **not usable slack** — the referee flags the
  instant the clock goes negative.
- `runner.py` does not wrap `get_move`. Any uncaught exception loses the game.
- The zip is first on `sys.path`. Never name a file `chess.py`, `types.py`.
- `print` is safe; the runner points fd 1 at stderr before importing.
- stdout cap 4096 bytes; over-cap counts as an illegal move. Separately, the
  validation log echoes stdout/stderr back up to 8 KB.
- `harness/package.py` zips **every `*.py` at the repo root.** Variant files left
  there ship inside the submission. This is how an engine wrapper becomes a
  disqualification — keep them in subdirectories and check the zip before upload.

### Free losses, in order of frequency

1. **Flagging.** Budget from the clock handed in, check it *inside* the search,
   always hold a best-move-so-far, leave real overhead for the IPC hop.
2. **Crashing** on an edge case: no legal moves, promotion, en passant, malformed
   FEN. Everything in `get_move` goes inside a try with a legal-move fallback.
3. **Illegal or malformed output.**
4. Blowing the 60 s import budget. bot3's numba import alone costs 14 s.
5. Writing outside `/tmp`.
6. More threads than cores. `torch.set_num_threads(1)`.

### Local compatibility surface

`tools/selftest.py` reaches into the module directly. Expose these, or propose
the selftest change in step 1 rather than dropping the check:

| Name | Shape |
|---|---|
| `get_move` | `(fen: str, time_left_ms: int) -> str` |
| `_budget` | `(time_left_ms: int, plies_played: int) -> tuple[float, float]` |
| `_board` | `chess.Board \| None`, resettable |
| `_history_keys` | `list`, resettable |
| `_tt` | `dict`, has `.clear()` |
| `INCREMENT_MS`, `OVERHEAD_MS` | `int` |

### Things not to do

- No Stockfish, Lc0, Maia or any existing engine inside the submission, including
  a pip package that embeds one. Retroactive disqualification. Training on
  engine-annotated data is fine — the ban covers what ships and runs in the zip.
- Any shipped model must be one we trained.
- No obfuscation. A judge reads the source; a finalist walks through how it was
  built.
- Do not edit `harness/`. It mirrors the platform.
- No opening book. Rated games start from unpublished curated near-level
  positions, so a book keyed on move one is out of book immediately. Those
  positions **are** recoverable from finished ladder games — harvest them
  instead.

### Style

Python 3.12, type-annotated, ruff and mypy strict clean. `agent.py` is what a
judge reads if games get flagged and what we explain at the final, so it stays
readable. Comments explain why, not what.
