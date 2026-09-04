# AI Chessathon — bot iteration

## This iteration

- **File:** `bot<N>_<style>.py`, `<N>` increments per family. NNUE builds are
  **directory builds**: `bots/bot<N>_<style>/bot<N>_<style>.py` plus exactly one
  `.npz` beside it, and a matching `WEIGHT_TARGETS` entry in `tools/agents.py`.
  Without that entry the weights land in the wrong place, the network silently
  does not load, and every measurement is of the fallback evaluation.
- **Reference build:** `bot5_nnue2` at `NET_WEIGHT = 128` with the hidden-512
  network. Every strength claim is against that, not bot4 and never a weak
  baseline. Weak baselines are a **failure gate**, not a score.
- **Idea:** read `v5_ITERATION_LOG.md` §8 and propose the next step from it.
- **Carry over:** `_budget`, `_sync`, `_record`, `get_move` and `_fallback`
  unchanged unless the iteration is *about* one of them. State explicitly what
  you are leaving alone.
- **Retraining:** almost certainly not needed. The kernel reads `hidden`, `qa`,
  `qb`, `scale` and `cp_scale` out of the `.npz`, so a network is a swappable
  asset. Retrain only for a feature-encoding change (mandatory — a mismatch is
  indistinguishable from a badly trained net), an architecture change, or better
  data. Say in step 1 which of those applies, if any.
- **Budget:** name it in step 1 — how much of the clock per node, how much import
  time, how many PC-hours of measurement.

---

## Step 1 — plan and ask, before any code

Read the iteration log first. Then produce this, and nothing else:

1. **Restate the idea** in three to five lines: what it buys over bot5, and where
   you expect it to be worse.
2. **The plan.** Components to write or change, in the order you would write
   them, one line of reasoning each. Say explicitly what you are leaving alone.
3. **The risks.** The two or three places most likely to flag, crash, or silently
   lose Elo, and the guard for each. Check them against §4's regression
   checklist — most entries there are pattern-level and recur in any rewrite.
   **Entries 11–14 are the ones that bit most recently and none of them raised an
   error**; they produced plausible wrong answers instead.
4. **The measurement plan.** Mandatory. Name, before writing a line:
   - the **one number** that confirms or kills the hypothesis,
   - which **gate** it clears first (§2 metric 2 is always first),
   - **where each run happens** — sandbox for fidelity, PC for volume,
   - the **wall-clock cost** of the whole plan on the PC,
   - and the **`--elo1` you are choosing and why**. At `elo1=5` a 73% result
     needs ~300 games; at 10 it is ~150, at 20 it is ~90. Five is a Fishtest
     setting for banking +2 Elo over months. Pick deliberately.
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
`Path(__file__).resolve().parent`, never the working directory, and a missing or
malformed weights file must fall back to the classical evaluation rather than
degrade — the worst case is that it plays as bot4, never that it plays badly.

**Anything the measurement needs to vary must be a module constant**, and an
environment override for A/B runs is fine as long as the shipped constant is the
default and the platform sets nothing. But note §4 bug #13: **the harness spawns
both agents as children of one `h2h` process, so an environment variable reaches
the opponent too.** Comparing two variants of the same build means baking the
constant into separate `versions/` directories — `tools/sweep.sh` does that.

After the file, at most five lines: what changed, and the one measurement that
settles it.

---

## Step 3 — the run plan

The session cannot run the measurement. Hand it over as a **staged, resumable
script**, not commands to hand-patch. Every long-running stage must be:

- **parallelised across at most 6 workers** — six games saturates six physical
  cores and SMT oversubscription distorts every timing number,
- **resumable**, checkpointing as it goes,
- **writing results into the repo**, not a scratch path,
- **stopping itself** where a human decision is needed, with the reason printed.

Fidelity checks stay in the session and run before the handover. There is no
point spending PC-hours on a build that fails `selftest.py`, and the correctness
gate is cheap. If the build has a network, **prove it loaded** before quoting any
number — a green selftest on the fallback evaluation is worse than a red one.

`tools/run.sh` and `tools/sweep.sh` already implement this shape. Extend them
rather than writing a third.

---

## Step 4 — after the results come back

Update `v5_ITERATION_LOG.md` per its own protocol: a ledger row in §1, the
regression checklist in §4, and move the item out of §8 with its measured result
**including if it failed**. A change that measured neutral is information and
must not be silently retried.

Report the sample size next to every number. "60%" and "60% over 156 games,
+22 Elo, interval −25 to +70" are different claims.

When two variants share a reference and an opening set, **compare them paired on
identical opening and colour slots**. That strips out opening variance and is far
sharper than reading two independent rows. It is what showed the blend's 85.6%
was real rather than an easy-openings artefact.

---

## Where compute runs

| | Assistant sandbox | Local PC (Ryzen 5 3600) | M4 Air |
|---|---|---|---|
| Cores | **1** @ 2.1 GHz Xeon | 6C/12T @ ~4.0 GHz | 4P + 6E, fanless |
| RAM | 3 GB | 16 GB | 16 GB |
| Job length | **~5 min/call, processes reaped** | unlimited | unlimited |
| Runs | **Fidelity** | **Volume** | Editing, short runs |

**Sandbox — fidelity.** 1 core, x86, ~2 GHz is a closer match to the competition
container than a 4 GHz Ryzen. nps, nodes-to-depth, import budget, clock
discipline, `selftest.py`, `kernelbench.py`, `tools/checknet.py`. A depth measured
on the Ryzen is a depth we will not reach in the Swiss, and `_budget` tuned
against it is tuned wrong.

**PC — volume.** Data generation, SPRT, ladder rungs, the 120 s + 0.5 s
validation. **LMDE natively for anything timed.** Data generation is
throughput-bound and does not care.

**The repo is the only channel between them.** `jmsyuen/chessathon`, public,
clonable from the sandbox. Commit measurement outputs; do not paste them.

---

## How this gets measured

Authority order. Items 1 and 2 gate everything else.

| # | Metric | Target | Runs on |
|---|---|---|---|
| 1 | Elo vs **bot5_nnue2 @ 128**, SPRT | pass before merge | PC |
| 2 | Failures/100 games (flag, crash, illegal, drawn-while-winning) | **exactly zero** | PC |
| 3 | nps and nodes-to-depth on the fixed suite | track per build | sandbox |
| 4 | First-move cutoff rate | >85% | sandbox |
| 5 | Average depth at the real 120 s + 0.5 s control | sanity check | PC |

Metric 4 is a ten-second proxy for evaluation quality when the search is held
fixed. Read it before spending an hour on games. bot5 blend 86.9%, bot4 89.1%,
bot5 pure net 83.5% — and the pure net is the one that fails a tactic.

---

## Things not to do

- No Stockfish, Lc0, Maia or any existing engine **inside** the submission,
  including a pip package that embeds one. Retroactive disqualification.
  Training on engine-annotated data is fine; the ban covers what ships and runs.
- Any shipped model must be one we trained. **Books and tablebases are permitted
  as shipped data** and `chess.polyglot` / `chess.syzygy` are in the base image.
- No obfuscation. `agent.py` is what a judge reads if games get flagged and what
  a finalist has to walk through.
- Do not edit `harness/`. It mirrors the platform.
- Do not tune against Stockfish. It blunders in different places than the field,
  so a change can help there and hurt in the Swiss. The house bots on the live
  ladder carry public CCRL ratings — read altitude off those instead.
- Do not run two SPRTs at once. Six workers is the whole machine.

---

## Style

Python 3.12, type-annotated, ruff and mypy strict clean against the repo's own
`pyproject.toml`. Comments explain **why**, not what — and where a constant came
from a measurement, say which one and over how many games.
