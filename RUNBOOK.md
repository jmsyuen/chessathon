# Compute runbook

How to run measurement on the PC, and how to get the results back into a session.

The short version: **the sandbox is one slow core and cannot run volume; the PC
is six fast cores and is the wrong hardware for fidelity.** Split them on
purpose, and let the repo carry results between them.

| | Assistant sandbox | PC (Ryzen 5 3600) | M4 Air |
|---|---|---|---|
| Cores | 1 @ 2.1 GHz | 6C/12T @ ~4 GHz | 4P + 6E, fanless |
| Job length | ~5 min, processes reaped | unlimited | unlimited |
| Runs | **fidelity** | **volume** | editing, short runs |

Fidelity means nps, nodes-to-depth, import budget, clock discipline — anything
where the *hardware* is the measurement. A Ryzen core is roughly 1.8× the
competition's, so a depth measured there is a depth you will not reach on the
day, and a `_budget` tuned against it is tuned wrong.

Volume means games and data. Nothing about a 300-game SPRT cares which core it
ran on, only how many of them there were.

---

## 0. One-time setup

Use **LMDE natively for anything timed**. SPRT and the 120 s validation are
measuring wall-clock behaviour, so hypervisor jitter contaminates the thing being
measured and can produce flags that look like time-management bugs. WSL2 is fine
for data generation, which is throughput-bound and does not care.

```bash
# Python 3.12 specifically — Debian 12 ships 3.11, Debian 13 ships 3.13
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12

sudo apt-get install -y stockfish        # sparring partner and data labeller
git clone https://github.com/jmsyuen/chessathon.git && cd chessathon
uv sync
```

Then, every session, before anything else:

```bash
uv run python -m tools.agents
```

This materialises `versions/<build>/agent.py` and `baselines/<name>/agent.py`
from the flat files in `bots/`. The harness loads agents as *directories*; the
repo stores them as files. Without this step every match tool points at a path
that does not exist. `--check` verifies without writing.

Confirm the environment before trusting a number from it:

```bash
uv run python -c "import chess, numba, numpy; print(chess.__version__, numba.__version__, numpy.__version__)"
# expect 1.11.2 / 0.67.0 / 2.5.2 — the platform's versions
nproc && stockfish bench 2>&1 | tail -2
```

---

## 1. The gate, always first

```bash
uv run python -m tools.selftest 2>&1 | tee results/bot5.selftest.txt      # ~4 min
uv run python -m tools.h2h --agent versions/bot5 --opponent baselines/random \
    --games 100 --workers 6 --base-ms 1000 --increment-ms 50 \
    --out results/bot5_gate.json                                          # ~5 min
```

The gate is **failure count, not score**. Random opponents are worthless as a
strength measurement and excellent at surfacing promotion, en-passant and
no-legal-move crashes, because random play wanders into strange positions far
faster than a real engine does.

**Zero failures or stop.** Hours of game time spent on a build that crashes one
game in fifty is hours wasted, and the crash will not be visible in the Elo.

---

## 2. SPRT — the merge decision

```bash
uv run python -m tools.h2h \
    --agent versions/bot5 --opponent versions/bot4_ordering \
    --games 300 --workers 6 --base-ms 8000 --increment-ms 500 \
    --out results/bot5_vs_bot4.json                                       # ~1 h
```

`--workers 6` shards by *opening pair*, so each worker stays colour-balanced and
no two workers play the same opening. The parent pools every shard and stops the
moment the **pooled** LLR is decisive — a per-shard LLR means nothing.

Resumable: re-run the same command to add games. Interrupt with Ctrl-C and
nothing is lost; every game is checkpointed as it finishes.

Read the running result at any time, from another terminal:

```bash
uv run python -m tools.h2h --report --out results/bot5_vs_bot4.json
```

**Cap workers at 6, not 12.** A game runs two agent processes but only one thinks
at a time, so six games saturates six physical cores. SMT oversubscription
distorts every timing number in the run.

### Choosing the hypotheses

`--elo1` decides how long this takes, and the default is slow on purpose:

| `--elo1` | Games to decide a 73% result | What it costs |
|---|---|---|
| 5 (default) | ~300 | ~1 h at 8 s + 0.5 s |
| 20 | ~90 | rejects a genuine +10 Elo change |

elo1=5 is a Fishtest setting, built for banking +2 Elo tweaks over months. With
eight days left you are hunting architectural wins, and a +10 Elo change will not
decide your Swiss placing. Both are now affordable, so pick deliberately rather
than inheriting the default.

---

## 3. Ladder — altitude, not tuning

```bash
# rungs are independent, so run one process each and get 5× for free
for n in 1000 3000 10000 30000; do
  SPAR_ENGINE=$(which stockfish) SPAR_LEVEL="nodes:$n" \
  uv run python -m tools.ladder --agent versions/bot5 --games 40 \
      --rungs "nodes:$n" --base-ms 120000 --increment-ms 500 \
      > results/bot5.rung$n.ladder.txt 2>&1 &
done; wait                                                                # ~3 h
```

**40 games per rung minimum.** At 6–8 the estimate swings by 3× on re-run — the
~340-node and ~100-node crossovers recorded for bot1 are the same engine measured
twice, and neither is trustworthy.

The bar, at the real control: clear `nodes:3000`, target `nodes:10000`, stretch
`nodes:30000`. Never tune against Stockfish — it blunders in different places
than the field, so a change can help here and hurt in the Swiss.

---

## 4. Data generation, for NNUE work

```bash
for s in 0 1 2 3 4 5; do
  uv run python -m bots.bot3_nnue.v3_gendata --seed $s --minutes 40 \
      --out data/train.s$s.csv &
done; wait
cat data/train.s*.csv > data/train.csv                       # ~4 min per 1M
```

Roughly 4,000 positions/second across six cores. A million takes four minutes,
ten million forty, fifty million one overnight run.

**Training length is not the bottleneck** — it is six seconds an epoch, four
minutes for a full run, on one core. Do not reach for the GPU. What the compute
buys is the freedom to *reject* aggressively: generate far more than you need and
throw away everything that does not flatten the material distribution. bot3
failed on 82% lopsided positions, not on undertraining.

---

## 5. Bringing results back

The repo is the only channel. Commit the results; do not paste raw JSON.

```bash
uv run python -m tools.collect          # writes results/SUMMARY.md
git add results/ && git commit -m "bot5: SPRT vs bot4, 300 games" && git push
```

Then in a session, either:

- **say "pull the repo and read `results/SUMMARY.md`"** — the sandbox can clone
  `codeload.github.com` directly, and this is the better route because it carries
  the code alongside the numbers, or
- run `uv run python -m tools.collect --paste` and paste the output, when a clone
  is inconvenient.

`SUMMARY.md` reports the failure gate first and separately, because a run
containing a crash has no strength information worth reading yet. It also records
the machine and commit — a depth from a 6-core desktop and one from a single slow
core are different numbers, and nobody remembers which was which a week later.

**Do not commit:** the training CSV, the parsed cache, float checkpoints, or the
Stockfish binary. `.gitignore` covers these. The binary especially — everything
that ships is checked, retroactively, and the penalty is disqualification.

---

## 6. Before any upload

```bash
uv run python -m tools.agents --ship bot4_ordering   # point root agent.py at a real build
uv run python -m tools.selftest
uv run python -m harness.package
unzip -l submission.zip | grep -iE 'spar|stockfish|engine' && echo "STOP — DO NOT UPLOAD"
```

`harness/package.py` zips **every `*.py` at the repo root**, and the sparring
agent carries an import-time guard against being run from there — but
`package.py` zips without importing, so the guard cannot fire. The `unzip` check
is the one that actually protects you.

Check what you are shipping is what you think you are shipping. The root
`agent.py` was still the starter's 29-line random mover well after bot4 existed.

---

## What runs where, in one table

| Task | Machine | Rough cost |
|---|---|---|
| `selftest.py`, `kernelbench.py`, `v3_checknet.py` | sandbox | minutes |
| nps, nodes-to-depth, clock discipline | sandbox | minutes |
| Failure gate, 100 games | PC | ~5 min |
| SPRT, 300 games at 8 s + 0.5 s | PC | ~1 h |
| Ladder, 4 rungs × 40 games at 120 s | PC | ~3 h |
| 120 s + 0.5 s validation, 130 games | PC | ~3.5 h, overnight |
| Data generation, 10M positions | PC | ~40 min |
| Training | anywhere | ~4 min |
