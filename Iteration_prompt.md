
next variation: 
then one of my style
different Strategies in different stages of the game, opening, midgame, endgame
try to predict using history of past moves, keep a record 


Done:
Understand how the engine works, and our options
Iteration prompt to quickly iterate strategy versions and reduce repeated code
Made benchmark to test our iterations - ladder of stockfish difficulty based on nodes and depth to simulate constraints, and test on previous iterations
Iteration log to carry over for next iterations

# Bot iteration prompt
# AI Chessathon — bot iteration

## This iteration

- **File:** pattern is `bot<N>_<style>.py`, `<N>` increments per family
- **Idea:** read iteration log to suggest next steps, from everything bot2 and before.
- **Hypothesis:** use the modified stockfish engine ladder and previous iterations to test how our bot performs.
- **Budget:** as much as you can
- **Carry over:** bot1's `_budget`, `_sync` and `get_move` wrapper unchanged

## Step 1 — plan and ask, before any code

Do not write the file yet. Produce this first:

  Read the current iteration log in the project files to see where our last iterations could improve. Then:

1. **Restate the idea** in three to five lines: what you think it buys over the current best
   bot, and where you expect it to be worse.
2. **The plan.** The components you will write or change, in the order you would write them,
   one line of reasoning each. Say explicitly which parts of the previous bot you are leaving
   alone.
3. **The risks.** The two or three places most likely to flag, crash, or silently lose Elo,
   and the guard for each.
4. **Questions.** Between three and eight. Ask only about things that would change the design,
   not preferences you could reasonably pick yourself. Where a question has an obvious default,
   state your default next to it so I can reply "defaults" and move on.

Then stop. Do not begin the file until I answer.

Good question territory: what the change is being measured against and at what time control,
how much of the clock this is allowed to spend, whether module state may carry across moves,
what happens to the idea in the endgame, and what I want to happen when the idea and safety
disagree. Bad question territory: naming, formatting, whether to add type hints.

## Step 2 — the file

Once I have answered, output **one Python file and nothing else**, named per the slot above.
No prose preamble, no separate patch, no second file. After the file, at most five lines:
what changed, and the one measurement that would confirm or kill the hypothesis.

The file must be a drop-in `agent.py`. It gets copied to `agent.py` to run against the harness,
so: no relative imports, no imports from sibling bot files, no `__main__` requirement. It has to
stand alone.

After testing, update the iteration log in the project files.
---

## Standing context

### The contract

- `agent.py` at the **root** of the zip. The platform does `import agent`.
- `get_move(fen: str, time_left_ms: int) -> str`, returning UCI (`e2e4`, `e7e8q`).
- Colour is the side to move in the FEN. There is no other input. **No move history is given.**
- 120 s + 0.5 s increment per side, wall clock. 60 s import budget before the clock starts.
- The process starts fresh per game and stays alive between your own moves. Module state
  survives to your next move in the same game, never to the next game. Pondering is allowed.
- One dedicated core, 2 GB RAM, no network, no GPU. Read-only FS plus 256 MB at `/tmp`.
- 50 MB unzipped. Six uploads per team per day; the latest that passed validation plays.

### The environment

Only these are installed and nothing else can be: **torch 2.13.0+cpu, numpy 2.5.2,
python-chess 1.11.2, onnxruntime 1.29.0, numba 0.67.0**. `requirements.txt` is ignored. An
import outside that stack works locally and crashes on the platform.

Native binaries are rejected and there is no compiler in the image, so Cython is out. numba is
the only speed path. Warm every jitted function at import, with the argument types the real
calls use, or the compile lands on your clock.

### Harness behaviour that is not on the docs page

These came from reading the starter repo's `harness/`. They are not obvious and each one has
cost people games:

- `referee.py` calls `board.outcome(claim_draw=True)`. Threefold and fifty-move draws are
  claimed **automatically against you**. You can hand back a won game by shuffling. Since you
  only get a FEN per move, you have to reconstruct position history yourself.
- The 300-ply cap adjudicates on **pure material** — P1 N3 B3 R5 Q9, king 0, no positional
  terms. Exploitable in both directions.
- `sandbox.py`'s watchdog kills at `time_left_ms + 500 ms`, but `referee.py` separately deducts
  wall-clock elapsed and flags the instant the clock goes negative. **The 500 ms grace is not
  usable slack.**
- `runner.py` does not wrap `get_move` in try/except. Any uncaught exception kills the process
  and loses the game.
- Your zip is first on `sys.path`. A file named `chess.py`, `types.py` or `random.py` shadows
  the real module and fails in a way that looks unrelated.
- `print` is safe. The runner points fd 1 at stderr before importing the agent.
- `harness/package.py` zips **every `*.py` at the repo root**. Variant files left at the root
  ship inside the submission. Keep them in a subdirectory.

### Free losses, in order of how often they happen

1. **Flagging.** Budget from the clock you were handed, check it *inside* the search, and always
   hold a best-move-so-far. Leave real overhead for the IPC hop.
2. **Crashing** on an edge case: no legal moves, promotion, en passant, a malformed FEN.
   Everything in `get_move` goes inside a try, with a fallback that returns a legal move.
3. **Illegal or malformed output.** A reply over 4 KB counts as illegal.
4. Blowing the 60 s import budget loading weights.
5. Writing outside `/tmp`.
6. More threads than cores. `torch.set_num_threads(1)`.

### Local compatibility surface

`tools/selftest.py` reaches into the module directly. Expose these names, or the selftest needs
editing alongside the bot:

| Name | Shape |
|---|---|
| `get_move` | `(fen: str, time_left_ms: int) -> str` |
| `_budget` | `(time_left_ms: int, plies_played: int) -> tuple[float, float]`, returning `(soft_ms, hard_ms)` |
| `_board` | `chess.Board \| None`, resettable |
| `_history_keys` | `list`, resettable |
| `_tt` | `dict`, has `.clear()` |
| `INCREMENT_MS` | `int` |
| `OVERHEAD_MS` | `int` |

If the design genuinely cannot expose one of these, say so in step 1 and propose the selftest
change, rather than dropping the check.

### How this gets measured

`tools/selftest.py` is the correctness gate — tactics, edge cases, malformed input, endgame
conversion, repetition tracking, clock discipline. It runs in under a minute. It has to pass
before strength is worth measuring at all.

`tools/bench.py` is the strength gate. It alternates colours over near-level opening positions
and prints an Elo estimate with a 95% interval. The opponent is the **previous bot**, not a
baseline. A 3% gain is invisible in 20 games; assume hundreds. Bench defaults to 10 s + 100 ms
while the real control is 120 s + 0.5 s, so anything tuned to a fast control needs a sanity
check at the real one before it goes up.

The current reference points: random scores 10% against greedy, greedy scores 0/6 against
minimax, numba-minimax barely beats minimax. `bot1_baseline` is the classical engine to beat.

### Things not to do

- No Stockfish, Lc0, Maia, or any existing engine inside the submission, including a pip
  package that embeds one. Instant disqualification, checked retroactively. Training on
  engine-annotated data is fine; the ban covers what ships and runs in the zip.
- Any shipped model must be one we trained.
- No obfuscation. What ships has to be source a judge can read, and a finalist has to walk
  through how it was built.
- Do not edit `harness/`. It mirrors the platform.
- An opening book is close to worthless. Rated games start from unpublished curated near-level
  positions, so a book keyed on move one is out of book immediately.

### Style

Python 3.12, type-annotated, ruff and mypy strict clean. `agent.py` is what a judge reads if
games get flagged and what we have to explain at the final, so it stays readable. Comments
explain why, not what.
