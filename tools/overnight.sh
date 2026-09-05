#!/usr/bin/env bash
# tools/overnight.sh — the whole decision battery, one command, resumable.
#
#   chmod +x tools/overnight.sh
#   DRY=1 bash tools/overnight.sh                                 # ~20 min sanity pass
#   nohup bash tools/overnight.sh > results/overnight.log 2>&1 &  # the real run
#   tail -f results/overnight.log
#
# Resumable per game, so a lid close or a reboot costs nothing — re-run the same
# command. Run on LMDE natively, not WSL2: hypervisor jitter contaminates
# wall-clock measurement and can manufacture time-management bugs that aren't there.
#
# Roughly 4.5-6 h at WORKERS=6. Nothing needs your attention until VERDICT.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
[ -f tools/agents.py ] || { echo "not the repo root (no tools/agents.py)"; exit 1; }

DRY="${DRY:-0}"
WORKERS="${WORKERS:-6}"                        # 6 physical cores. Do NOT raise to 12;
                                               # SMT oversubscription distorts every timing.
CONTROL="--base-ms 8000 --increment-ms 500"    # matches every run already in SUMMARY.md
SPRT="--elo0 0 --elo1 20"                      # NOT 5: SUMMARY.md has a +79 Elo result
                                               # over 427 games still reading "undecided".
mkdir -p results
say() { printf '\n\033[1m=== %s ===\033[0m %s\n' "$1" "$(date +%H:%M:%S)"; }
FAILED=0

# ---------------------------------------------------------------- interpreter
# This repo is uv-managed: pyproject.toml pins chess/numpy/numba/torch, and every
# Makefile target runs `uv run python`. A bare `python3` is the system interpreter,
# which has none of that — that is the "No module named chess" from a fresh shell.
# (Stockfish is unrelated; it is a binary for the sparring ladder and this battery
# never invokes it.)
if command -v uv >/dev/null 2>&1 && [ -f uv.lock ]; then
  # --no-sync matters twice over. It stops uv re-resolving the lock on every one
  # of the thousands of interpreter spawns an h2h run makes, and it stops a
  # single unreachable wheel (torch, which nothing here imports) from turning
  # every subsequent call into a failure. The sync happens once, below.
  PYRUN="uv run --no-sync python"
  uv run --no-sync python -c "" 2>/dev/null || PYRUN="uv run python"
elif [ -x .venv/bin/python ]; then
  PYRUN=".venv/bin/python"
else
  PYRUN="python3"
fi

run_match() { if [ "$DRY" = "1" ]; then echo "  [DRY] skipped"; else "$@" 2>&1 | tail -9; fi; }

# ---------------------------------------------------------------- 0. preflight
say "0/6 PREFLIGHT"
echo "  interpreter: $PYRUN"
if command -v uv >/dev/null 2>&1 && [ -f uv.lock ]; then
  echo "  uv sync (idempotent; slow only the first time — torch is ~200 MB)"
  # Deliberately non-fatal. torch is a TRAINING dependency: NNUE inference here is
  # numpy + a numba kernel, and nothing in this battery imports torch. A flaky
  # pytorch CDN or a 403 on the wheel must not cost you a night's compute, so the
  # real gate is the import check below, not sync's exit code.
  if ! uv sync 2>&1 | tail -2; then :; fi
  if ! $PYRUN -c "import chess, numba, numpy" >/dev/null 2>&1; then
    echo "  sync incomplete. Retrying without torch, which this battery never imports."
    uv sync --no-install-package torch 2>&1 | tail -2
  fi
fi
$PYRUN -c "import chess,numba,numpy,sys;print('  py',sys.version.split()[0],'chess',chess.__version__,'numba',numba.__version__)" || {
  echo "  ABORT: dependencies missing."
  echo "  If uv is not on PATH after a restart:  source \$HOME/.local/bin/env"
  echo "  If uv is not installed:  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
}
$PYRUN -m tools.agents | tail -3

# The weight-path trap. tools/agents.py's build() placed an unlisted NNUE build's
# weights flat while ship() placed them under weights/, so such a build loaded its
# net when shipped and silently fell back to the CLASSICAL evaluation when
# benchmarked. No crash, no selftest failure — it just measures as a different
# engine. bot5.1 hit exactly this. Never run the battery without asserting it.
say "0b/6 NET LOAD ASSERTION"
for B in bot5_nnue2 bot5.1_nnue2; do
  $PYRUN - "versions/$B" "$B" <<'PY' || {
import sys
sys.path.insert(0, sys.argv[1])
import agent as A
ok = A._forward is not None
print(f"  {sys.argv[2]:<14} net={ok}  hidden={getattr(A, '_HIDDEN', 0)}"
      f"  NET_WEIGHT={A.NET_WEIGHT}  STABLE_ITERATIONS={A.STABLE_ITERATIONS}")
sys.exit(0 if ok else 1)
PY
    echo "  ABORT: $B is not loading its network. Patch tools/agents.py:"
    echo "    sed -i 's|WEIGHT_TARGETS.get(name, weight.name)|WEIGHT_TARGETS.get(name, f\"weights/{weight.name}\")|' tools/agents.py"
    echo "    rm -rf versions/$B && $PYRUN -m tools.agents"
    exit 1
  }
done

# ------------------------------------------------------- 1. correctness gate
say "1/6 CORRECTNESS GATE"
for B in bot4_nobreak bot5_nnue2 bot5.1_nnue2 bot6B_numbas; do
  $PYRUN -m tools.agents --ship "$B" >/dev/null
  OUT=$(timeout 1800 $PYRUN -m tools.selftest 2>&1 | tail -3)
  echo "  $B: $(echo "$OUT" | tr '\n' ' ')"
  echo "$OUT" | grep -q "all checks passed" || { echo "  *** GATE FAIL: $B"; FAILED=1; }
done

$PYRUN -m tools.agents --ship bot6B_numbas >/dev/null
$PYRUN - <<'PY' || FAILED=1
import agent as A
CASES=[("start","rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",[20,400,8902,197281,4865609]),
 ("kiwipete","r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",[48,2039,97862,4085603]),
 ("pos3","8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",[14,191,2812,43238,674624]),
 ("pos4","r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",[6,264,9467,422333]),
 ("pos5","rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",[44,1486,62379,2103487]),
 ("pos6","r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",[46,2079,89890,3894594])]
ok=True
for n,f,refs in CASES:
    b=A.BitboardBoard(f)
    for d,r in enumerate(refs,1):
        got=A.perft(b._bb,b._st,b._key,b._undo,b._scratch,b._pool,d,0)
        if got!=r:
            ok=False; print(f"  PERFT FAIL {n} d{d}: {got} != {r}")
print("  bot6B perft:", "ALL PASS" if ok else "FAILURES")
raise SystemExit(0 if ok else 1)
PY

# ------------------------------------------------ 2. failure gate at scale
# Before the matches: a crash outranks any Elo number, and this is the cheapest
# block. --ply-cap 300 is the whole point — bot6's heap corruption lived ONLY in
# long games and six games found it by luck.
say "2/6 FAILURE GATE (400 games, ply-cap 300)"
run_match $PYRUN -m tools.h2h --agent versions/bot6B_numbas --opponent baselines/greedy \
  --games 400 --workers "$WORKERS" --base-ms 3000 --increment-ms 100 --ply-cap 300 \
  --out results/bot6_gate.json

# ------------------------------------------------------------- 3-5. matches
say "3/6 MATCH A — bot5.1 vs bot5   (does the break merge pay?)"
run_match $PYRUN -m tools.h2h --agent versions/bot5.1_nnue2 --opponent versions/bot5_nnue2 \
  --games 200 --workers "$WORKERS" $CONTROL --ply-cap 200 $SPRT \
  --out results/bot5_1_vs_bot5.json

say "4/6 MATCH B — bot6 vs bot5.1   (THE DECISION)"
run_match $PYRUN -m tools.h2h --agent versions/bot6B_numbas --opponent versions/bot5.1_nnue2 \
  --games 400 --workers "$WORKERS" $CONTROL --ply-cap 200 $SPRT \
  --out results/bot6_vs_bot5_1.json

say "5/6 MATCH C — bot6 vs bot4_nobreak   (kernel or eval?)"
run_match $PYRUN -m tools.h2h --agent versions/bot6B_numbas --opponent versions/bot4_nobreak \
  --games 200 --workers "$WORKERS" $CONTROL --ply-cap 200 $SPRT \
  --out results/bot6_vs_bot4_nobreak.json

# ------------------------------------- 6. metric 5 at the real control
say "6/6 METRIC 5 — 24 games at the REAL 120s + 0.5s"
run_match $PYRUN -m tools.h2h --agent versions/bot6B_numbas --opponent versions/bot5.1_nnue2 \
  --games 24 --workers "$WORKERS" --base-ms 120000 --increment-ms 500 --ply-cap 300 \
  --out results/bot6_vs_bot5_1_realcontrol.json

# ------------------------------------------------------------------ summary
say "COLLECT"
$PYRUN -m tools.collect 2>&1 | tail -5
echo
printf '\033[1m=== VERDICT ===\033[0m\n'
$PYRUN - <<'PY'
import json, glob
FAILS = {"crash", "illegal", "flag", "init", "both_failed"}

def read(stem):
    """Pool a run across shards. h2h writes <stem>.json for one process and
    <stem>.shardN.json when parallel; globbing only one reads a 6-worker run
    as zero games."""
    games = []
    for f in sorted(set(glob.glob(f"results/{stem}.json")
                        + glob.glob(f"results/{stem}.shard*.json"))):
        try:
            games += json.load(open(f)).get("games", [])
        except Exception:
            pass
    return games

ROWS = [("MATCH A  bot5.1 vs bot5",  "bot5_1_vs_bot5"),
        ("MATCH B  bot6 vs bot5.1",  "bot6_vs_bot5_1"),
        ("MATCH C  bot6 vs bot4_nb", "bot6_vs_bot4_nobreak"),
        ("REAL     bot6 vs bot5.1",  "bot6_vs_bot5_1_realcontrol"),
        ("GATE     bot6 vs greedy",  "bot6_gate")]

v = {}
for label, stem in ROWS:
    g = read(stem)
    if not g:
        print(f"{label:26s}  NOT RUN")
        continue
    fails = sum(1 for x in g if x.get("termination") in FAILS)
    win   = sum(1 for x in g if x.get("outcome") == "win")
    draw  = sum(1 for x in g if x.get("outcome") == "draw")
    score = (win + 0.5 * draw) / len(g)
    v[stem] = (score, fails, len(g))
    print(f"{label:26s}  n={len(g):4d}  +{win} ={draw} -{len(g)-win-draw}"
          f"  score={score:5.1%}  failures={fails}"
          + ("   <<< FAILURES" if fails else ""))

print()
gate, a, b = v.get("bot6_gate"), v.get("bot5_1_vs_bot5"), v.get("bot6_vs_bot5_1")
if gate is None or b is None:
    print("SHIP: undecided — the battery did not finish. Re-run the script; it resumes.")
elif gate[1] > 0:
    print(f"SHIP: nothing yet. bot6 FAILED the failure gate ({gate[1]} failures in {gate[2]} games).")
    print("      Grep results/overnight.log for the stderr tail BEFORE theorising —")
    print("      the last crash was a glibc heap abort, not a Python exception.")
elif b[2] >= 150 and b[0] > 0.53:
    print("SHIP: bot6B_numbas")
elif a is not None and a[2] >= 100 and a[0] > 0.52:
    print("SHIP: bot5.1_nnue2")
else:
    print("SHIP: bot5_nnue2 unchanged — neither challenger cleared its baseline.")

print("\nThen, and never skip the ship step:")
print("  uv run python -m tools.agents --ship <BUILD>")
print("  uv run python -m tools.selftest")
print("  uv run python -m harness.package")
print("  unzip -l submission.zip | grep -iE 'spar|stockfish|engine'   # must be empty")
PY
exit $FAILED
