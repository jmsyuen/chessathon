#!/usr/bin/env bash
# The bot5 measurement run. This is RUNBOOK.md sections 1-5 applied to bot5, in
# order, with the parts specific to a learned evaluation filled in.
#
#   bash tools/run.sh check        ~2 min    environment, layout, network loads
#   bash tools/run.sh data         ~2-2.5 h  10M stratified positions, 6 workers
#   bash tools/run.sh train        ~80 min   hidden 256 and 512
#   bash tools/run.sh gates        ~40 min   calibrate, checknet, evalcmp. STOPS ON FAIL
#   bash tools/run.sh correctness  ~15 min   selftest per arm + the 100-game failure gate
#   bash tools/run.sh verdict      ~1 h/arm  SPRT against bot4
#   bash tools/run.sh ship         ~5 min    install the winner, package, check the zip
#   bash tools/run.sh ladder       ~3 h      altitude at the real control
#   bash tools/run.sh filler       ~2 h      bot4's own SPRT, stability-break A/B
#   bash tools/run.sh all                    check -> verdict, stopping at any gate
#
# Everything is resumable. gendata workers are line buffered, so a killed one
# appends rather than restarting. train checkpoints weights, Adam moments and the
# epoch counter into --state. h2h checkpoints after every game and resumes from
# its --out file, so re-running the same command adds games.
#
# LMDE natively. Every timed stage measures wall-clock behaviour, and hypervisor
# jitter produces flags that look like time-management bugs.

set -euo pipefail

PY="${PY:-uv run python}"
BUILD="${BUILD:-bot5_nnue2}"          # tools/agents.py names versions/ from the
VS="${VS:-bot4_ordering}"             # file stem, so these are the real names
WORKERS="${WORKERS:-6}"               # six games saturates six physical cores;
                                      # SMT oversubscription distorts timing
POSITIONS_PER_WORKER="${POSITIONS_PER_WORKER:-1700000}"   # 6 x 1.7M = 10.2M
GAMES="${GAMES:-300}"                 # ~300 at elo1=5, ~150 at elo1=10
ELO1="${ELO1:-10}"                    # see the note at the bottom of this file
BASE_MS="${BASE_MS:-8000}"
INCREMENT_MS="${INCREMENT_MS:-500}"
HIDDEN_SIZES="${HIDDEN_SIZES:-256 512}"
ARMS="${ARMS:-256 128}"               # 256 = network alone, 128 = blend with bot4
OPENING_COUNT="${OPENING_COUNT:-256}" # 64 openings is only 128 distinct games
: "${SPAR_ENGINE:=$(command -v stockfish || true)}"
export SPAR_ENGINE

cd "$(dirname "$0")/.."
REPO="$PWD"
mkdir -p data results weights

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

# CHESSATHON_WEIGHTS has to be absolute: the harness spawns the agent as a
# subprocess, so a relative path would resolve against the child's directory.
arm_env() { echo "CHESSATHON_NET_WEIGHT=$1 CHESSATHON_WEIGHTS=$REPO/weights/nnue.npz"; }

stage_check() {
  say "check"
  [ -n "$SPAR_ENGINE" ] || die "no stockfish on PATH; apt-get install stockfish"
  $PY -c "import chess,numba,numpy;print('chess',chess.__version__,'numba',numba.__version__,'numpy',numpy.__version__)"
  [ -f "bots/$BUILD/$BUILD.py" ] || die "bots/$BUILD/$BUILD.py missing. bot5 must be a
  DIRECTORY build so tools/agents.py carries its weights: bots/$BUILD/$BUILD.py
  plus exactly one .npz beside it, mirroring bots/bot3_nnue/."
  grep -q "\"$BUILD\": \"weights/nnue.npz\"" tools/agents.py || die "tools/agents.py
  has no WEIGHT_TARGETS entry for $BUILD. Without it the .npz lands beside the
  agent instead of at weights/nnue.npz, the network does not load, and every
  measurement below is silently of bot4's evaluation."
  $PY -m tools.agents
  say "does the network actually load?"
  BUILD="$BUILD" $PY - <<'EOF'
import importlib.util, os, pathlib, sys
path = pathlib.Path("versions") / os.environ["BUILD"] / "agent.py"
spec = importlib.util.spec_from_file_location("probe", path)
mod = importlib.util.module_from_spec(spec); sys.modules["probe"] = mod
spec.loader.exec_module(mod)
loaded = mod._forward is not None
print(f"  network loaded: {loaded}, hidden {mod._HIDDEN}, cp_scale {mod._CP_SCALE}")
if not loaded:
    print("  (expected before training; it MUST read True before the gates stage)")
EOF
  $PY -m tools.gen_openings --count "$OPENING_COUNT" --out tools/openings.py \
      --judge "versions/$VS"
  echo "commit tools/openings.py before going further"
}

stage_data() {
  say "data: $WORKERS workers x $POSITIONS_PER_WORKER positions"
  for k in $(seq 0 $((WORKERS - 1))); do
    $PY -m tools.gendata --seed "$k" --positions "$POSITIONS_PER_WORKER" \
        --out "data/train.$k.csv" > "results/gen.$k.log" 2>&1 &
  done
  wait
  cat data/train.*.csv > data/train.csv
  wc -l data/train.csv
  grep -h "histogram" results/gen.*.log || true
  echo "expect roughly 45% of rows inside 120 cp of level, against v3's 31.5%"
}

stage_train() {
  for hidden in $HIDDEN_SIZES; do
    say "train hidden $hidden"
    $PY -m tools.train --data data/train.csv --hidden "$hidden" \
        --out "weights/nnue$hidden.npz" --state "data/state$hidden.npz" --epochs 40 \
        2>&1 | tee "results/train.$hidden.log"
  done
}

install_hidden() {
  cp "weights/nnue$1.npz" weights/nnue.npz
  cp weights/nnue.npz "bots/$BUILD/$BUILD.npz"
  $PY -m tools.agents "$BUILD" > /dev/null
}

stage_gates() {
  : > results/passed.txt
  for hidden in $HIDDEN_SIZES; do
    say "gates, hidden $hidden"
    install_hidden "$hidden"

    # Put the network on bot4's centipawn scale. Every pruning margin in the
    # search is a constant relative to bot4's evaluation, so a network on a
    # different scale widens all of them at once. It does not crash; it looks
    # like the evaluation got slightly worse.
    $PY -m tools.calibrate --weights "versions/$BUILD/weights/nnue.npz" \
        --agent "versions/$BUILD" --reference "versions/$VS" \
        2>&1 | tee "results/calibrate.$hidden.txt"
    cp "versions/$BUILD/weights/nnue.npz" "weights/nnue$hidden.npz"   # keep it
    install_hidden "$hidden"

    # Quantisation: a scale error does not crash, it makes the search thrash.
    $PY -m tools.checknet --agent "versions/$BUILD" --state "data/state$hidden.npz" \
        --data data/train.csv 2>&1 | tee "results/checknet.$hidden.txt"

    # The gate that decides: beat bot4's own r on quiet level positions, on the
    # same positions in the same run.
    if $PY -m tools.evalcmp --from-openings --max-imbalance 120 --positions 2000 \
         --agent "versions/$BUILD" --reference "versions/$VS" \
         2>&1 | tee "results/evalcmp.$hidden.txt"; then
      echo "$hidden" >> results/passed.txt
    fi
  done

  [ -s results/passed.txt ] || die "no network beat $VS as an evaluator on quiet level
  positions. Do not run the stages below. bot4's search amplifies evaluation
  error: reverse futility, futility, late move pruning and the improving flag all
  key off the static score, so a network with no edge there prunes badly as well
  as choosing badly. Ship NET_WEIGHT = 0, which is bot4 exactly, and commit
  results/ anyway. A failed gate is a measured result about the data."
  echo "passed: $(tr '\n' ' ' < results/passed.txt)"
}

best_hidden() { head -1 results/passed.txt; }

stage_correctness() {
  install_hidden "$(best_hidden)"
  $PY -m tools.agents --ship "$BUILD"
  for w in $ARMS; do
    say "selftest, arm $w"
    env $(arm_env "$w") $PY -m tools.selftest 2>&1 | tee "results/$BUILD.w$w.selftest.txt"
  done

  say "failure gate: 100 games against random, RUNBOOK section 1"
  # Failure count, not score. Random play wanders into promotion, en-passant and
  # no-legal-move positions far faster than a real engine does.
  env $(arm_env "${ARMS%% *}") $PY -m tools.h2h \
      --agent "versions/$BUILD" --opponent baselines/random \
      --games 100 --workers "$WORKERS" --base-ms 1000 --increment-ms 50 \
      --out "results/${BUILD}_gate.json" 2>&1 | tee "results/$BUILD.gate.txt"
  if grep -qiE "terminations:.*(flag|illegal|crash|error|timeout)" "results/$BUILD.gate.txt"; then
    die "the failure gate found a non-chess termination. Fix it before spending an
  hour on an SPRT that cannot see it."
  fi

  say "kernelbench: nps, nodes-to-depth, first-move cutoff"
  echo "run this in the assistant sandbox too. A Ryzen core is ~1.8x the"
  echo "competition's, so a depth measured here is a depth you will not reach."
  $PY -m tools.kernelbench --agent "versions/$BUILD" --think-ms 2000 \
      2>&1 | tee "results/$BUILD.bench.txt"
  $PY -m tools.kernelbench --agent "versions/$VS" --think-ms 2000 \
      2>&1 | tee "results/$VS.bench.txt"
}

stage_verdict() {
  install_hidden "$(best_hidden)"
  for w in $ARMS; do
    say "SPRT: $BUILD arm $w vs $VS, $GAMES games at elo1=$ELO1"
    env $(arm_env "$w") $PY -m tools.h2h \
        --agent "versions/$BUILD" --opponent "versions/$VS" \
        --games "$GAMES" --workers "$WORKERS" --elo1 "$ELO1" \
        --base-ms "$BASE_MS" --increment-ms "$INCREMENT_MS" \
        --out "results/${BUILD}_w${w}_vs_${VS}.json"
  done
  $PY -m tools.collect
}

stage_ship() {
  install_hidden "$(best_hidden)"
  $PY -m tools.agents --ship "$BUILD"
  $PY -m tools.selftest
  $PY -m harness.package
  unzip -l submission.zip
  if unzip -l submission.zip | grep -iE 'spar|stockfish|engine'; then
    die "something engine-shaped is in the zip. DO NOT UPLOAD."
  fi
  echo
  echo "zip is clean. NET_WEIGHT in agent.py is what plays; the environment"
  echo "variable is only for A/B runs and the platform sets none of them:"
  grep -n "^NET_WEIGHT" agent.py
}

stage_ladder() {
  install_hidden "$(best_hidden)"
  # Rungs are independent, so one process each. Altitude only: never tune
  # against Stockfish, it blunders in different places than the field.
  for n in 1000 3000 10000 30000; do
    env $(arm_env "${ARMS%% *}") SPAR_LEVEL="nodes:$n" \
    $PY -m bots.ladder --agent "versions/$BUILD" --games 40 --rungs "nodes:$n" \
        --base-ms 120000 --increment-ms 500 \
        > "results/$BUILD.rung$n.ladder.txt" 2>&1 &
  done
  wait
  $PY -m tools.collect
}

stage_filler() {
  # Neither depends on bot5. Both settle open items in the log.
  say "bot4's own SPRT against bot1_baseline"
  $PY -m tools.h2h --agent "versions/$VS" --opponent versions/bot1_baseline \
      --games "$GAMES" --workers "$WORKERS" --elo1 "$ELO1" \
      --base-ms "$BASE_MS" --increment-ms "$INCREMENT_MS" \
      --out "results/${VS}_vs_bot1_baseline.json"

  say "stability early-break disabled: one constant, possibly a free ply"
  mkdir -p versions/bot4_nobreak
  sed 's/^STABLE_ITERATIONS: Final = 3$/STABLE_ITERATIONS: Final = 99/' \
      "bots/$VS.py" > versions/bot4_nobreak/agent.py
  grep -q "STABLE_ITERATIONS: Final = 99" versions/bot4_nobreak/agent.py \
    || die "the sed did not apply; bot4's constant must have moved"
  $PY -m tools.h2h --agent versions/bot4_nobreak --opponent "versions/$VS" \
      --games "$GAMES" --workers "$WORKERS" --elo1 "$ELO1" \
      --base-ms "$BASE_MS" --increment-ms "$INCREMENT_MS" \
      --out "results/nobreak_vs_${VS}.json"
  $PY -m tools.collect
}

case "${1:-all}" in
  check)       stage_check ;;
  data)        stage_data ;;
  train)       stage_train ;;
  gates)       stage_gates ;;
  correctness) stage_correctness ;;
  verdict)     stage_verdict ;;
  ship)        stage_ship ;;
  ladder)      stage_ladder ;;
  filler)      stage_filler ;;
  all)         stage_check; stage_data; stage_train; stage_gates
               stage_correctness; stage_verdict ;;
  *) echo "unknown stage: $1"; sed -n '2,15p' "$0"; exit 2 ;;
esac

say "done. commit results/ and bots/$BUILD/$BUILD.npz"

# On --elo1: the hypotheses are what make SPRT slow. At elo1=5 the test asks "is
# this at least 5 Elo better", a score difference of 0.007, so each game carries
# almost no evidence and a 73% result needs ~300 games. At elo1=10 it is ~150, at
# elo1=20 ~90. elo1=5 is a Fishtest setting for banking tiny gains over months.
# With the lock close, 10 is the deliberate choice: it still rejects noise and
# only gives up on changes under 10 Elo, which is not what bot5 is.
