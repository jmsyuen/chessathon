#!/usr/bin/env bash
# Sweep bot5's two free constants. No retraining: both are one line in agent.py
# and the network is read from the .npz unchanged.
#
#   bash tools/sweep.sh nobreak    ~30 min  does the stability break compose?
#   bash tools/sweep.sh weight     ~2 h     NET_WEIGHT sweep vs the reference
#   bash tools/sweep.sh weight 48 64 80      sweep specific values
#
# Why this builds directories instead of setting CHESSATHON_NET_WEIGHT:
# the harness spawns both agents as children of one h2h process and they inherit
# its environment, so an env var reaches the OPPONENT as well. That was fine when
# the opponent was bot4, which has no such knob. Comparing two arms of bot5 that
# way silently gives both sides the same weight and measures nothing.

set -euo pipefail
PY="${PY:-uv run python}"
WORKERS="${WORKERS:-6}"
GAMES="${GAMES:-150}"
ELO1="${ELO1:-10}"
BASE_MS="${BASE_MS:-8000}"
INCREMENT_MS="${INCREMENT_MS:-500}"
SRC="bots/bot5_nnue2/bot5_nnue2.py"
NPZ="bots/bot5_nnue2/bot5_nnue2.npz"

cd "$(dirname "$0")/.."
mkdir -p results

# build a version directory with the two constants baked in
variant() {  # name, net_weight, stable_iterations
  local dir="versions/$1"
  mkdir -p "$dir/weights"
  sed -e "s/^NET_WEIGHT: int = .*/NET_WEIGHT: int = $2/" \
      -e "s/^STABLE_ITERATIONS: Final = .*/STABLE_ITERATIONS: Final = $3/" \
      "$SRC" > "$dir/agent.py"
  cp "$NPZ" "$dir/weights/nnue.npz"
  grep -q "^NET_WEIGHT: int = $2\$" "$dir/agent.py" || { echo "sed failed on NET_WEIGHT"; exit 1; }
  grep -q "^STABLE_ITERATIONS: Final = $3\$" "$dir/agent.py" || { echo "sed failed on STABLE"; exit 1; }
  echo "  built $dir  (net $2/256, stable $3)"
}

match() {  # agent, opponent, tag
  echo; echo "=== $1 vs $2 ==="
  $PY -m tools.h2h --agent "versions/$1" --opponent "versions/$2" \
      --games "$GAMES" --workers "$WORKERS" --elo1 "$ELO1" \
      --base-ms "$BASE_MS" --increment-ms "$INCREMENT_MS" \
      --out "results/$3.json"
}

case "${1:-}" in
  nobreak)
    variant bot5_blend        128 3
    variant bot5_blend_nb     128 99
    match bot5_blend_nb bot5_blend sweep_nobreak
    echo; echo "If this passes, bot5_blend_nb is the new reference for the weight sweep."
    ;;
  weight)
    REF="${REF:-bot5_blend_nb}"       # set REF=bot5_blend if nobreak did not pass
    STABLE=99; [ "$REF" = "bot5_blend" ] && STABLE=3
    variant "$REF" 128 "$STABLE"
    shift || true
    WEIGHTS=("$@"); [ ${#WEIGHTS[@]} -gt 0 ] || WEIGHTS=(96 160 192)
    for w in "${WEIGHTS[@]}"; do
      variant "bot5_w$w" "$w" "$STABLE"
      match "bot5_w$w" "$REF" "sweep_w$w"
    done
    ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
$PY -m tools.collect
