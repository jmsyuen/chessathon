"""Materialise runnable agent directories from the flat files in `bots/`.

    python3 -m tools.agents            # build everything
    python3 -m tools.agents --check    # report what is missing, write nothing
    python3 -m tools.agents --ship bot4_ordering   # also point root agent.py at it

Why this exists: the harness loads an agent as a *directory* containing
`agent.py`, but builds are stored as flat files in `bots/`. Nothing in the repo
bridged the two, so a fresh clone could run `selftest.py` and `kernelbench.py`
and could not play a single game — every match tool defaulted to a path that did
not exist. That gap was being closed by hand at the start of each session, which
is the kind of undocumented manual step that quietly invalidates a comparison
when someone does it slightly differently.

Layout it produces:

    versions/<name>/agent.py        our builds, the things we measure
    baselines/<name>/agent.py       opponents: random, greedy, minimax, numba
    baselines/stockfish/agent.py    the sparring engine

Weights travel with their agent. `bot3_nnue.py` loads from
`Path(__file__).parent / "weights" / "nnue.npz"`, so the `.npz` is copied there
under that name rather than left beside the source.

Safe to re-run: it only rewrites a file whose contents differ, so timestamps stay
stable and a build that is mid-benchmark is not disturbed.

It never writes to the repo root. `harness/package.py` globs every `*.py` there
into `submission.zip`, so a stray agent copy at the root is how an engine wrapper
becomes a disqualification.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BOTS = REPO / "bots"
VERSIONS = REPO / "versions"
BASELINES = REPO / "baselines"

# Anything matching these goes to baselines/ under a shortened name. They are
# opponents and sparring partners, never things we ship or measure as candidates.
BASELINE_NAMES: dict[str, str] = {
    "bot0_random": "random",
    "bot0_greedy": "greedy",
    "bot0_minimax": "minimax",
    "bot0_numba": "numba",
    "bot_stockfish_spar": "stockfish",
}

# Weights are copied to the name the agent actually opens, not the name they are
# stored under. Keyed by build name.
WEIGHT_TARGETS: dict[str, str] = {
    "bot3_nnue": "weights/nnue.npz",
    "bot5_nnue2": "weights/nnue.npz",
}

WEIGHT_SUFFIXES = (".npz", ".onnx", ".safetensors", ".pt")

# Files under bots/ that are not agents. `ladder.py` is a measurement tool that
# happens to live there; the iteration log calls it tools/ladder.py. Materialising
# it as an agent produces a versions/ladder that imports argparse and dies on the
# first get_move call, which is a confusing way to find out.
NOT_AGENTS: frozenset[str] = frozenset({"ladder", "__init__"})


class Plan:
    def __init__(self) -> None:
        self.written: list[str] = []
        self.unchanged: list[str] = []
        self.missing: list[str] = []


def _place(source: Path, target: Path, plan: Plan, check: bool) -> None:
    label = str(target.relative_to(REPO))
    if target.exists() and filecmp.cmp(source, target, shallow=False):
        plan.unchanged.append(label)
        return
    if check:
        plan.missing.append(label)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    plan.written.append(label)


def _destination(name: str) -> Path:
    if name in BASELINE_NAMES:
        return BASELINES / BASELINE_NAMES[name]
    return VERSIONS / name


def discover() -> dict[str, tuple[Path, list[Path]]]:
    """Map build name -> (source .py, [weight files]).

    A build is either `bots/<name>.py` or `bots/<name>/<name>.py`. The directory
    form carries its weights and its training scripts; only the weights travel.
    """
    builds: dict[str, tuple[Path, list[Path]]] = {}
    if not BOTS.is_dir():
        return builds

    for entry in sorted(BOTS.iterdir()):
        if entry.is_file() and entry.suffix == ".py":
            if entry.stem in NOT_AGENTS:
                continue
            builds[entry.stem] = (entry, [])
        elif entry.is_dir():
            source = entry / f"{entry.name}.py"
            if not source.exists():
                candidates = [p for p in sorted(entry.glob("*.py")) if not p.name.startswith("v")]
                if not candidates:
                    continue
                source = candidates[0]
            weights = [p for p in sorted(entry.iterdir()) if p.suffix in WEIGHT_SUFFIXES]
            builds[entry.name] = (source, weights)
    return builds


def build(names: list[str] | None, check: bool) -> Plan:
    plan = Plan()
    builds = discover()
    if not builds:
        print(f"no builds found under {BOTS}", file=sys.stderr)
        return plan

    for name, (source, weights) in builds.items():
        if names and name not in names:
            continue
        destination = _destination(name)
        _place(source, destination / "agent.py", plan, check)
        for weight in weights:
            relative = WEIGHT_TARGETS.get(name, weight.name)
            _place(weight, destination / relative, plan, check)
    return plan


def ship(name: str) -> int:
    """Point the root agent.py at a build. This is what actually gets zipped."""
    builds = discover()
    if name not in builds:
        print(f"unknown build {name!r}; have {', '.join(sorted(builds))}", file=sys.stderr)
        return 1
    if name in BASELINE_NAMES:
        print(f"{name} is a baseline, not a candidate. Refusing to ship it.", file=sys.stderr)
        return 1

    source, weights = builds[name]
    target = REPO / "agent.py"
    shutil.copy2(source, target)
    placed = [f"agent.py <- bots/{source.relative_to(BOTS)}"]

    # package.py includes weights/ via DEFAULT_INCLUDES, so weights go there.
    for weight in weights:
        relative = WEIGHT_TARGETS.get(name, f"weights/{weight.name}")
        destination = REPO / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weight, destination)
        placed.append(f"{relative} <- {weight.relative_to(REPO)}")

    for line in placed:
        print(f"  {line}")
    print(f"\nroot agent.py is now {name}. Run the gate before zipping:")
    print("  uv run python -m tools.selftest")
    print("  uv run python -m harness.package && unzip -l submission.zip")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="builds to materialise; default all")
    parser.add_argument("--check", action="store_true", help="report only, write nothing")
    parser.add_argument("--ship", metavar="BUILD", help="copy a build to the root agent.py")
    arguments = parser.parse_args()

    if arguments.ship:
        return ship(arguments.ship)

    plan = build(arguments.names or None, arguments.check)

    if arguments.check:
        if plan.missing:
            print(f"{len(plan.missing)} file(s) missing or stale:")
            for label in plan.missing:
                print(f"  {label}")
            print("\nrun: python3 -m tools.agents")
            return 1
        print(f"all {len(plan.unchanged)} file(s) present and current")
        return 0

    for label in plan.written:
        print(f"  wrote {label}")
    print(f"\n{len(plan.written)} written, {len(plan.unchanged)} already current")
    if plan.written or plan.unchanged:
        print("\nagents ready:")
        for directory in (VERSIONS, BASELINES):
            if directory.is_dir():
                for entry in sorted(directory.iterdir()):
                    if (entry / "agent.py").exists():
                        print(f"  {entry.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
