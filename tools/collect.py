"""Digest everything under `results/` into one file small enough to read.

    python3 -m tools.collect                  # write results/SUMMARY.md, print it
    python3 -m tools.collect --paste          # print only, for dropping in a chat

This is the return channel. A session cannot watch the PC work, so what comes
back has to be self-contained and short. Raw per-game JSON is neither: a 300-game
SPRT is thousands of lines that say almost nothing a reader needs.

So: raw files stay in `results/` for audit, and this writes the digest that gets
read. Commit both. The digest is the thing to paste into an iteration chat when a
clone is inconvenient.

What it reads
-------------
  results/*.json              h2h runs, including shards, pooled per base name
  results/*.ladder.txt        tools.ladder output, captured with tee
  results/*.selftest.txt      tools.selftest output
  results/*.bench.txt         tools.bench output

Naming is the only convention: call the run what it is. `bot5_vs_bot4.json`
produces a row headed bot5_vs_bot4. Shards (`bot5_vs_bot4.shard0.json`) are
folded into their parent automatically, so a parallel run reads as one result.

The failure gate is reported first and separately, because a run with a crash in
it has no strength information worth reading yet.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.referee import FAILED_TERMINATIONS
from tools.h2h import (
    LOWER_BOUND,
    UPPER_BOUND,
    elo_difference,
    log_likelihood_ratio,
    score_interval,
    tally,
)

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

SHARD = re.compile(r"^(?P<base>.+?)\.shard\d+$")


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _machine() -> str:
    """Record where a number came from. A depth measured on a 6-core desktop and
    one measured on a single slow core are not the same number, and six months
    from now nobody will remember which was which."""
    cores = ""
    try:
        import os

        cores = f"{os.cpu_count()} cores, "
    except Exception:
        pass
    return f"{platform.node()} ({cores}{platform.machine()}, {platform.system()})"


def load_runs() -> dict[str, list[dict[str, Any]]]:
    """Pool every JSON under results/, folding shards into their parent."""
    runs: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, dict[str, Any]] = {}
    if not RESULTS.is_dir():
        return runs

    for path in sorted(RESULTS.glob("*.json")):
        if path.name == "SUMMARY.json":
            continue
        stem = path.stem
        match = SHARD.match(stem)
        base = match.group("base") if match else stem
        try:
            state = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  (skipping unreadable {path.name})", file=sys.stderr)
            continue
        runs.setdefault(base, []).extend(state.get("games", []))
        for key in ("agent", "opponent", "base_ms", "increment_ms"):
            if key in state:
                meta.setdefault(base, {}).setdefault(key, state[key])

    for base in runs:
        runs[base] = runs[base]
    load_runs.meta = meta  # type: ignore[attr-defined]
    return runs


def summarise(lines: list[str]) -> int:
    runs = load_runs()
    meta: dict[str, dict[str, Any]] = getattr(load_runs, "meta", {})
    failures_total = 0

    lines.append(f"# Results — {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    lines.append("")
    branch, commit = _git("rev-parse", "--abbrev-ref", "HEAD"), _git("rev-parse", "--short", "HEAD")
    dirty = " (uncommitted changes present)" if _git("status", "--porcelain") else ""
    lines.append(f"- machine: {_machine()}")
    if commit:
        lines.append(f"- commit: `{commit}` on `{branch}`{dirty}")
    lines.append("")

    if not runs:
        lines.append("No match results found under `results/`.")
        return 0

    # --- gate first -------------------------------------------------------
    lines.append("## Failure gate")
    lines.append("")
    gate_rows: list[str] = []
    for base, games in sorted(runs.items()):
        counts: dict[str, int] = {}
        for game in games:
            name = game.get("termination", "?")
            if name in FAILED_TERMINATIONS:
                counts[name] = counts.get(name, 0) + 1
        failures = sum(counts.values())
        failures_total += failures
        detail = ", ".join(f"{n} {c}" for n, c in sorted(counts.items())) if counts else "clean"
        gate_rows.append(f"| {base} | {len(games)} | {failures} | {detail} |")
    lines.append("| run | games | failures | detail |")
    lines.append("|---|---|---|---|")
    lines.extend(gate_rows)
    lines.append("")
    if failures_total:
        lines.append(
            f"**{failures_total} failed game(s). The gate is not clean, so nothing below "
            "is worth reading yet.** Find whose fault each one was before treating any "
            "strength number as real."
        )
    else:
        lines.append("**Gate clean: zero failures across all runs.**")
    lines.append("")

    # --- strength ---------------------------------------------------------
    lines.append("## Match results")
    lines.append("")
    lines.append("| run | control | +W =D -L | score | Elo (95%) | LLR | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for base, games in sorted(runs.items()):
        wins, draws, losses = tally(games)
        if wins + draws + losses == 0:
            continue
        score, low, high = score_interval(wins, draws, losses)
        llr = log_likelihood_ratio(wins, draws, losses)
        elo = elo_difference(score)
        information = meta.get(base, {})
        control = f"{information.get('base_ms', '?')}ms+{information.get('increment_ms', '?')}ms"
        if llr >= UPPER_BOUND:
            verdict = "**PASS**"
        elif llr <= LOWER_BOUND:
            verdict = "**FAIL**"
            
        else:
            verdict = "undecided"
        elo_text = (
            "n/a" if abs(elo) == float("inf")
            else f"{elo:+.0f} ({elo_difference(low):+.0f}..{elo_difference(high):+.0f})"
        )
        lines.append(
            f"| {base} | {control} | +{wins} ={draws} -{losses} | {score:.1%} | "
            f"{elo_text} | {llr:+.2f} | {verdict} |"
        )
    lines.append("")
    lines.append(
        "Elo intervals are wide at small samples by nature. A verdict of *undecided* "
        "means exactly that — not 'probably fine'."
    )
    lines.append("")

    # --- captured console output -----------------------------------------
    for label, pattern in (
        ("Ladder", "*.ladder.txt"),
        ("Selftest", "*.selftest.txt"),
        ("Bench", "*.bench.txt"),
        ("Kernel benchmark", "*.kernelbench.txt"),
    ):
        captured = sorted(RESULTS.glob(pattern))
        if not captured:
            continue
        lines.append(f"## {label}")
        lines.append("")
        for path in captured:
            body = path.read_text().strip()
            lines.append(f"### {path.name}")
            lines.append("")
            lines.append("```")
            lines.append(body if len(body) < 4000 else body[-4000:])
            lines.append("```")
            lines.append("")

    return 1 if failures_total else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paste", action="store_true", help="print only, write nothing")
    arguments = parser.parse_args()

    lines: list[str] = []
    status = summarise(lines)
    text = "\n".join(lines)

    if not arguments.paste:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "SUMMARY.md").write_text(text + "\n")
        print(f"wrote {(RESULTS / 'SUMMARY.md').relative_to(REPO)}\n")
    print(text)
    return status


if __name__ == "__main__":
    sys.exit(main())
