SHELL := /bin/bash

.PHONY: setup agents play arena zip gate sprt gate-fast collect ship

setup:
	uv sync
	uv run python -m tools.agents

# Materialise versions/ and baselines/ from the flat files in bots/. The harness
# loads agents as directories; the repo stores them as files. Run after any clone.
agents:
	uv run python -m tools.agents

# The failure gate. Zero failures or stop — score here is meaningless.
gate-fast:
	uv run python -m tools.selftest
	uv run python -m tools.h2h --agent versions/$(BUILD) --opponent baselines/random \
		--games 100 --workers 6 --base-ms 1000 --increment-ms 50 \
		--out results/$(BUILD)_gate.json

# The merge decision. BUILD=bot5 VS=bot4_ordering make sprt
sprt:
	uv run python -m tools.h2h --agent versions/$(BUILD) --opponent versions/$(VS) \
		--games $(or $(GAMES),300) --workers 6 --base-ms 8000 --increment-ms 500 \
		--elo1 $(or $(ELO1),5) --out results/$(BUILD)_vs_$(VS).json

collect:
	uv run python -m tools.collect

# Point the root agent.py at a build. This is what actually gets zipped.
ship:
	uv run python -m tools.agents --ship $(BUILD)

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

zip:
	uv run python -m harness.package

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
