"""Re-export of the NNUE trainer under the module name its callers expect.

`v3_checknet.py` does `from tools.train import PAD, encode` and `v3_evalcmp.py`
does `from tools.gendata import SF, Engine`, but the files themselves live in
`bots/bot3_nnue/` as version artifacts. From a clean clone neither import
resolves, so the whole eval-gate chain is unrunnable — which matters now that
the machine running it is not the machine that wrote it.

This is a re-export rather than a copy on purpose. The feature encoding has to
be byte-identical between the trainer and the agent kernel; a second copy of
`encode` is the exact shape of bug that looks like a badly trained network.

    python3 -m tools.train --data data/train.csv --out weights/nnue.npz --hidden 512
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SOURCE = Path(__file__).resolve().parent.parent / "bots" / "bot3_nnue" / "v3_train.py"

_spec = importlib.util.spec_from_file_location("_v3_train", _SOURCE)
if _spec is None or _spec.loader is None:  # pragma: no cover - a broken checkout
    raise ImportError(f"cannot load the trainer from {_SOURCE}")
_module = importlib.util.module_from_spec(_spec)
sys.modules["_v3_train"] = _module
_spec.loader.exec_module(_module)

PAD = _module.PAD
SCALE = _module.SCALE
QA = _module.QA
QB = _module.QB
MAX_PIECES = _module.MAX_PIECES
encode = _module.encode
material_of = _module.material_of
load = _module.load
save = _module.save
main = _module.main

__all__ = [
    "MAX_PIECES", "PAD", "QA", "QB", "SCALE",
    "encode", "load", "main", "material_of", "save",
]

if __name__ == "__main__":
    raise SystemExit(main())
