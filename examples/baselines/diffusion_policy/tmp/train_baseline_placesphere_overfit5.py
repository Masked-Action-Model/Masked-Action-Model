"""PlaceSphere 5-demo overfit entry for the baseline pipeline.

This is a thin copy-style entrypoint: it delegates to the normal baseline
trainer so the training/eval pipeline stays identical to the main file.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
DP_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[4]

for path in (REPO_ROOT, DP_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

runpy.run_path(str(DP_ROOT / "train_baseline.py"), run_name="__main__")
