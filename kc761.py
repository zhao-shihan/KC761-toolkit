#!/usr/bin/env python3
"""Repository-root launcher for the KC761 toolkit.

Packaging is intentionally not provided (docs/plan.md D-7), so this script
adds the repository root to ``sys.path`` and calls the CLI. It executes as
``__main__`` and never imports itself; the ``kc761/`` package takes precedence
for ordinary imports.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _run(argv: list[str]) -> int:
    root = str(Path(__file__).resolve().parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from kc761.cli import main

    return main(argv)


if __name__ == "__main__":
    raise SystemExit(_run(sys.argv[1:]))
