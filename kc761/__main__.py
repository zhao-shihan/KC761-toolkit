"""Entry point for ``python -m kc761``."""

from __future__ import annotations

import sys

from kc761.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
