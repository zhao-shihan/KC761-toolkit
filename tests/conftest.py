"""Pytest configuration for the KC761 contract tests.

``pytest.ini`` sets ``pythonpath = .``; the explicit insert below keeps the
repository importable when pytest is invoked from another directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
