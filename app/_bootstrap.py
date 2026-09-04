"""Bootstrap the app/ scripts: make the repository root importable.

Importing this module inserts the repository root (which holds the kc761*
packages) at the front of ``sys.path``, so the app scripts can import those
packages when run directly, e.g. ``python app/calib.py``.  The app/
directory itself is importable in every supported invocation mode: direct
runs get it as ``sys.path[0]`` and ``python -m app.calib`` from the
repository root gets it from ``app/__init__.py``, so the scripts always
import this module as plain ``_bootstrap``.  It also exposes ``APP_DIR``
and ``REPO_ROOT`` for scripts that need them (default outputs are collected
in the out/ directory next to the repository root).
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
