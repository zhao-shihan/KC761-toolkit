"""Package marker for the app/ scripts.

Making app/ a package keeps ``python -m app.calib`` (from the repository
root) able to resolve ``_bootstrap`` exactly like the direct-run form
``python app/calib.py``: ``-m`` imports this module before the script body
runs, so inserting the app/ directory into ``sys.path`` here lets every
script import ``_bootstrap`` with a single clean import in both modes.
Direct runs never import this module (the app/ directory already sits at
``sys.path[0]``).
"""

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
