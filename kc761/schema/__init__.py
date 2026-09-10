"""Product contracts, axes and uproot IO.

Owns every object name, unit convention and file format (docs/formats.md) and
is the only layer allowed to touch ``uproot``. W2 implements the readers and
writers; the low-level primitives in ``_uproot.py`` were verified in W0.
"""

from __future__ import annotations
