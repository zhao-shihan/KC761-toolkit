"""Pure numerics for the KC761 toolkit.

Rules (docs/architecture.md): standard library plus ``numpy``/``scipy`` (and
``numba``/``sympy`` inside generated kernel modules) only. No IO, no
``uproot``, no ``matplotlib``, no Geant4, no imports of other ``kc761tool``
subpackages.
"""

from __future__ import annotations
