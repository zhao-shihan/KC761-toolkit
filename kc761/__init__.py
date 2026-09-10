"""KC761 toolkit (rewrite).

The rewrite specification is ``docs/plan.md`` and the formula registry is
``docs/derivations.md``. W0-W5 implement the numeric core, product IO, the
calibration/unfolding pipelines and the Geant4 simulation; the ``kc761`` CLI
surface (W6) is still being wired.
"""

from __future__ import annotations

__version__ = "0.0.0+dev"

__all__ = ["__version__"]
