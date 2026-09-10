"""Geant4 simulation workstream package (W5).

Owns geometry, materials, sources, detector construction, physics, actions
and the batch runner (formula IDs F-SIM-1..F-SIM-5). Geant4 is imported
lazily inside functions only, so this module must stay importable without
Geant4 installed.

``SOURCE_KEYS`` mirrors the frozen pre-rewrite source registry so the W0 CLI
surface is stable; W5 owns the final physics registry and may revise the keys
through the contract-change process (docs/plan.md Appendix A item 6).
"""

from __future__ import annotations

SOURCE_KEYS: tuple[str, ...] = (
    "k40",
    "lu176",
    "am241",
    "th232",
    "th232-unshielded",
    "ra226",
    "ra226-unshielded",
)
