"""Unfolding workstream package (W4).

Public entries for the W6 CLI:

* :func:`kc761.unfold.compose.run_compose` - compose ``R = C . p_tilde .
  diag(eta)`` over the full primary axis and optionally write the inspection
  artifact (F-RESP-2/F-RESP-3).
* :func:`kc761.unfold.unfold.run_unfold` - full unfold (window/pad, exact-zero
  pruning, non-negative Tikhonov solve, strict stat/syst bands) or the
  ``calib_only`` channel-to-energy relabeling (F-SOLVE/F-UNC/F-UNF).

The numerics live in ``kc761.core``; this package wires them to products.
"""

from __future__ import annotations

from kc761.unfold.compose import run_compose
from kc761.unfold.types import ComposeResult, UnfoldResult, UnfoldSettings
from kc761.unfold.unfold import run_unfold

__all__ = [
    "ComposeResult",
    "UnfoldResult",
    "UnfoldSettings",
    "run_compose",
    "run_unfold",
]
