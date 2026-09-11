"""Unfolding package.

Public entries for the CLI:

* :func:`kc761tool.unfold.compose.run_compose` - compose ``R = C . p_tilde .
  diag(eta)`` over the full primary axis and optionally write the inspection
  artifact (F-RESP-2/F-RESP-3).
* :func:`kc761tool.unfold.unfold.run_unfold` - full unfold (window/pad, exact-zero
  pruning, non-negative Tikhonov solve, strict stat/syst bands) or the
  ``calib_only`` channel-to-energy relabeling (F-SOLVE/F-UNC/F-UNF).

The numerics live in ``kc761tool.core``; this package wires them to products.
"""

from __future__ import annotations

from kc761tool.unfold.compose import run_compose
from kc761tool.unfold.types import ComposeResult, UnfoldResult, UnfoldSettings
from kc761tool.unfold.unfold import run_unfold

__all__ = [
    "ComposeResult",
    "UnfoldResult",
    "UnfoldSettings",
    "run_compose",
    "run_unfold",
]
