"""Unfolding workstream package (W4).

User-facing orchestration for composition, non-negative regularized
unfolding and uncertainty bands. Formula IDs F-SOLVE-*, F-UNC-*. The core
numerics live in ``kc761.core``; this package wires them to products.
Implementation starts in W4; the W0 CLI only parses arguments.
"""

from __future__ import annotations
