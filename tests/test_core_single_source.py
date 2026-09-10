"""Auxiliary checks for the mechanical single-source gate (D-60/D-75)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "kc761_check_single_source", ROOT / "tools" / "check_single_source.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_source_check_passes(capsys) -> None:
    module = _load_checker()
    assert module.main() == 0
    assert "passed" in capsys.readouterr().out


def test_generated_kernels_match_manifest() -> None:
    from kc761.core import _gen

    assert _gen.stale_files() == []
