"""The shipped examples/*.toml parse under the strict config parser (D-140)."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

from kc761tool.cli.config import (
    load_calib_config,
    load_compose_config,
    load_sim_config,
    load_unfold_config,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
EXAMPLE_NAMES = ("calib", "compose", "sim", "unfold")


def test_sim_example_parses() -> None:
    config = load_sim_config(
        EXAMPLES / "sim.toml",
    )
    assert len(config.runs) == 6
    assert [run.source_key for run in config.runs] == [
        "am241",
        "lu176",
        "ra226",
        "th232",
        "k40",
        None,
    ]
    assert [run.events for run in config.runs[:-1]] == [
        3_000_000,
        20_000_000,
        100_000_000,
        200_000_000,
        1_000_000_000,
    ]
    matrix = config.runs[-1]
    assert matrix.matrix_mode == "plane-front-gamma"
    assert matrix.calib is not None
    assert matrix.threads == 8
    assert config.resume is True


def test_calib_example_parses() -> None:
    config = load_calib_config(EXAMPLES / "calib.toml")
    assert [entry.label for entry in config.datasets] == [
        "Am241",
        "Lu176",
        "Th232",
        "Ra226",
    ]
    assert [entry.channel_low for entry in config.datasets] == [140, 140, 140, 140]
    assert [entry.channel_high for entry in config.datasets] == [165, 450, 1400, 1400]
    assert all(entry.syst_frac == 0.05 for entry in config.datasets)
    assert config.output is not None


def test_compose_example_parses() -> None:
    config = load_compose_config(EXAMPLES / "compose.toml")
    calib = load_calib_config(EXAMPLES / "calib.toml")
    assert calib.output is not None
    # compose reads the product the calib example writes, so the two examples
    # pin one path spelling between them (rule 11) instead of a stored literal.
    assert config.calib == calib.output
    assert config.output is None


def test_unfold_example_parses() -> None:
    """The shipped example relies on the D-191/D-193 defaults."""
    config = load_unfold_config(EXAMPLES / "unfold.toml")
    assert config.alpha == 1.0
    assert config.energy_low_kev == 30.0
    assert config.energy_high_kev == 3000.0
    assert config.syst_frac == 0.05
    assert config.snip_enabled is True
    assert config.snip_floor == 0.01
    assert config.snip_max_iterations == 32
    assert config.log_plot is False
    assert config.calib_only is False


def _example_work_paths() -> set[str]:
    """Every ``work/`` path the shipped examples name, as repo-relative posix."""
    paths: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and node.startswith("work/"):
            paths.add(node)

    for name in EXAMPLE_NAMES:
        with (EXAMPLES / f"{name}.toml").open("rb") as handle:
            walk(tomllib.load(handle))
    return paths


def _load_benchmarks():
    spec = importlib.util.spec_from_file_location(
        "kc761tool_benchmarks", ROOT / "tools" / "benchmarks.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmarks_use_the_example_paths() -> None:
    """D-185: ``examples/`` and ``tools/benchmarks.py`` follow one convention."""
    benchmarks = _load_benchmarks()
    used = {
        benchmarks.CALIB_PRODUCT,
        benchmarks.SIM_PRODUCT,
        benchmarks.DATA_PRODUCT,
        *(benchmarks.REPO_ROOT / row[1] for row in benchmarks.CALIB_DATASETS),
        *(benchmarks.REPO_ROOT / row[2] for row in benchmarks.CALIB_DATASETS),
    }
    relative = {path.relative_to(benchmarks.REPO_ROOT).as_posix() for path in used}
    missing = sorted(relative - _example_work_paths())
    assert not missing, missing
