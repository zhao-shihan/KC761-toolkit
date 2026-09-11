"""sim config batch driver: argv expansion, resume, failure policy (D-134..D-137).

The child processes are faked so the test exercises the driver without Geant4;
``test_cli_sim_g4.py`` is the real end-to-end run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from kc761tool.cli import main
from kc761tool.schema.axes import channel_axis
from kc761tool.schema.io import write_product
from kc761tool.schema.products import (
    SCHEMA_VERSION,
    Histogram1D,
    Provenance,
    SpectrumProduct,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class Recorder:
    def __init__(self, codes: list[int] | None = None) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self.codes = list(codes or [])

    def __call__(self, command, cwd=None, check=False):  # noqa: ANN001
        self.calls.append((list(command), cwd))
        code = self.codes.pop(0) if self.codes else 0
        return subprocess.CompletedProcess(command, code)


def _config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "sim.toml"
    path.write_text("config_version = 1\n[sim]\n" + body, encoding="utf-8")
    return path


def _valid_product(path: Path) -> None:
    product = SpectrumProduct(
        format_version=SCHEMA_VERSION,
        spectrum=Histogram1D(
            axis=channel_axis(2),
            values=[1.0, 2.0],
            variances=[1.0, 2.0],
        ),
        daq_time_s=1.0,
        source_file="fixture.csv",
        provenance=Provenance(
            created_utc="2026-09-10T00:00:00Z",
            producer="test",
            command="test",
            arguments=(),
            git_revision="unknown",
            git_dirty=False,
            python_version="3",
            dependency_versions=(),
            inputs=(),
        ),
    )
    write_product(product, path, force=True)


def test_dry_run_prints_commands_without_spawning(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    recorder = Recorder()
    monkeypatch.setattr("kc761tool.cli.sim.subprocess.run", recorder)
    out1 = tmp_path / "a.root"
    out2 = tmp_path / "b.root"
    config = _config(
        tmp_path,
        "[[sim.runs]]\n"
        'source = "am241"\nevents = 5\n'
        f'output = "{out1}"\n'
        "[[sim.runs]]\n"
        'source = "lu176"\nevents = 6\n'
        f'output = "{out2}"\n',
    )
    assert main(["sim", "-c", str(config), "--dry-run"]) == 0
    assert recorder.calls == []
    out = capsys.readouterr().out
    assert out.count("(dry-run)") == 2
    assert "--am241" in out and "--lu176" in out


def test_serial_order_and_argv_expansion(tmp_path: Path, monkeypatch: Any) -> None:
    recorder = Recorder([0, 0])
    monkeypatch.setattr("kc761tool.cli.sim.subprocess.run", recorder)
    out1 = tmp_path / "a.root"
    out2 = tmp_path / "b.root"
    config = _config(
        tmp_path,
        "[[sim.runs]]\n"
        'source = "am241"\nevents = 5\n'
        f'output = "{out1}"\n'
        "[[sim.runs]]\n"
        'source = "lu176"\nevents = 6\nseed = 99\n'
        f'output = "{out2}"\n',
    )
    assert main(["sim", "-c", str(config), "--strict"]) == 0
    assert len(recorder.calls) == 2
    first, second = recorder.calls
    assert first[0][1:4] == ["-m", "kc761tool", "sim"]
    assert "--am241" in first[0]
    assert "--lu176" in second[0]
    assert "-s" in second[0] and "99" in second[0]
    assert "--strict" in first[0]
    assert "--provenance-input" in first[0]
    assert str(config.resolve()) in first[0]
    assert Path(first[1]) == REPO_ROOT


def test_matrix_run_uses_mode_flag_and_calib(tmp_path: Path, monkeypatch: Any) -> None:
    recorder = Recorder([0])
    monkeypatch.setattr("kc761tool.cli.sim.subprocess.run", recorder)
    calib = tmp_path / "calib.root"
    output = tmp_path / "g.root"
    config = _config(
        tmp_path,
        "[[sim.runs]]\n"
        'mode = "sphere-gamma"\n'
        f'calib = "{calib}"\nevents = 7\nthreads = 2\n'
        f'output = "{output}"\n',
    )
    assert main(["sim", "-c", str(config)]) == 0
    command = recorder.calls[0][0]
    assert "--sphere-gamma" in command
    assert str(calib) in command
    assert "--plane-front-gamma" not in command


def test_batch_failure_continues_and_returns_one(
    tmp_path: Path, monkeypatch: Any
) -> None:
    recorder = Recorder([1, 0])
    monkeypatch.setattr("kc761tool.cli.sim.subprocess.run", recorder)
    config = _config(
        tmp_path,
        "[[sim.runs]]\n"
        'source = "am241"\nevents = 5\n'
        f'output = "{tmp_path / "a.root"}"\n'
        "[[sim.runs]]\n"
        'source = "lu176"\nevents = 5\n'
        f'output = "{tmp_path / "b.root"}"\n',
    )
    assert main(["sim", "-c", str(config)]) == 1
    assert len(recorder.calls) == 2


def test_config_matrix_default_filename_token(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    recorder = Recorder()
    monkeypatch.setattr("kc761tool.cli.sim.subprocess.run", recorder)
    calib = tmp_path / "calib.root"
    config = _config(
        tmp_path,
        "[[sim.runs]]\n"
        'mode = "plane-front-gamma"\n'
        f'calib = "{calib}"\nevents = 5\n',
    )
    assert main(["sim", "-c", str(config), "--dry-run"]) == 0
    assert recorder.calls == []
    out = capsys.readouterr().out
    assert "calib-plane-front-gamma-n5-s908136382.root" in out


def test_resume_skips_a_valid_existing_target(
    tmp_path: Path, monkeypatch: Any
) -> None:
    recorder = Recorder()
    monkeypatch.setattr("kc761tool.cli.sim.subprocess.run", recorder)
    target = tmp_path / "existing.root"
    _valid_product(target)
    config = _config(
        tmp_path,
        "[[sim.runs]]\n"
        'source = "am241"\nevents = 5\n'
        f'output = "{target}"\n',
    )
    assert main(["sim", "-c", str(config)]) == 0
    assert recorder.calls == []


def test_resume_invalid_existing_target_fails_without_spawning(
    tmp_path: Path, monkeypatch: Any
) -> None:
    recorder = Recorder()
    monkeypatch.setattr("kc761tool.cli.sim.subprocess.run", recorder)
    target = tmp_path / "broken.root"
    target.write_bytes(b"not a root file")
    config = _config(
        tmp_path,
        "[[sim.runs]]\n"
        'source = "am241"\nevents = 5\n'
        f'output = "{target}"\n',
    )
    assert main(["sim", "-c", str(config)]) == 1
    assert recorder.calls == []


def test_batch_force_flag_is_propagated(tmp_path: Path, monkeypatch: Any) -> None:
    recorder = Recorder([0])
    monkeypatch.setattr("kc761tool.cli.sim.subprocess.run", recorder)
    config = _config(
        tmp_path,
        "force = true\nresume = false\n"
        "[[sim.runs]]\n"
        'source = "am241"\nevents = 5\n'
        f'output = "{tmp_path / "a.root"}"\n',
    )
    assert main(["sim", "-c", str(config)]) == 0
    assert "--force" in recorder.calls[0][0]


def test_parent_does_not_import_geant4(tmp_path: Path, monkeypatch: Any) -> None:
    recorder = Recorder([0])
    monkeypatch.setattr("kc761tool.cli.sim.subprocess.run", recorder)
    config = _config(
        tmp_path,
        "[[sim.runs]]\n"
        'source = "am241"\nevents = 5\n'
        f'output = "{tmp_path / "a.root"}"\n',
    )
    assert main(["sim", "-c", str(config)]) == 0
    assert "geant4_pybind" not in sys.modules
