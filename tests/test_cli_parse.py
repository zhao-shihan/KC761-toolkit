"""CLI argument parsing, exit codes and default naming (D-19/D-66/D-68)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kc761.cli import main
from kc761.cli._common import REPO_ROOT, default_output

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "data"
SMALL_CSV = FIXTURES / "kc761_small.csv"
EXAMPLES = REPO_ROOT / "examples"


def run_cli(argv: list[str]) -> int:
    """Run the CLI in-process, converting argparse SystemExit into a code."""
    try:
        return main(argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 2


def test_version_exits_zero() -> None:
    assert run_cli(["--version"]) == 0


def test_no_subcommand_is_usage_error() -> None:
    assert run_cli([]) == 2


def test_unknown_subcommand_is_usage_error() -> None:
    assert run_cli(["bogus"]) == 2


def test_csv2root_missing_positional_is_usage_error() -> None:
    assert run_cli(["csv2root"]) == 2


def test_sim_without_selection_is_usage_error() -> None:
    assert run_cli(["sim"]) == 2


def test_calib_legacy_sim_flag_is_not_accepted() -> None:
    """D-144: the calibration simulation option is --mc, with no --sim alias."""
    assert (
        run_cli(
            [
                "calib",
                "--data",
                "d.root",
                "--sim",
                "m.root",
                "--label",
                "am241",
            ]
        )
        == 2
    )


def test_calib_mc_flag_parses_dry_run(capsys: Any) -> None:
    code = run_cli(
        [
            "calib",
            "--data",
            "d.root",
            "--mc",
            "m.root",
            "--label",
            "am241",
            "--channel-low",
            "0",
            "--channel-high",
            "10",
            "--dry-run",
        ]
    )
    assert code == 0
    assert "mc=m.root" in capsys.readouterr().out


def test_calib_optimizer_flags_map_to_fit_settings() -> None:
    from kc761.calib.types import FitSettings
    from kc761.cli.calib import _settings

    assert _settings(None, None) is None
    custom = _settings(123, 1e-4)
    assert isinstance(custom, FitSettings)
    assert custom.maxiter == 123
    assert (custom.ftol, custom.xtol, custom.gtol) == (1e-4, 1e-4, 1e-4)


def test_unfold_full_without_alpha_is_usage_error() -> None:
    assert (
        run_cli(
            [
                "unfold",
                "--data",
                "d.root",
                "--calib",
                "c.root",
                "--sim",
                "s.root",
                "--energy-low",
                "30",
                "--energy-high",
                "1500",
            ]
        )
        == 2
    )


def test_unfold_calib_only_dry_run(capsys: Any) -> None:
    code = run_cli(
        [
            "unfold",
            "--data",
            "d.root",
            "--calib",
            "c.root",
            "--calib-only",
            "--dry-run",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "calib_only" in out
    assert "d.root" in out


def test_sim_source_dry_run_prints_default_output(capsys: Any) -> None:
    code = run_cli(["sim", "--am241", "-n", "10", "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "out/sim" in out
    assert "am241-n10-s908136382.root" in out


def test_sim_matrix_dry_run_uses_cli_mode_token(capsys: Any) -> None:
    code = run_cli(["sim", "--plane-front-gamma", "calib.root", "-n", "5", "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "calib-plane-front-gamma-n5-s908136382.root" in out


def test_config_rejects_run_options() -> None:
    assert run_cli(["sim", "-c", str(EXAMPLES / "sim.toml"), "--am241"]) == 2
    assert run_cli(["calib", "-c", str(EXAMPLES / "calib.toml"), "--data", "x"]) == 2
    assert run_cli(["calib", "-c", str(EXAMPLES / "calib.toml"), "--no-plot"]) == 2
    assert run_cli(["unfold", "-c", str(EXAMPLES / "unfold.toml"), "--alpha", "1"]) == 2


def test_default_output_convention() -> None:
    assert default_output("sim", "x.root") == REPO_ROOT / "out" / "sim" / "x.root"
    assert default_output("calib", "calib-a.root") == (
        REPO_ROOT / "out" / "calib" / "calib-a.root"
    )


def test_csv2root_success_refusal_and_force(tmp_path: Path) -> None:
    output = tmp_path / "small.root"
    assert run_cli(["csv2root", str(SMALL_CSV), "-o", str(output)]) == 0
    assert output.is_file()
    # D-17: an existing target is refused without --force.
    assert run_cli(["csv2root", str(SMALL_CSV), "-o", str(output)]) == 1
    assert run_cli(["csv2root", str(SMALL_CSV), "-o", str(output), "--force"]) == 0


def test_csv2root_missing_input_is_runtime_failure(tmp_path: Path) -> None:
    assert run_cli(["csv2root", str(tmp_path / "absent.csv")]) == 1


def test_filesystem_error_maps_to_exit_one(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    assert (
        run_cli(["csv2root", str(SMALL_CSV), "-o", str(blocker / "out.root")]) == 1
    )
