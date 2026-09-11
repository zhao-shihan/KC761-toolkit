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
    assert "work/sim" in out
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
    assert default_output("sim", "x.root") == REPO_ROOT / "work" / "sim" / "x.root"
    assert default_output("calib", "calib-a.root") == (
        REPO_ROOT / "work" / "calib" / "calib-a.root"
    )


def test_csv2root_success_refusal_and_force(tmp_path: Path) -> None:
    output = tmp_path / "small.root"
    assert run_cli(["csv2root", str(SMALL_CSV), "-o", str(output)]) == 0
    assert output.is_file()
    # D-17/D-171: an existing target is refused before the input is read.
    assert run_cli(["csv2root", str(SMALL_CSV), "-o", str(output)]) == 2
    assert run_cli(["csv2root", str(SMALL_CSV), "-o", str(output), "--force"]) == 0


def test_csv2root_missing_input_is_runtime_failure(tmp_path: Path) -> None:
    assert run_cli(["csv2root", str(tmp_path / "absent.csv")]) == 1


def test_unusable_output_path_maps_to_exit_two(tmp_path: Path) -> None:
    """D-171: an unusable output path fails before the run (UsageError, 2).

    Runtime filesystem failures after the run starts still map to exit 1.
    """
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    assert (
        run_cli(["csv2root", str(SMALL_CSV), "-o", str(blocker / "out.root")]) == 2
    )


def test_unfold_snip_flags_parse(capsys: Any) -> None:
    """D-154/D-161: full snip-prefixed parameter surface on the CLI."""
    code = run_cli(
        [
            "unfold",
            "--data",
            "d.root",
            "--calib",
            "c.root",
            "--sim",
            "s.root",
            "--energy-low",
            "10",
            "--energy-high",
            "20",
            "--alpha",
            "0.1",
            "--no-snip",
            "--snip-threshold",
            "4.5",
            "--snip-protect",
            "1.5",
            "--snip-floor",
            "0.05",
            "--snip-iterations",
            "3",
            "--snip-max-iterations",
            "12",
            "--dry-run",
        ]
    )
    assert code == 0
    assert "unfold" in capsys.readouterr().out.lower()


def test_calib_config_validates_output_before_reading_inputs(tmp_path: Any) -> None:
    """D-168: the overwrite check runs before the input products are opened."""
    target = tmp_path / "calib.root"
    target.write_text("existing")
    config = tmp_path / "calib.toml"
    config.write_text(
        "config_version = 1\n[calib]\n"
        f'output = "{target}"\n'
        "[[calib.datasets]]\n"
        'data = "missing-data.root"\nmc = "missing-mc.root"\nlabel = "x"\n'
        "channel_low = 0\nchannel_high = 10\n",
        encoding="utf-8",
    )
    assert run_cli(["calib", "-c", str(config)]) == 2


def test_calib_progress_flags_parse(capsys: Any) -> None:
    code = run_cli(
        [
            "calib",
            "--data",
            "d.root",
            "--mc",
            "m.root",
            "--label",
            "x",
            "--channel-low",
            "0",
            "--channel-high",
            "10",
            "--no-progress",
            "--progress-every",
            "2",
            "--dry-run",
        ]
    )
    assert code == 0
    assert "dataset x" in capsys.readouterr().out


def _existing(tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    target.write_text("existing")
    return target


def test_compose_validates_output_before_reading_inputs(tmp_path: Path) -> None:
    target = _existing(tmp_path, "c.root")
    assert (
        run_cli(
            [
                "compose",
                "--calib",
                "missing-c.root",
                "--sim",
                "missing-s.root",
                "-o",
                str(target),
            ]
        )
        == 2
    )


def test_sim_validates_output_before_starting(tmp_path: Path) -> None:
    target = _existing(tmp_path, "s.root")
    assert (
        run_cli(["sim", "--am241", "-n", "10", "-o", str(target)]) == 2
    )


def test_csv2root_validates_output_before_reading(tmp_path: Path) -> None:
    target = _existing(tmp_path, "x.root")
    assert run_cli(["csv2root", str(SMALL_CSV), "-o", str(target)]) == 2


def test_subbkg_validates_output_before_reading(tmp_path: Path) -> None:
    target = _existing(tmp_path, "n.root")
    assert (
        run_cli(
            [
                "subbkg",
                "--signal",
                "missing-s.root",
                "--background",
                "missing-b.root",
                "-o",
                str(target),
            ]
        )
        == 2
    )


def test_unfold_validates_output_before_reading_inputs(tmp_path: Path) -> None:
    target = _existing(tmp_path, "u.root")
    assert (
        run_cli(
            [
                "unfold",
                "--data",
                "missing-d.root",
                "--calib",
                "missing-c.root",
                "--sim",
                "missing-s.root",
                "--energy-low",
                "1",
                "--energy-high",
                "2",
                "--alpha",
                "0.1",
                "-o",
                str(target),
            ]
        )
        == 2
    )


def test_unfold_config_validates_output_before_reading_inputs(tmp_path: Path) -> None:
    target = _existing(tmp_path, "u.root")
    config = tmp_path / "u.toml"
    config.write_text(
        "config_version = 1\n[unfold]\n"
        'data = "missing-d.root"\ncalib = "missing-c.root"\nsim = "missing-s.root"\n'
        "energy_low = 1.0\nenergy_high = 2.0\nalpha = 0.1\n"
        f'output = "{target}"\n',
        encoding="utf-8",
    )
    assert run_cli(["unfold", "-c", str(config)]) == 2


def test_unfold_validates_figure_target_before_inputs(tmp_path: Path) -> None:
    product = tmp_path / "u.root"  # absent
    (tmp_path / "u.pdf").write_text("existing figure")
    assert (
        run_cli(
            [
                "unfold",
                "--data",
                "missing-d.root",
                "--calib",
                "missing-c.root",
                "--sim",
                "missing-s.root",
                "--energy-low",
                "1",
                "--energy-high",
                "2",
                "--alpha",
                "0.1",
                "-o",
                str(product),
            ]
        )
        == 2
    )
