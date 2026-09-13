"""Declarative CLI option surface: structural invariants and D-130 enforcement.

These tests are the point of D-190: the option declaration is the single source
for the parser, the TOML schema, the defaults and the validation, so the checks
here are about *structure* (no drift is representable) rather than about
individual messages.
"""

from __future__ import annotations

import argparse

import pytest

from kc761tool.cli import calib, compose, csv2root, sim, specadd, specsub, unfold
from kc761tool.cli._registry import (
    Kind,
    Requirement,
    Scope,
    add_run_options,
    config_keys,
    nested_config_keys,
    nested_policy,
    options_by_dest,
    provided_options,
    validate_policy,
)
from kc761tool.errors import UsageError

POLICIES = {
    "csv2root": csv2root.CSV2ROOT_POLICY,
    "specadd": specadd.SPECADD_POLICY,
    "specsub": specsub.SPECSUB_POLICY,
    "sim": sim.SIM_POLICY,
    "calib": calib.CALIB_POLICY,
    "compose": compose.COMPOSE_POLICY,
    "unfold": unfold.UNFOLD_POLICY,
}
CONFIG_POLICIES = {name: policy for name, policy in POLICIES.items() if policy.config}

_ALL_POLICIES = sorted(POLICIES.items())


def test_every_policy_declaration_is_structurally_sound() -> None:
    """Unique flags/config keys, CONST groups consistent, run options spelled."""
    for name, policy in _ALL_POLICIES:
        validate_policy(policy)  # raises AssertionError on a malformed declaration
        assert policy.command == name


def test_parser_round_trips_every_run_option() -> None:
    """Every RUN option can be supplied and is then *structurally* present.

    The parser is built from the declaration and a synthetic argv is parsed per
    option, so an option that argparse cannot express (duplicate dest, wrong
    action, missing metavar) fails here rather than in the field.
    """
    samples = {
        Kind.PATH: "x.root",
        Kind.STRING: "value",
        Kind.INT: "3",
        Kind.FLOAT: "1.5",
        Kind.PATH_LIST: "x.root",
        Kind.FLOAT_LIST: "0.5",
        Kind.STRING_LIST: "label",
        Kind.COUNT: None,
        Kind.BOOL: None,
        Kind.BOOL_PAIR: None,
        Kind.CONST: None,
    }
    for name, policy in _ALL_POLICIES:
        parser = argparse.ArgumentParser(prog=name)
        add_run_options(parser, policy)
        checked = 0
        for option in policy.spec:
            if option.scope is not Scope.RUN or not option.flags:
                continue
            flag = next(f for f in option.flags if f.startswith("--"))
            sample = samples[option.kind]
            argv = [flag] if sample is None else [flag, sample]
            args = parser.parse_args(argv)
            # Presence is the structural fact this test pins; requirement and
            # bound rules are exercised by the command-level tests below.
            assert option.dest in provided_options(policy, args), (name, option.dest, argv)
            checked += 1
        assert checked > 0


def test_config_keys_are_exactly_the_loader_schema() -> None:
    """The loaders' key sets come from the policy, so they cannot drift."""
    assert sorted(config_keys(compose.COMPOSE_POLICY)) == [
        "calib",
        "dry_run",
        "force",
        "output",
        "sim",
    ]
    assert sorted(config_keys(calib.CALIB_POLICY)) == ["dry_run", "force", "no_plot", "output"]
    assert sorted(nested_config_keys(calib.CALIB_POLICY)) == [
        "channel_high",
        "channel_low",
        "data",
        "label",
        "mc",
        "syst_frac",
    ]
    assert sorted(config_keys(sim.SIM_POLICY)) == ["dry_run", "force", "resume"]
    assert sorted(nested_config_keys(sim.SIM_POLICY)) == [
        "calib",
        "events",
        "mode",
        "output",
        "seed",
        "source",
        "threads",
        "verbose",
    ]
    expected_unfold = {
        "alpha",
        "calib",
        "calib_only",
        "data",
        "difference_order",
        "dry_run",
        "energy_high",
        "energy_low",
        "force",
        "no_plot",
        "output",
        "sim",
        "snip_enabled",
        "snip_floor",
        "snip_iterations",
        "snip_max_iterations",
        "snip_protect_bins",
        "snip_threshold",
        "syst_frac",
    }
    assert set(config_keys(unfold.UNFOLD_POLICY)) == expected_unfold


def test_config_mapping_covers_every_declared_key() -> None:
    """``<config>.values()`` exposes exactly the dests of its declared keys.

    A key the loader accepts but does not map would silently fall back to the
    declaration default, which is exactly the drift D-190 removes.
    """
    import tempfile
    from pathlib import Path

    from kc761tool.cli.config import (
        load_calib_config,
        load_compose_config,
        load_sim_config,
        load_unfold_config,
    )

    text = {
        "compose": '[compose]\ncalib = "c.root"\nsim = "s.root"\n',
        "calib": ('[calib]\n[[calib.datasets]]\ndata = "d.root"\nmc = "m.root"\nlabel = "x"\n'),
        "unfold": (
            '[unfold]\ndata = "d.root"\ncalib = "c.root"\nsim = "s.root"\n'
            "alpha = 1.0\nenergy_low = 1.0\nenergy_high = 2.0\n"
        ),
        "sim": '[[sim.runs]]\nsource = "am241"\nevents = 10\n',
    }
    loaders = {
        "compose": load_compose_config,
        "calib": load_calib_config,
        "unfold": load_unfold_config,
        "sim": load_sim_config,
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, policy in CONFIG_POLICIES.items():
            path = Path(tmp) / f"{name}.toml"
            path.write_text(f"config_version = 1\n{text[name]}", encoding="utf-8")
            config = loaders[name](path)
            values = config.values()
            declared = {option.dest for option in config_keys(policy).values()}
            assert set(values) == declared, name


@pytest.mark.parametrize(
    ("name", "flag"),
    (
        # D-190: these used to be silently ignored together with --config,
        # because the mirror table either missed them entirely (the whole SNIP
        # family) or compared them against their own default value.
        ("unfold", ["--syst-frac", "0.05"]),
        ("unfold", ["--difference-order", "2"]),
        ("unfold", ["--snip-floor", "0.1"]),
        ("unfold", ["--snip"]),
        ("unfold", ["--snip-max-iterations", "8"]),
        ("sim", ["--seed", "908136382"]),
        ("sim", ["--verbose"]),
        ("calib", ["--max-iter", "5"]),
        ("calib", ["--progress-every", "1"]),
    ),
)
def test_default_valued_run_options_are_rejected_with_config(name: str, flag: list[str]) -> None:
    """The blind spot of the old value comparison is gone (D-190)."""
    from kc761tool.cli import main

    example = {"unfold": "unfold.toml", "sim": "sim.toml", "calib": "calib.toml"}[name]
    from kc761tool.cli._common import REPO_ROOT

    code = main([name, "-c", str(REPO_ROOT / "examples" / example), *flag])
    assert code == 2


def test_global_options_may_still_accompany_config() -> None:
    """D-130: the runtime options stay allowed with ``--config`` (dry run)."""
    from kc761tool.cli import main
    from kc761tool.cli._common import REPO_ROOT

    code = main(
        [
            "unfold",
            "-c",
            str(REPO_ROOT / "examples" / "unfold.toml"),
            "--dry-run",
            "--no-progress" if False else "--strict",
        ]
    )
    assert code == 0


def test_dry_run_and_a_real_run_share_validation() -> None:
    """Both paths call the same resolver, so ``--dry-run`` cannot accept more."""
    from kc761tool.cli import main

    common = [
        "unfold",
        "--data",
        "d.root",
        "--calib",
        "c.root",
        "--sim",
        "s.root",
        "--elo",
        "10",
        "--ehi",
        "20",
        "--alpha",
        "0.1",
    ]
    assert main([*common, "--snip-protect-bins", "0", "--dry-run"]) == 2
    assert main([*common, "--snip-protect-bins", "0"]) == 2
    assert main([*common, "--snip-protect", "2", "--dry-run"]) == 2
    assert main([*common, "--snip-protect", "2"]) == 2


def test_retired_spelling_reports_the_replacement() -> None:
    """The retired flag is a declared entry, not an argv heuristic (D-188/D-190)."""
    policy = unfold.UNFOLD_POLICY
    retired = options_by_dest(policy)["snip_protect_retired"]
    assert retired.retired is not None and "snip-protect-bins" in retired.retired
    assert retired.is_hidden


def test_nested_policy_keeps_only_declared_nested_rules() -> None:
    """A per-dataset/per-run check must not inherit flat-only requirements."""
    nested = nested_policy(calib.CALIB_POLICY)
    requirements = {option.dest: option.requirement for option in nested.spec}
    assert requirements["dataset_data"] is Requirement.ALWAYS
    assert requirements["channel_low"] is Requirement.ARGS
    assert "data" not in requirements  # the flat repeatable option is not nested
    assert "syst_frac" not in requirements


def test_invalid_values_are_rejected_by_the_shared_checker() -> None:
    """One bound, two surfaces: the same rule fires on the CLI and in the TOML."""
    from kc761tool.cli._registry import check_policy

    policy = unfold.UNFOLD_POLICY
    values: dict[str, object] = {option.dest: option.default for option in policy.spec}
    values.update(
        {
            "data": "d.root",
            "calib": "c.root",
            "sim": "s.root",
            "energy_low": 1.0,
            "energy_high": 2.0,
            "alpha": -1.0,
        }
    )
    with pytest.raises(UsageError, match="alpha"):
        check_policy(policy, values, config_mode=False)


def test_config_surface_enforces_the_same_bounds_as_the_command_line(tmp_path) -> None:
    """D-190: a bound declared for an option guards its TOML carrier too.

    The calibration systematic is repeatable on the command line and a scalar in
    ``[[calib.datasets]]``; before this pin only the command-line spelling was
    checked, so a negative value passed ``--dry-run`` in config mode while the
    same value exited 2 on the command line.
    """
    from kc761tool.cli import main

    config = tmp_path / "calib.toml"
    config.write_text(
        "config_version = 1\n[calib]\n[[calib.datasets]]\n"
        'data = "d.root"\nmc = "m.root"\nlabel = "x"\nsyst_frac = -0.5\n',
        encoding="utf-8",
    )
    assert main(["calib", "-c", str(config), "--dry-run"]) == 2
    config.write_text(
        "config_version = 1\n[calib]\n[[calib.datasets]]\n"
        'data = "d.root"\nmc = "m.root"\nlabel = "x"\nsyst_frac = 0.0\n',
        encoding="utf-8",
    )
    assert main(["calib", "-c", str(config), "--dry-run"]) == 0


def test_retired_toml_keys_report_their_replacement() -> None:
    """Both surfaces point at the replacement (D-188/D-190)."""
    import tempfile
    from pathlib import Path

    from kc761tool.cli.config import load_unfold_config

    body = (
        'data = "d.root"\ncalib = "c.root"\nsim = "s.root"\n'
        "alpha = 1.0\nenergy_low = 1.0\nenergy_high = 2.0\n"
    )
    expectations = {
        "pad_nsigma": "removed in D-187",
        "snip_protect": "renamed to 'snip_protect_bins'",
        "snip_protect_sigma": "renamed to 'snip_protect_bins'",
    }
    with tempfile.TemporaryDirectory() as tmp:
        for key, hint in expectations.items():
            path = Path(tmp) / f"{key}.toml"
            path.write_text(f"config_version = 1\n[unfold]\n{body}{key} = 2.0\n", encoding="utf-8")
            with pytest.raises(UsageError, match=hint):
                load_unfold_config(path)


def test_report_states_the_window_edge_limitation() -> None:
    """D-187: the text report names the boundary-layer caveat, not only the docs."""
    from kc761tool.unfold import run_unfold
    from tests.test_unfold_support import (
        make_calib_product,
        make_sim_product,
        make_spectrum_product,
        response_of,
        truth_vector,
    )

    calib = make_calib_product()
    sim = make_sim_product(calib)
    signal = response_of(calib, sim) @ truth_vector(
        (25, 30, 35, 40, 45, 50), (500.0, 800.0, 1200.0, 700.0, 400.0, 300.0)
    )
    result = run_unfold(
        make_spectrum_product(signal),
        calib,
        sim,
        snip_enabled=False,
        energy_low_kev=200.0,
        energy_high_kev=560.0,
        alpha=1e-3,
        strict=True,
        plot=False,
    )
    assert "leakage-limited" in result.report
    assert "D-187" in result.report
