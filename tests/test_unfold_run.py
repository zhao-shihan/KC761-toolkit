"""``run_unfold`` orchestration: full, calib-only, errors, report and plot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kc761tool.errors import SchemaError, UsageError, ValidationError
from kc761tool.schema.axes import channel_axis
from kc761tool.schema.io import input_sha256, read_product, sha256_file
from kc761tool.schema.products import (
    UNFOLD_MODE_CALIB_ONLY,
    UNFOLD_MODE_FULL,
    UNFOLD_SETTING_TYPES,
    Histogram1D,
    SpectrumProduct,
)
from kc761tool.unfold import run_unfold
from tests.test_unfold_support import (
    N_CHANNELS,
    SCHEMA_VERSION,
    make_calib_product,
    make_sim_product,
    make_spectrum_product,
    response_of,
    synthetic_provenance,
    truth_vector,
    write,
)

TRUTH_INDICES = (25, 30, 35, 40, 45, 50)
TRUTH_AMPLITUDES = (500.0, 800.0, 1200.0, 700.0, 400.0, 300.0)
WINDOW = (220.0, 520.0)
ALPHA = 1e-3


def _fixture(tmp_path: Path):
    calib = make_calib_product()
    sim = make_sim_product(calib)
    signal = response_of(calib, sim) @ truth_vector(TRUTH_INDICES, TRUTH_AMPLITUDES)
    data = make_spectrum_product(signal)
    return calib, sim, data


def test_full_product_roundtrip_settings_and_provenance(tmp_path: Path) -> None:
    calib, sim, data = _fixture(tmp_path)
    calib_path = write(tmp_path / "calib.root", calib)
    sim_path = write(tmp_path / "sim.root", sim)
    data_path = write(tmp_path / "data.root", data)
    output = tmp_path / "unfold.root"
    result = run_unfold(
        data_path,
        calib_path,
        sim_path,
        snip_enabled=False,
        energy_low_kev=WINDOW[0],
        energy_high_kev=WINDOW[1],
        alpha=ALPHA,
        output=output,
        strict=True,
        plot=False,
    )
    assert result.product_path == output
    product = read_product(output, strict=True)
    assert product.mode == UNFOLD_MODE_FULL
    assert dict(product.settings).keys() == set(UNFOLD_SETTING_TYPES)
    assert len(product.settings) == len(UNFOLD_SETTING_TYPES)
    assert product.sigma_statistical is not None
    assert product.sigma_total is not None
    assert np.allclose(product.sigma_total.variances, product.sigma_total.values**2)
    for path in (data_path, calib_path, sim_path):
        assert input_sha256(product.provenance, path) == sha256_file(path)


def test_calib_only_relabels_axis_without_touching_counts(tmp_path: Path) -> None:
    calib, _, data = _fixture(tmp_path)
    calib_path = write(tmp_path / "calib.root", calib)
    data_path = write(tmp_path / "data.root", data)
    output = tmp_path / "calib_only.root"
    result = run_unfold(
        data_path,
        calib_path,
        snip_enabled=False,
        energy_low_kev=WINDOW[0],
        energy_high_kev=WINDOW[1],
        calib_only=True,
        output=output,
        strict=True,
        plot=False,
    )
    assert result.mode == UNFOLD_MODE_CALIB_ONLY
    product = read_product(output, strict=True)
    assert product.mode == UNFOLD_MODE_CALIB_ONLY
    assert product.settings == ()
    assert product.sigma_total is None
    assert product.spectrum.axis.unit == "kev"
    assert np.array_equal(product.spectrum.axis.edges, calib.deposition_to_channel.y.edges)
    assert np.array_equal(product.spectrum.values, np.asarray(data.spectrum.values))
    assert np.array_equal(product.spectrum.variances, np.asarray(data.spectrum.variances))


def test_calib_only_rejects_mismatched_channel_axis(tmp_path: Path) -> None:
    calib, _, data = _fixture(tmp_path)
    calib_path = write(tmp_path / "calib.root", calib)
    wrong = SpectrumProduct(
        format_version=SCHEMA_VERSION,
        spectrum=Histogram1D(
            axis=channel_axis(N_CHANNELS + 1),
            values=np.ones(N_CHANNELS + 1),
            variances=np.ones(N_CHANNELS + 1),
        ),
        daq_time_s=1.0,
        source_file="wrong.csv",
        provenance=synthetic_provenance(),
    )
    wrong_path = write(tmp_path / "wrong.root", wrong)
    with pytest.raises(ValidationError, match="channel axis mismatch"):
        run_unfold(
            wrong_path,
            calib_path,
            snip_enabled=False,
            energy_low_kev=WINDOW[0],
            energy_high_kev=WINDOW[1],
            calib_only=True,
            plot=False,
        )


def test_full_unfold_defaults_alpha_and_requires_sim(tmp_path: Path) -> None:
    """D-191: the library defaults are the named constants; ``--sim`` stays mandatory."""
    from kc761tool.core.solver import (
        DEFAULT_ALPHA,
        DEFAULT_SNIP_FLOOR,
        DEFAULT_SNIP_MAX_ITERATIONS,
    )
    from kc761tool.unfold.types import UnfoldSettings

    calib, sim, data = _fixture(tmp_path)
    calib_path = write(tmp_path / "calib.root", calib)
    sim_path = write(tmp_path / "sim.root", sim)
    data_path = write(tmp_path / "data.root", data)
    result = run_unfold(
        data_path,
        calib_path,
        sim_path,
        snip_enabled=False,
        energy_low_kev=WINDOW[0],
        energy_high_kev=WINDOW[1],
        plot=False,
    )
    assert result.settings is not None
    assert result.settings.alpha == DEFAULT_ALPHA
    assert result.settings.regularization().alpha == DEFAULT_ALPHA
    assert result.settings.snip_floor == DEFAULT_SNIP_FLOOR
    assert result.settings.snip_max_iterations == DEFAULT_SNIP_MAX_ITERATIONS
    # The settings container carries the same defaults as ``run_unfold``.
    settings = UnfoldSettings(energy_low_kev=WINDOW[0], energy_high_kev=WINDOW[1])
    assert settings.alpha == DEFAULT_ALPHA
    assert settings.snip_floor == DEFAULT_SNIP_FLOOR
    assert settings.snip_max_iterations == DEFAULT_SNIP_MAX_ITERATIONS
    # An explicit None is still rejected: no silent substitution.
    with pytest.raises(ValidationError, match="alpha"):
        run_unfold(
            data_path,
            calib_path,
            sim_path,
            snip_enabled=False,
            energy_low_kev=WINDOW[0],
            energy_high_kev=WINDOW[1],
            alpha=None,
            plot=False,
        )
    with pytest.raises(ValidationError, match="sim is required"):
        run_unfold(
            data_path,
            calib_path,
            snip_enabled=False,
            energy_low_kev=WINDOW[0],
            energy_high_kev=WINDOW[1],
            alpha=ALPHA,
            plot=False,
        )


def test_window_outside_primary_axis_is_rejected(tmp_path: Path) -> None:
    calib, sim, data = _fixture(tmp_path)
    calib_path = write(tmp_path / "calib.root", calib)
    sim_path = write(tmp_path / "sim.root", sim)
    data_path = write(tmp_path / "data.root", data)
    with pytest.raises(ValidationError, match="outside"):
        run_unfold(
            data_path,
            calib_path,
            sim_path,
            snip_enabled=False,
            energy_low_kev=-10.0,
            energy_high_kev=200.0,
            alpha=ALPHA,
            plot=False,
        )


def test_wrong_data_product_kind_is_rejected(tmp_path: Path) -> None:
    calib, sim, _ = _fixture(tmp_path)
    calib_path = write(tmp_path / "calib.root", calib)
    sim_path = write(tmp_path / "sim.root", sim)
    with pytest.raises(SchemaError, match="expected a spectrum product"):
        run_unfold(
            sim_path,
            calib_path,
            sim_path,
            snip_enabled=False,
            energy_low_kev=WINDOW[0],
            energy_high_kev=WINDOW[1],
            alpha=ALPHA,
            plot=False,
        )


def test_report_and_plot_are_written_and_plot_refuses_overwrite(tmp_path: Path) -> None:
    calib, sim, data = _fixture(tmp_path)
    calib_path = write(tmp_path / "calib.root", calib)
    sim_path = write(tmp_path / "sim.root", sim)
    data_path = write(tmp_path / "data.root", data)
    output = tmp_path / "unfold.root"
    result = run_unfold(
        data_path,
        calib_path,
        sim_path,
        snip_enabled=False,
        energy_low_kev=WINDOW[0],
        energy_high_kev=WINDOW[1],
        alpha=ALPHA,
        output=output,
        strict=True,
        plot=True,
    )
    assert "KC761 unfold" in result.report
    assert "KKT" in result.report
    assert result.plot_path is not None and result.plot_path.is_file()
    with pytest.raises(UsageError, match="refusing to overwrite"):
        run_unfold(
            data_path,
            calib_path,
            sim_path,
            snip_enabled=False,
            energy_low_kev=WINDOW[0],
            energy_high_kev=WINDOW[1],
            alpha=ALPHA,
            output=output,
            force=True,
            strict=True,
            plot=True,
        )


def test_unfold_validates_output_before_reading_inputs(tmp_path: Path) -> None:
    """D-171: the overwrite check runs before any product is opened."""
    target = tmp_path / "u.root"
    target.write_text("existing")
    with pytest.raises(UsageError, match="refusing to overwrite"):
        run_unfold(
            "missing-d.root",
            "missing-c.root",
            "missing-s.root",
            energy_low_kev=1.0,
            energy_high_kev=2.0,
            alpha=0.1,
            output=target,
            plot=False,
        )


def test_unfold_validates_figure_target_before_inputs(tmp_path: Path) -> None:
    product = tmp_path / "u.root"
    (tmp_path / "u.pdf").write_text("existing figure")
    with pytest.raises(UsageError, match="refusing to overwrite"):
        run_unfold(
            "missing-d.root",
            "missing-c.root",
            "missing-s.root",
            energy_low_kev=1.0,
            energy_high_kev=2.0,
            alpha=0.1,
            output=product,
            plot=True,
        )


def _figure_summary(plot_module: Any, tmp_path: Any, result: Any, **kwargs: Any) -> dict[str, Any]:
    """Render ``result`` and report panel titles, scales and figure size (D-193)."""
    captured: dict[str, Any] = {}
    original = plot_module._save_fig

    def spy(fig, out_plot, force):
        captured.update(
            titles=[ax.get_title() for ax in fig.axes],
            x=[ax.get_xscale() for ax in fig.axes],
            y=[ax.get_yscale() for ax in fig.axes],
            size=tuple(float(value) for value in fig.get_size_inches()),
        )
        return original(fig, out_plot, force)

    plot_module._save_fig = spy
    try:
        plot_module.plot_unfold(result, path=tmp_path / "u.pdf", force=True, **kwargs)
    finally:
        plot_module._save_fig = original
    return captured


def test_unfold_figure_panels_without_and_with_the_log_panel(tmp_path: Any) -> None:
    """D-189/D-193: linear x-axis, no log panel by default, exact opt-in geometry."""
    import kc761tool.unfold.plot as plot_module

    calib = make_calib_product()
    sim = make_sim_product(calib)
    signal = response_of(calib, sim) @ truth_vector(TRUTH_INDICES, TRUTH_AMPLITUDES)
    result = run_unfold(
        make_spectrum_product(signal),
        calib,
        sim,
        snip_enabled=False,
        energy_low_kev=WINDOW[0],
        energy_high_kev=WINDOW[1],
        alpha=ALPHA,
        strict=True,
        plot=False,
    )
    default = _figure_summary(plot_module, tmp_path, result)
    with_log = _figure_summary(plot_module, tmp_path, result, log_panel=True)
    assert default["titles"] == ["Spectrum", "Relative residuals"]
    assert default["y"] == ["linear", "linear"]
    assert default["size"] == pytest.approx((9.5, 6.3))  # 4.2 in spectrum + 2.1 in residuals
    assert with_log["titles"] == [
        "Spectrum",
        "Spectrum (log y-axis)",
        "Relative residuals",
    ]
    assert with_log["y"] == ["linear", "log", "linear"]
    # D-193: the opt-in figure reproduces the pre-D-193 three-panel geometry.
    assert with_log["size"] == pytest.approx((9.5, 10.5))  # 4.2 + 4.2 + 2.1 in
    assert all(x == "linear" for x in default["x"] + with_log["x"])


def test_run_unfold_default_figure_is_the_two_panel_one(tmp_path: Path, monkeypatch: Any) -> None:
    """D-193: the library default writes the log-free figure, like the CLI default."""
    from kc761tool.unfold import unfold as unfold_module

    calib, sim, data = _fixture(tmp_path)
    output = tmp_path / "u.root"
    calls: list[bool] = []
    original = unfold_module.plot_unfold

    def spy(result, *, path, force=False, log_panel=False):
        calls.append(log_panel)
        return original(result, path=path, force=force, log_panel=log_panel)

    monkeypatch.setattr(unfold_module, "plot_unfold", spy)
    run_unfold(
        data,
        calib,
        sim,
        snip_enabled=False,
        energy_low_kev=WINDOW[0],
        energy_high_kev=WINDOW[1],
        alpha=ALPHA,
        output=output,
    )
    assert calls == [False]
    assert output.with_suffix(".pdf").is_file()


def test_calib_only_figure_has_one_panel_by_default(tmp_path: Any) -> None:
    """D-193: ``calib_only`` carries the linear spectrum alone unless opted in."""
    import kc761tool.unfold.plot as plot_module

    calib, _, data = _fixture(tmp_path)
    result = run_unfold(
        data,
        calib,
        calib_only=True,
        energy_low_kev=WINDOW[0],
        energy_high_kev=WINDOW[1],
        plot=False,
    )
    default = _figure_summary(plot_module, tmp_path, result)
    with_log = _figure_summary(plot_module, tmp_path, result, log_panel=True)
    assert default["titles"] == ["Spectrum"]
    assert default["y"] == ["linear"]
    assert default["size"] == pytest.approx((9.5, 4.2))
    assert with_log["titles"] == ["Spectrum", "Spectrum (log y-axis)"]
    assert with_log["y"] == ["linear", "log"]
    assert with_log["size"] == pytest.approx((9.5, 8.4))
