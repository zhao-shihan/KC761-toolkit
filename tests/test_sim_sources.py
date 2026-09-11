"""No-Geant4 tests: source registry, geometry provenance and schedules."""

from __future__ import annotations

import numpy as np
import pytest

from kc761 import errors
from kc761.core import binning
from kc761.sim import geometry, sources
from kc761.sim.detector import build_plane_gamma_source, build_sphere_gamma_source
from kc761.sim.generator import column_seed


def test_source_registry_matches_the_seven_frozen_keys() -> None:
    assert sources.SOURCE_KEYS == (
        "k40",
        "lu176",
        "am241",
        "th232",
        "th232-unshielded",
        "ra226",
        "ra226-unshielded",
    )
    assert tuple(sources.SOURCES) == sources.SOURCE_KEYS
    for key, spec in sources.SOURCES.items():
        assert spec.key == key
        assert spec.provenance
        assert spec.geometry_param_mm > 0.0
        assert spec.geometry_name in {
            "box",
            "cylinder",
            "disk",
            "sandwich",
            "sphere",
            "ellipsoid",
        }


def test_get_source_rejects_unknown_keys() -> None:
    with pytest.raises(errors.ValidationError, match="unknown source key"):
        sources.get_source("nope")


def test_ra226_variants_share_the_base_geometry() -> None:
    shielded = sources.SOURCES["ra226"]
    unshielded = sources.SOURCES["ra226-unshielded"]
    assert shielded.geometry == unshielded.geometry
    assert shielded.container == unshielded.container
    assert shielded.shield is not None
    assert unshielded.shield is None


def test_geometry_provenance_covers_every_field() -> None:
    fields = set(geometry.DetectorGeometry.__dataclass_fields__)
    assert fields == set(geometry.geometry_provenance())
    assert geometry.DEFAULT_GEOMETRY.detector_front_z_mm == 13.7
    assert sources.BETA_SHIELD.lowest_z == geometry.DEFAULT_GEOMETRY.detector_front_z_mm


def test_geometry_rejects_non_positive_values() -> None:
    with pytest.raises(errors.ValidationError, match="must be positive"):
        geometry.DetectorGeometry(crystal_half_x_mm=0.0)


def test_material_provenance_covers_all_custom_materials() -> None:
    from kc761.sim import materials

    for name in materials.CUSTOM_MATERIAL_ATOMS:
        assert name in materials.MATERIAL_PROVENANCE
    for name in ("CsI_Tl", "ABS", "R4600"):
        assert name in materials.MATERIAL_PROVENANCE
    fractions = materials.csi_tl_mass_fractions()
    assert abs(sum(fractions.values()) - 1.0) < 1e-12


def test_plane_and_sphere_sources_derive_from_geometry() -> None:
    plane = build_plane_gamma_source()
    assert plane.size_x_mm == 2.0 * geometry.DEFAULT_GEOMETRY.crystal_half_x_mm
    assert plane.z_mm == geometry.DEFAULT_GEOMETRY.detector_front_z_mm
    sphere = build_sphere_gamma_source()
    expected = np.sqrt(
        geometry.DEFAULT_GEOMETRY.housing_half_x_mm**2
        + geometry.DEFAULT_GEOMETRY.housing_half_y_mm**2
        + geometry.DEFAULT_GEOMETRY.housing_half_z_mm**2
    )
    assert np.isclose(sphere.radius_mm, expected)


def test_matrix_primary_axis_is_calibration_derived() -> None:
    # D-121 revised: the matrix primary edges are the calibration C.y edges
    # (non-uniform), so no fixed source-mode default axis remains.
    edges = np.array([0.0, 1.0, 1.5, 2.5, 5.0])
    axis = sources.make_primary_axis(edges)
    assert axis.n_columns == edges.size - 1
    assert np.array_equal(np.asarray(axis.edges_kev), edges)
    assert not hasattr(sources, "default_primary_axis")


def test_fixed_axis_has_a_single_source_shared_with_calib() -> None:
    from kc761.calib import model as calib_model

    assert calib_model.FIXED_DEPOSITION_BINS is binning.SOURCE_MODE_DEPOSITION_BINS
    assert calib_model.FIXED_DEPOSITION_MAX_KEV is binning.SOURCE_MODE_DEPOSITION_MAX_KEV
    assert (
        calib_model.fixed_deposition_edges_kev
        is binning.source_mode_deposition_edges_kev
    )


def test_primary_axis_sampling_and_validation() -> None:
    axis = sources.make_primary_axis([0.0, 1.0, 3.0])
    assert axis.active == (0, 1)
    assert axis.energy_bounds_kev(1) == (1.0, 3.0)
    assert axis.sample_energy_kev(1, 0.0) == 1.0
    assert axis.sample_energy_kev(1, 0.5) == 2.0
    with pytest.raises(errors.ValidationError, match=r"\[0, 1\)"):
        axis.sample_energy_kev(1, 1.0)
    inactive = sources.make_primary_axis([-2.0, -1.0, 0.0, 1.0])
    with pytest.raises(errors.ValidationError, match="not active"):
        inactive.energy_bounds_kev(0)
    with pytest.raises(errors.ValidationError, match="not strictly increasing"):
        sources.make_primary_axis([0.0, 2.0, 1.0])


def test_primary_axis_requires_an_active_column() -> None:
    with pytest.raises(errors.ValidationError, match="no active column"):
        sources.make_primary_axis([-2.0, -1.0])


def test_column_schedule_fixed_total_remainder_rule() -> None:
    axis = sources.make_primary_axis([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert axis.active == (2, 3)
    schedule = sources.ColumnSchedule.fixed_total(axis, 5)
    assert schedule.counts == (0, 0, 3, 2)
    assert schedule.total_events == 5
    assert dict(schedule.column_counts()) == {2: 3, 3: 2}


def test_column_schedule_validates_and_rejects_bad_counts() -> None:
    axis = sources.make_primary_axis([0.0, 1.0, 2.0])
    with pytest.raises(errors.ValidationError, match="negative"):
        sources.ColumnSchedule(axis=axis, counts=(-1, 0))
    inactive = sources.make_primary_axis([-2.0, -1.0, 0.0, 1.0])
    with pytest.raises(errors.ValidationError, match="inactive column"):
        sources.ColumnSchedule(axis=inactive, counts=(1, 0, 0))


def test_slices_partition_columns_and_locate_events() -> None:
    axis = sources.make_primary_axis([0.0, 1.0, 2.0, 3.0, 4.0])
    schedule = sources.ColumnSchedule.fixed_total(axis, 10)
    one = schedule.slices(1)
    assert len(one) == 1
    assert one[0].columns == (0, 1, 2, 3)
    assert one[0].locate(0) == (0, 0)
    assert one[0].locate(2) == (0, 2)
    assert one[0].locate(3) == (1, 0)
    many = schedule.slices(3)
    assert [column for chunk in many for column in chunk.columns] == [0, 1, 2, 3]
    assert sum(chunk.total_events for chunk in many) == 10


def test_column_and_block_seeds_are_stable_distinct_and_in_range() -> None:
    from kc761.sim.generator import block_seed, splitmix64

    assert splitmix64(0) == splitmix64(0)
    assert column_seed(908136382, 0) == column_seed(908136382, 0)
    assert column_seed(908136382, 0) != column_seed(908136382, 1)
    assert block_seed(908136382, 0) != column_seed(908136382, 0)
    for column in range(32):
        seed = column_seed(908136382, column)
        assert 0 <= seed < 1 << 63
    with pytest.raises(errors.ValidationError, match="base seed"):
        column_seed(-1, 0)
    with pytest.raises(errors.ValidationError, match="non-negative"):
        column_seed(1, -1)


def test_mode_metadata_labels() -> None:
    plane = build_plane_gamma_source()
    sphere = build_sphere_gamma_source()
    assert sources.mode_metadata(plane) == (1, "plane-front-gamma", "plane", plane.z_mm)
    assert sources.mode_metadata(sphere) == (
        2,
        "sphere-gamma",
        "sphere",
        sphere.radius_mm,
    )


def test_every_source_material_is_resolvable() -> None:
    """Ra226/Th232 shields use the prebuilt R4600 material.

    Without this invariant the shielded sources failed at Geant4 build time
    with "unknown custom source material 'R4600'".
    """
    from kc761.sim import materials

    def material_names(spec: sources.SourceSpec):
        yield spec.material
        geometry = spec.geometry
        if hasattr(geometry, "layers"):
            for layer in geometry.layers:
                yield layer.material
        if spec.container is not None:
            yield spec.container.material
        if spec.shield is not None:
            yield spec.shield.material

    for key, spec in sources.SOURCES.items():
        for name in material_names(spec):
            assert (
                name.startswith("G4_")
                or name in materials.PREBUILT_MATERIALS
                or name in materials.CUSTOM_MATERIAL_ATOMS
            ), f"{key}: unresolved material {name!r}"
