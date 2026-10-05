"""Tests for gf.component_functions and the gf.components cells that wrap them."""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable, Iterator
from typing import Any

import pytest

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.gpdk import PDK
from gdsfactory.typings import CrossSectionSpec

MARKER = "DRC_MARKER"

# Parameters a gf.components cell fixes instead of passing through, by cell name.
FIXED_PARAMETERS: dict[str, set[str]] = {}

component_function_names = sorted(
    name
    for name in cf.__all__
    if inspect.isfunction(getattr(cf, name)) and name != "get_component"
)


@pytest.fixture
def restore_pdk() -> Iterator[None]:
    gf.clear_cache()
    try:
        yield
    finally:
        gf.clear_cache()
        PDK.activate(force=True)


def _activate_pdk(name: str, cells: dict[str, Callable[..., Any]]) -> None:
    gf.clear_cache()
    PDK.model_copy(update={"name": name, "cells": cells}).activate(force=True)


@gf.cell(basename="marked_straight", register_factory=False)
def marked_straight(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
) -> gf.Component:
    c = cf.straight(
        length=length, npoints=npoints, cross_section=cross_section, width=width
    )
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=MARKER)
    return c


@gf.cell(basename="marked_bezier", register_factory=False)
def marked_bezier(
    control_points: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (5.0, 0.0),
        (5.0, 1.8),
        (10.0, 1.8),
    ),
    npoints: int = 201,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    width: float | None = None,
) -> gf.Component:
    c = cf.bezier(
        control_points=control_points,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        width=width,
    )
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=MARKER)
    return c


def _has_marker(c: gf.Component) -> bool:
    return not c.kdb_cell.bbox(gf.get_layer(MARKER)).empty()


@pytest.mark.parametrize("name", component_function_names)
def test_cell_matches_component_function(name: str) -> None:
    """Each gf.components cell re-exposes its component function's arguments."""
    function = getattr(cf, name)
    cell = getattr(gf.components, name)
    fixed = FIXED_PARAMETERS.get(name, set())

    function_signature = inspect.signature(function)
    cell_signature = inspect.signature(cell)
    expected = [
        p for p in function_signature.parameters.values() if p.name not in fixed
    ]
    assert list(cell_signature.parameters.values()) == expected
    assert cell_signature.return_annotation == function_signature.return_annotation
    if not fixed:
        assert inspect.getdoc(cell) == inspect.getdoc(function)


def test_component_functions_are_not_cached() -> None:
    assert cf.straight() is not cf.straight()
    assert gf.components.straight() is gf.components.straight()


def test_component_functions_not_registered_as_pdk_cells() -> None:
    functions = {getattr(cf, name) for name in component_function_names}
    assert functions.isdisjoint(gf.get_active_pdk().cells.values())


def test_pdk_override_applies_inside_components(restore_pdk: None) -> None:
    assert not _has_marker(gf.components.mmi1x2())
    assert not _has_marker(gf.components.mzi())

    _activate_pdk("override_straight", {**PDK.cells, "straight": marked_straight})

    # mmi1x2 used to default to gf.components.straight, bypassing the PDK.
    assert _has_marker(gf.components.mmi1x2())
    assert _has_marker(gf.components.mzi())
    assert _has_marker(gf.components.straight_array())


def test_fallback_to_gf_components_warns(restore_pdk: None) -> None:
    _activate_pdk("missing_taper", {k: v for k, v in PDK.cells.items() if k != "taper"})

    with pytest.warns(cf.ComponentFallbackWarning, match="'taper' is not in PDK"):
        c = gf.components.mmi1x2()
    assert {p.name for p in c.ports} == {"o1", "o2", "o3"}

    with pytest.warns(cf.ComponentFallbackWarning):
        taper = cf.get_component(
            {"component": "taper", "settings": {"length": 5}}, width2=2
        )
    assert taper.info["length"] == 5
    assert taper.info["width2"] == 2


def test_get_component_takes_component_setting() -> None:
    """Containers can get their own ``component`` setting as a keyword."""
    c = cf.get_component("array", component="straight", columns=2)
    assert c.settings["component"] == "straight"
    assert c.settings["columns"] == 2


@pytest.mark.parametrize("fallback", [False, True])
def test_get_component_settings_precedence(restore_pdk: None, fallback: bool) -> None:
    """Spec settings, then kwargs, then settings, with or without fallback."""
    if fallback:
        _activate_pdk(
            "missing_taper", {k: v for k, v in PDK.cells.items() if k != "taper"}
        )
        warns = pytest.warns(cf.ComponentFallbackWarning)
    else:
        warns = contextlib.nullcontext()
    spec = {"component": "taper", "settings": {"length": 5, "width2": 2}}

    with warns:
        taper = cf.get_component(spec, length=7)
    assert taper.info["length"] == 7
    assert taper.info["width2"] == 2

    with warns:
        taper = cf.get_component(spec, settings={"length": 9}, length=7)
    assert taper.info["length"] == 9


def test_unknown_component_raises() -> None:
    with pytest.raises(ValueError, match="not in PDK"):
        cf.get_component("does_not_exist")


def test_pdk_override_applies_inside_mzis(restore_pdk: None) -> None:
    assert not _has_marker(gf.components.mzi_lattice())
    assert not _has_marker(gf.components.mzi_lattice_mmi())
    assert not _has_marker(gf.components.mzi_pads_center())
    assert not _has_marker(gf.components.mzit())
    assert not _has_marker(gf.components.mzit_lattice())

    _activate_pdk("override_straight", {**PDK.cells, "straight": marked_straight})

    assert _has_marker(gf.components.mzi_lattice())
    assert _has_marker(gf.components.mzi_lattice_mmi())
    assert _has_marker(gf.components.mzi_pads_center())
    assert _has_marker(gf.components.mzit())
    assert _has_marker(gf.components.mzit_lattice())


def test_pdk_override_applies_inside_mmis(restore_pdk: None) -> None:
    assert not _has_marker(gf.components.mmi())
    assert not _has_marker(gf.components.mmi1x2_with_sbend())
    assert not _has_marker(gf.components.mmi2x2())
    assert not _has_marker(gf.components.mmi_90degree_hybrid())

    _activate_pdk("override_straight", {**PDK.cells, "straight": marked_straight})

    # mmi2x2 used to default to gf.components.straight and mmi1x2_with_sbend
    # called it directly, bypassing the PDK.
    assert _has_marker(gf.components.mmi())
    assert _has_marker(gf.components.mmi1x2_with_sbend())
    assert _has_marker(gf.components.mmi2x2())
    assert _has_marker(gf.components.mmi_90degree_hybrid())


def test_pdk_override_applies_inside_bends(restore_pdk: None) -> None:
    assert not _has_marker(gf.components.bend_s())
    assert not _has_marker(gf.components.bend_s(size=(10, 0)))

    _activate_pdk(
        "override_bends",
        {**PDK.cells, "bezier": marked_bezier, "straight": marked_straight},
    )

    # bend_s used to call bezier and gf.components.straight directly.
    assert _has_marker(gf.components.bend_s())
    assert _has_marker(gf.components.bend_s(size=(10, 0)))


def test_pdk_override_applies_inside_tapers(restore_pdk: None) -> None:
    assert not _has_marker(
        gf.components.taper_cross_section(
            cross_section1="strip", cross_section2="strip"
        )
    )

    _activate_pdk("override_straight", {**PDK.cells, "straight": marked_straight})

    # taper_cross_section used to call gf.components.straight directly.
    assert _has_marker(
        gf.components.taper_cross_section(
            cross_section1="strip", cross_section2="strip"
        )
    )


def test_pdk_override_applies_inside_waveguides(restore_pdk: None) -> None:
    names = [
        "straight_heater_doped_rib",
        "straight_heater_meander",
        "straight_heater_meander_doped",
        "straight_heater_metal",
        "straight_heater_metal_simple",
        "straight_pin",
        "straight_pin_slot",
    ]
    for name in names:
        assert not _has_marker(getattr(gf.components, name)()), name

    _activate_pdk("override_straight", {**PDK.cells, "straight": marked_straight})

    for name in names:
        assert _has_marker(getattr(gf.components, name)()), name


def test_cached_component_sequence_is_not_modified(restore_pdk: None) -> None:
    """straight_heater_metal_undercut works when a PDK caches its sequence."""
    cached = gf.cell(
        gf.components.component_sequence,
        basename="cached_component_sequence",
        register_factory=False,
    )
    _activate_pdk("cached_sequence", {**PDK.cells, "component_sequence": cached})

    c = gf.components.straight_heater_metal_undercut()
    assert {"l_e1", "r_e1"} <= {p.name for p in c.ports}


def test_pdk_override_applies_inside_detectors(restore_pdk: None) -> None:
    assert not _has_marker(gf.components.ge_detector_straight_si_contacts())

    _activate_pdk("override_straight", {**PDK.cells, "straight": marked_straight})

    # ge_detector_straight_si_contacts used to call gf.components.straight
    # directly.
    assert _has_marker(gf.components.ge_detector_straight_si_contacts())
