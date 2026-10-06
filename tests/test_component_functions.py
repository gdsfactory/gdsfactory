"""Tests for gf.component_functions and the gf.components cells that wrap them."""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

import pytest

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.gpdk import PDK
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Delta, Ints, LayerSpec

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


@gf.cell(basename="marked_taper", register_factory=False)
def marked_taper(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    with_two_ports: bool = True,
    cross_section: CrossSectionSpec = "strip",
    port_names: tuple[str, str] = ("o1", "o2"),
    port_types: tuple[str, str] = ("optical", "optical"),
) -> gf.Component:
    c = cf.taper(
        length=length,
        width1=width1,
        width2=width2,
        with_two_ports=with_two_ports,
        cross_section=cross_section,
        port_names=port_names,
        port_types=port_types,
    )
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=MARKER)
    return c


@gf.cell(basename="marked_edge_coupler_array", register_factory=False)
def marked_edge_coupler_array(**kwargs: Any) -> gf.Component:
    c = cf.edge_coupler_array(**kwargs)
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


@gf.cell(basename="marked_optimal_hairpin", register_factory=False)
def marked_optimal_hairpin(
    width: float = 0.2,
    pitch: float = 0.6,
    length: float = 10,
    turn_ratio: float = 4,
    num_pts: int = 50,
    layer: LayerSpec = (1, 0),
) -> gf.Component:
    c = cf.optimal_hairpin(
        width=width,
        pitch=pitch,
        length=length,
        turn_ratio=turn_ratio,
        num_pts=num_pts,
        layer=layer,
    )
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=MARKER)
    return c


@gf.cell(basename="marked_pad", register_factory=False)
def marked_pad(
    size: tuple[float, float] = (100.0, 100.0),
    layer: str = "MTOP",
    port_orientation: float | None = 0,
    port_orientations: tuple[int, ...] | None = (180, 90, 0, -90),
) -> gf.Component:
    c = cf.pad(
        size=size,
        layer=layer,
        port_orientation=port_orientation,
        port_orientations=port_orientations,
    )
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=MARKER)
    return c


@gf.cell(basename="marked_pixel_array", register_factory=False)
def marked_pixel_array(
    pixels: str = gf.components.character_a,
    pixel_size: float = 10.0,
    layer: LayerSpec = "M1",
) -> gf.Component:
    c = cf.pixel_array(pixels=pixels, pixel_size=pixel_size, layer=layer)
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=MARKER)
    return c


@gf.cell(basename="marked_compass", register_factory=False)
def marked_compass(
    size: tuple[float, float] = (4.0, 2.0),
    layer: LayerSpec = "WG",
    port_type: str | None = "electrical",
    port_inclusion: float = 0.0,
    port_orientations: Ints | None = (180, 90, 0, -90),
    auto_rename_ports: bool = True,
) -> gf.Component:
    c = cf.compass(
        size=size,
        layer=layer,
        port_type=port_type,
        port_inclusion=port_inclusion,
        port_orientations=port_orientations,
        auto_rename_ports=auto_rename_ports,
    )
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=MARKER)
    return c


@gf.cell(basename="marked_triangle", register_factory=False)
def marked_triangle(
    x: float = 10,
    xtop: float = 0,
    y: float = 20,
    ybot: float = 0,
    layer: LayerSpec = "WG",
) -> gf.Component:
    c = cf.triangle(x=x, xtop=xtop, y=y, ybot=ybot, layer=layer)
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=MARKER)
    return c


@gf.cell(basename="marked_via", register_factory=False)
def marked_via(
    size: tuple[float, float] = (0.7, 0.7),
    enclosure: float = 1.0,
    layer: LayerSpec = "VIAC",
    pitch: float = 2,
) -> gf.Component:
    c = cf.via(size=size, enclosure=enclosure, layer=layer, pitch=pitch)
    # Inside the via, so the via bbox (used to place the via array) is unchanged.
    c.add_polygon([(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)], layer=MARKER)
    return c


@gf.cell(basename="marked_free_propagation_region", register_factory=False)
def marked_free_propagation_region(**kwargs: Any) -> gf.Component:
    c = cf.free_propagation_region(**kwargs)
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=MARKER)
    return c


@gf.cell(basename="marked_dbr_cell", register_factory=False)
def marked_dbr_cell(**kwargs: Any) -> gf.Component:
    c = cf.dbr_cell(**kwargs)
    c.add_polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)], layer=MARKER)
    return c


@gf.cell(basename="marked_circle", register_factory=False)
def marked_circle(**kwargs: Any) -> gf.Component:
    c = cf.circle(**kwargs)
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=MARKER)
    return c


@gf.cell(basename="marked_taper_cross_section", register_factory=False)
def marked_taper_cross_section(**kwargs: Any) -> gf.Component:
    c = cf.taper_cross_section(**kwargs)
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=MARKER)
    return c


@gf.cell(basename="marked_bend_euler", register_factory=False)
def marked_bend_euler(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int | None = None,
    angular_step: float | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
) -> gf.Component:
    c = cf.bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        angular_step=angular_step,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
    )
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=MARKER)
    return c


@gf.cell(basename="marked_bend_circular", register_factory=False)
def marked_bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    c = cf.bend_circular(
        radius=radius, angle=angle, width=width, cross_section=cross_section
    )
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=MARKER)
    return c


@gf.cell(basename="marked_coupler_symmetric", register_factory=False)
def marked_coupler_symmetric(
    bend: ComponentSpec = "bend_s",
    gap: float = 0.234,
    dy: Delta = 4.0,
    dx: Delta = 10.0,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
) -> gf.Component:
    c = cf.coupler_symmetric(
        bend=bend,
        gap=gap,
        dy=dy,
        dx=dx,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
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


def test_pdk_override_applies_inside_edge_couplers(restore_pdk: None) -> None:
    names = [
        "edge_coupler_silicon",
        "edge_coupler_array",
        "edge_coupler_array_with_loopback",
    ]
    for name in names:
        assert not _has_marker(getattr(gf.components, name)()), name

    _activate_pdk("override_taper", {**PDK.cells, "taper": marked_taper})

    # edge_coupler_silicon used to call gf.components.taper directly.
    for name in names:
        assert _has_marker(getattr(gf.components, name)()), name

    _activate_pdk(
        "override_edge_coupler_array",
        {**PDK.cells, "edge_coupler_array": marked_edge_coupler_array},
    )

    # edge_coupler_array_with_loopback used to call edge_coupler_array directly.
    assert _has_marker(gf.components.edge_coupler_array_with_loopback())

    _activate_pdk("override_straight", {**PDK.cells, "straight": marked_straight})

    assert _has_marker(gf.components.edge_coupler_array_with_loopback())


def test_pdk_override_applies_inside_superconductors(restore_pdk: None) -> None:
    assert not _has_marker(gf.components.snspd())

    _activate_pdk(
        "override_optimal_hairpin",
        {**PDK.cells, "optimal_hairpin": marked_optimal_hairpin},
    )

    # snspd used to call optimal_hairpin and gf.c.compass directly.
    assert _has_marker(gf.components.snspd())


def test_pdk_override_applies_inside_pads(restore_pdk: None) -> None:
    pad_names = ["pad_array", "pad_gsg_short", "pads_shorted"]
    straight_names = ["pad_gs", "pad_gsg"]
    for name in pad_names + straight_names:
        assert not _has_marker(getattr(gf.components, name)()), name

    _activate_pdk(
        "override_pads",
        {**PDK.cells, "pad": marked_pad, "straight": marked_straight},
    )

    # pad_gs and pad_gsg used to call gf.c.straight directly.
    for name in pad_names + straight_names:
        assert _has_marker(getattr(gf.components, name)()), name


def test_pdk_override_applies_inside_texts(restore_pdk: None) -> None:
    names = ["text_lines", "text_rectangular", "text_rectangular_multi_layer"]
    for name in names:
        assert not _has_marker(getattr(gf.components, name)()), name

    _activate_pdk(
        "override_pixel_array", {**PDK.cells, "pixel_array": marked_pixel_array}
    )

    # text_rectangular used to call pixel_array directly and text_lines
    # called gf.c.text_rectangular.
    for name in names:
        assert _has_marker(getattr(gf.components, name)()), name


def test_pdk_override_applies_inside_shapes(restore_pdk: None) -> None:
    components: dict[str, Callable[[], gf.Component]] = {
        "rectangle": gf.components.rectangle,
        "rectangles": gf.components.rectangles,
        "cross": gf.components.cross,
        "fiducial_squares": gf.components.fiducial_squares,
        "nxn": gf.components.nxn,
        "rect_su_shape": lambda: gf.components.rect_su_shape(L1=-10),
        "triangle2": gf.components.triangle2,
        "triangle4": gf.components.triangle4,
    }
    for name, component in components.items():
        assert not _has_marker(component()), name

    _activate_pdk(
        "override_shapes",
        {**PDK.cells, "compass": marked_compass, "triangle": marked_triangle},
    )

    # These used to call rectangle, compass and triangle directly.
    for name, component in components.items():
        assert _has_marker(component()), name


def test_pdk_override_applies_inside_vias(restore_pdk: None) -> None:
    names = [
        "via_chain",
        "via_corner",
        "via_stack",
        "via_stack_corner45",
        "via_stack_corner45_extended",
        "via_stack_with_offset",
    ]
    for name in names:
        assert not _has_marker(getattr(gf.components, name)()), name

    # via1, via2 and viac are aliases of via, so overriding via reaches them.
    _activate_pdk("override_via", {**PDK.cells, "via": marked_via})

    for name in names:
        assert _has_marker(getattr(gf.components, name)()), name


def test_cell_alias_matches_partial() -> None:
    """A CellAlias serializes, names its cells and shows its signature like a partial."""
    from gdsfactory.serialization import clean_value_json

    alias = cf.CellAlias(gf.components.via, layer="VIA1")
    plain = partial(gf.components.via, layer="VIA1")

    assert inspect.signature(alias) == inspect.signature(plain)
    assert clean_value_json(alias) == clean_value_json(plain)
    assert alias() is plain()


def test_cell_alias_of_alias_is_flattened() -> None:
    alias = cf.CellAlias(gf.components.via1, size=(1, 1))
    assert alias.func is gf.components.via
    assert alias.keywords == {**gf.components.via1.keywords, "size": (1, 1)}


def test_cell_alias_takes_keywords_only() -> None:
    with pytest.raises(TypeError):
        cf.CellAlias(gf.components.via, (1, 1))
    with pytest.raises(TypeError):
        gf.components.via1((1, 1))


def test_cell_alias_resolves_base_cell_by_name(restore_pdk: None) -> None:
    assert not _has_marker(gf.components.via1())

    _activate_pdk("override_via", {**PDK.cells, "via": marked_via})

    assert _has_marker(gf.components.via1())
    assert _has_marker(gf.components.via_stack_m2_m3())
    assert gf.components.via1().settings["layer"] == "VIA1"


def test_partials_of_cells_are_cell_aliases() -> None:
    """Every partial of a cell in gf.components is a CellAlias."""
    cells = gf.get_active_pdk().cells
    plain = sorted(
        name
        for name, cell in cells.items()
        if isinstance(cell, partial) and not isinstance(cell, cf.CellAlias)
    )
    assert not plain


def test_pdk_override_applies_inside_filters(restore_pdk: None) -> None:
    overrides: dict[str, tuple[Callable[..., Any], list[str]]] = {
        "straight": (
            marked_straight,
            ["dbr", "dbr_cell", "dbr_tapered", "loop_mirror"],
        ),
        "taper": (
            marked_taper,
            ["dbr_tapered", "mode_converter", "polarization_splitter_rotator"],
        ),
        "free_propagation_region": (marked_free_propagation_region, ["awg"]),
        "dbr_cell": (marked_dbr_cell, ["dbr"]),
        "circle": (marked_circle, ["fiber", "fiber_array"]),
        "taper_cross_section": (marked_taper_cross_section, ["terminator"]),
    }
    for _, names in overrides.values():
        for name in names:
            assert not _has_marker(getattr(gf.components, name)()), name

    # These used to call straight, taper, dbr_cell, circle and taper_cross_section
    # directly. awg reaches free_propagation_region through the
    # free_propagation_region_input/output aliases, now looked up by name.
    for cell_name, (cell, names) in overrides.items():
        _activate_pdk(f"override_{cell_name}", {**PDK.cells, cell_name: cell})
        for name in names:
            assert _has_marker(getattr(gf.components, name)()), (cell_name, name)


def test_pdk_override_applies_inside_spirals(restore_pdk: None) -> None:
    overrides: dict[str, tuple[Callable[..., gf.Component], list[str]]] = {
        "straight": (
            marked_straight,
            [
                "delay_snake",
                "delay_snake2",
                "delay_snake_sbend",
                "spiral",
                "spiral_racetrack",
                "spiral_racetrack_fixed_length",
                "spiral_racetrack_heater_doped",
                "spiral_racetrack_heater_metal",
            ],
        ),
        # delay_snake and delay_snake2 use bend_euler180, an alias of bend_euler.
        "bend_euler": (
            marked_bend_euler,
            [
                "delay_snake",
                "delay_snake2",
                "delay_snake_sbend",
                "spiral",
                "spiral_racetrack",
                "spiral_racetrack_heater_doped",
                "spiral_racetrack_heater_metal",
            ],
        ),
        "bend_circular": (
            marked_bend_circular,
            ["spiral_double", "spiral_racetrack_fixed_length"],
        ),
    }
    for cell_name, (marked, names) in overrides.items():
        _activate_pdk(PDK.name, dict(PDK.cells))
        for name in names:
            assert not _has_marker(getattr(gf.components, name)()), name

        _activate_pdk(f"override_{cell_name}", {**PDK.cells, cell_name: marked})

        # These used to call straight and spiral_racetrack directly.
        for name in names:
            assert _has_marker(getattr(gf.components, name)()), (cell_name, name)


def test_pdk_override_applies_inside_couplers(restore_pdk: None) -> None:
    straight_names = [
        "coupler90",
        "coupler_adiabatic",
        "coupler_asymmetric",
        "coupler_broadband",
        "coupler_ring",
        "coupler_straight",
        "coupler_straight_asymmetric",
    ]
    bend_euler_names = [
        "coupler90",
        "coupler90bend",
        "coupler_broadband",
        "coupler_ring",
    ]
    bezier_names = [
        "coupler",
        "coupler_adiabatic",
        "coupler_asymmetric",
        "coupler_full",
        "coupler_symmetric",
    ]
    for name in {*straight_names, *bend_euler_names, *bezier_names}:
        assert not _has_marker(getattr(gf.components, name)()), name

    _activate_pdk("override_straight", {**PDK.cells, "straight": marked_straight})

    # coupler_straight, coupler_asymmetric and coupler_straight_asymmetric used
    # to call gf.c.straight directly, and coupler_ring called coupler90 and
    # coupler_straight directly.
    for name in straight_names:
        assert _has_marker(getattr(gf.components, name)()), name

    _activate_pdk("override_bend_euler", {**PDK.cells, "bend_euler": marked_bend_euler})

    for name in bend_euler_names:
        assert _has_marker(getattr(gf.components, name)()), name

    _activate_pdk("override_bezier", {**PDK.cells, "bezier": marked_bezier})

    # coupler_adiabatic used to call bezier directly.
    for name in bezier_names:
        assert _has_marker(getattr(gf.components, name)()), name

    _activate_pdk(
        "override_coupler_symmetric",
        {**PDK.cells, "coupler_symmetric": marked_coupler_symmetric},
    )

    # coupler used to call coupler_symmetric directly.
    assert _has_marker(gf.components.coupler())
