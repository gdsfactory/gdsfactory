"""Profile and file-persistence requirements from kfactory #1054."""

from __future__ import annotations

import inspect
import subprocess
from functools import partial
from pathlib import Path
from unittest.mock import patch

import kfactory as kf
import pytest

import gdsfactory as gf
from gdsfactory._kcl import temporary_kcl
from gdsfactory.typings import ComponentFactory


@pytest.mark.parametrize("all_angle", [False, True])
@pytest.mark.parametrize("offset", [0.0, 0.125])
def test_extrude_bbox_flag(offset: float, all_angle: bool) -> None:
    xs = gf.cross_section.cross_section(
        width=0.5,
        offset=offset,
        sections=[("SLAB90", -1.0, 1.0)],
        bbox_layers=["DEVREC", "M1"],
        bbox_offsets=[2.0, 0.5],
    )
    assert isinstance(xs, gf.AsymmetricCrossSection) == bool(offset)
    path = gf.path.straight(10)
    bare = path.extrude(xs, all_angle=all_angle)
    explicit = path.extrude(xs, all_angle=all_angle, add_bbox=False)
    padded = path.extrude(xs, all_angle=all_angle, add_bbox=True)
    functional = gf.path.extrude(path, xs, all_angle=all_angle, add_bbox=True)
    for layer, padding in xs.bbox_sections.items():
        index = padded.kcl.layer(layer)
        assert bare.dbbox(index).empty()
        assert explicit.dbbox(index).empty()
        assert padded.dbbox(index) == bare.dbbox().enlarged(padding)
        assert functional.dbbox(index) == padded.dbbox(index)
    for component in (bare, explicit, padded, functional):
        for port in component.ports:
            assert port.cross_section.base == xs.base
    for section in xs.get_sections():
        index = padded.kcl.layer(section.layer)
        assert padded.dbbox(index) == bare.dbbox(index)


@pytest.mark.parametrize("all_angle", [False, True])
@pytest.mark.parametrize("offset", [0.0, 0.125])
def test_extrude_calls_kfactory_add_bbox(offset: float, all_angle: bool) -> None:
    gf.clear_cache()
    xs = gf.cross_section.cross_section(
        width=0.5,
        offset=offset,
        radius=10,
        radius_min=5,
        bbox_layers=["DEVREC", "M1"],
        bbox_offsets=[2.0, 0.5],
    )
    cls = kf.DAsymmetricCrossSection if offset else kf.DCrossSection
    assert type(xs) is cls
    path = gf.path.straight(10)
    with patch.object(cls, "add_bbox", autospec=True, side_effect=cls.add_bbox) as draw:
        path.extrude(xs, all_angle=all_angle)
        draw.assert_not_called()
        c = path.extrude(xs, all_angle=all_angle, add_bbox=True)
        draw.assert_called_once_with(xs, c)


@pytest.mark.parametrize("offset", [0.0, 0.125])
@pytest.mark.parametrize(
    "factory",
    [
        gf.c.straight,
        gf.c.straight_all_angle,
        gf.c.bend_euler,
        gf.c.bend_s,
        gf.c.taper,
        gf.c.wire_corner,
        gf.c.grating_coupler_elliptical,
    ],
)
def test_components_call_kfactory_add_bbox(
    factory: ComponentFactory, offset: float
) -> None:
    gf.clear_cache()
    xs = gf.cross_section.cross_section(
        width=0.5,
        offset=offset,
        radius=10,
        radius_min=5,
        bbox_layers=["DEVREC", "M1"],
        bbox_offsets=[2.0, 0.5],
    )
    cls = type(xs)
    with patch.object(cls, "add_bbox", autospec=True, side_effect=cls.add_bbox) as draw:
        # Exercise construction even when a virtual factory has cached its result.
        c = inspect.unwrap(factory)(cross_section=xs)
    assert any(call.args[0].base == xs.base for call in draw.call_args_list)
    for layer in xs.bbox_sections:
        assert not c.dbbox(c.kcl.layer(layer)).empty()


@pytest.mark.parametrize("all_angle", [False, True])
@pytest.mark.parametrize("offset", [0.0, 0.125])
@pytest.mark.parametrize("explicit_ref", [False, True])
def test_component_add_bbox_with_pending_vinsts(
    all_angle: bool, offset: float, explicit_ref: bool
) -> None:
    gf.clear_cache()
    xs = gf.cross_section.cross_section(
        width=0.5,
        offset=offset,
        radius=10,
        radius_min=5,
        bbox_layers=["DEVREC", "M1"],
        bbox_offsets=[2.0, 0.5],
    )
    c = gf.ComponentAllAngle() if all_angle else gf.Component()
    child = gf.ComponentAllAngle()
    bounds = gf.kdb.DBox(0, -1, 10, 2)
    child.shapes(xs.layer).insert(bounds)
    c.create_vinst(child)
    with patch("kfactory.cross_section.logger.warning") as warning:
        xs.add_bbox(c, ref=bounds if explicit_ref else None)
        if all_angle:
            warning.assert_not_called()
        else:
            warning.assert_called_once()
            assert "insert_vinsts()" in warning.call_args.args[0]
    for layer, padding in xs.bbox_sections.items():
        assert c.dbbox(c.kcl.layer(layer)) == bounds.enlarged(padding)
    assert len(c.vinsts) == 1


@pytest.mark.parametrize("all_angle", [False, True])
def test_bbox_uses_emitted_geometry(all_angle: bool) -> None:
    xs = gf.cross_section.cross_section(
        width=0.5,
        sections=[("SLAB90", -5.0, 5.0)],
        bbox_layers=["DEVREC"],
        bbox_offsets=[1.0],
    )
    c = gf.path.straight(10).extrude(
        xs,
        all_angle=all_angle,
        add_bbox=True,
        hidden=[1],
        insets={0: (2.0, 3.0)},
        width_function=lambda t: 0.5 + t,
        offset_function=lambda t: t,
        ports={},
    )
    assert c.dbbox(gf.get_layer("DEVREC")) == gf.kdb.DBox(1.0, -2.75, 8.0, 1.25)
    assert c.dbbox(gf.get_layer("SLAB90")).empty()
    empty = gf.path.straight(10).extrude(
        xs, all_angle=all_angle, add_bbox=True, hidden=[0, 1], ports={}
    )
    assert empty.dbbox().empty()


@pytest.mark.parametrize("angle", [90, -90, 180, -180])
@pytest.mark.parametrize("bend", [gf.c.bend_circular, gf.c.bend_euler, gf.c.bend_topic])
def test_bend_bbox_clipping(bend: ComponentFactory, angle: float) -> None:
    xs = gf.cross_section.cross_section(
        width=0.5,
        radius=10,
        radius_min=5,
        bbox_layers=["DEVREC", "M1"],
        bbox_offsets=[2.0, 0.5],
    )
    c = bend(cross_section=xs, angle=angle)
    core = c.dbbox(c.kcl.layer(xs.layer))
    for layer, padding in xs.bbox_sections.items():
        expected = core.enlarged(padding)
        if angle == 90:
            expected.top = core.top
        elif angle == -90:
            expected.bottom = core.bottom
        assert c.dbbox(c.kcl.layer(layer)) == expected


@pytest.mark.parametrize(
    "factory", [gf.c.straight, gf.c.straight_all_angle, gf.c.wire_corner, gf.c.taper]
)
def test_component_bbox_layers(factory: ComponentFactory) -> None:
    xs = gf.cross_section.cross_section(
        width=0.5,
        radius=10,
        radius_min=5,
        bbox_layers=["DEVREC", "M1"],
        bbox_offsets=[2.0, 0.5],
    )
    c = factory(cross_section=xs)
    core = c.dbbox(c.kcl.layer(xs.layer))
    for layer, padding in xs.bbox_sections.items():
        assert c.dbbox(c.kcl.layer(layer)) == core.enlarged(padding)


@pytest.mark.parametrize("offset", [0.0, 0.125, -0.125])
@pytest.mark.parametrize("width", [0.5, 0.501, 0.451])
def test_edges_snap_once(width: float, offset: float) -> None:
    with temporary_kcl("edge_snapping") as kcl:
        xs = gf.cross_section.cross_section(width=width, offset=offset, kcl=kcl)
        main = xs.base.get_sections()[0]
        assert main.section_min == kcl.to_dbu(offset - width / 2)
        assert main.section_max == kcl.to_dbu(offset + width / 2)
        assert isinstance(xs, gf.SymmetricCrossSection) == (
            main.section_min == -main.section_max
        )
        rebuilt = gf.cross_section.cross_section(
            width=None, sections=xs.get_sections(), kcl=kcl
        )
        assert rebuilt.base == xs.base


def test_overlaps_preserve_main_strip_and_merge_auxiliary() -> None:
    with temporary_kcl("overlapping_profile") as kcl:
        xs = gf.cross_section.cross_section(
            width=0.5,
            sections=[
                ("WG", -6.75, 0.25),
                ("SLAB90", -2.55, -1.55),
                ("SLAB90", -2.05, -1.05),
            ],
            kcl=kcl,
        )
        assert isinstance(xs, gf.AsymmetricCrossSection)
        assert xs.width == 0.5
        assert len(xs.get_sections()) == 3
        slab = next(
            s for s in xs.get_sections() if s.layer == gf.get_layer_info("SLAB90")
        )
        assert (slab.section_min, slab.section_max) == pytest.approx((-2.55, -1.05))


def test_relative_enclosure_and_absolute_width_replacement() -> None:
    with temporary_kcl("enclosure_width") as kcl:
        xs = gf.cross_section.cross_section(
            width=0.5, cladding_layers=["SLAB90"], cladding_offsets=3.0, kcl=kcl
        )
        assert xs.get_sections()[1].section_max == 3.25
        widened = kf.DCrossSection(
            kcl=kcl,
            width=1.1,
            layer=xs.layer,
            sections=[(gf.get_layer_info("SLAB90"), 3.0)],
        )
        assert widened.get_sections()[1].section_max == pytest.approx(3.55)
        absolute = gf.cross_section.with_width(xs, 1.1)
        assert absolute.get_sections()[1].section_max == 3.25


def test_ring_enclosure_and_core_identity() -> None:
    with temporary_kcl("ring_enclosure") as kcl:
        xs = gf.cross_section.cross_section(
            width=0.5,
            sections=[("WG", -0.1, 0.1), ("SLAB90", -2.0, -1.0), ("SLAB90", 1.0, 2.0)],
            kcl=kcl,
        )
        assert isinstance(xs, kf.DCrossSection)
        assert xs.width == 0.5
        assert len(xs.get_sections()) == 4
        assert xs.sections[gf.get_layer_info("SLAB90")] == [(0.75, 1.75)]


def test_explicit_name_and_radius_conflicts() -> None:
    with temporary_kcl("profile_names") as kcl:
        xs = gf.cross_section.cross_section(
            width=0.5, radius=10.0, name="canonical", kcl=kcl
        )
        assert gf.cross_section.cross_section(width=0.5, kcl=kcl).base is xs.base
        with pytest.raises(kf.exceptions.CrossSectionNamingConflictError):
            gf.cross_section.cross_section(width=0.5, name="alias", kcl=kcl)
        with pytest.raises(kf.exceptions.CrossSectionNamingConflictError):
            gf.cross_section.cross_section(width=0.5, radius=5.0, kcl=kcl)


@pytest.mark.parametrize("offset", [0, 0.125])
@pytest.mark.parametrize("field", ["radius", "radius_min"])
def test_radii_cannot_be_added_after_creation(offset: float, field: str) -> None:
    with temporary_kcl("late_radius") as kcl:
        xs = gf.cross_section.cross_section(offset=offset, kcl=kcl)
        with pytest.raises(kf.exceptions.CrossSectionNamingConflictError):
            gf.cross_section.cross_section(offset=offset, kcl=kcl, **{field: 10.0})
        assert xs.radius is None
        assert xs.radius_min is None


@pytest.mark.parametrize("ports_first", [False, True])
def test_port_profiles_have_radii_at_creation(ports_first: bool) -> None:
    from gdsfactory.cross_section.utils import section_cross_section

    with temporary_kcl("port_radius") as kcl:

        def make_ports() -> gf.Component:
            c = gf.Component(kcl=kcl)
            c.add_port("o1", center=(0, 0), width=0.74, layer="WG")
            heated = gf.cross_section.strip_heater_metal(heater_width=2.8, kcl=kcl)
            profile, _ = section_cross_section(heated.get_sections()[1], kcl)
            c.add_port(
                "e1", center=(0, 0), cross_section=profile, port_type="electrical"
            )
            return c

        if ports_first:
            c = make_ports()
        optical = gf.cross_section.strip(width=0.74, kcl=kcl)
        electrical = gf.cross_section.heater_metal(width=2.8, kcl=kcl)
        if not ports_first:
            c = make_ports()
        assert c.ports["o1"].cross_section.base is optical.base
        assert optical.radius == 10
        assert optical.radius_min == 3.5
        assert c.ports["e1"].cross_section.base is electrical.base
        assert electrical.radius == pytest.approx(2.8)


@pytest.mark.parametrize(
    ("factory", "layer", "width", "port"),
    [
        (gf.components.wire_corner, "M3", 19.772, "e1"),
        (gf.components.grating_coupler_elliptical, "WG", 0.742, "o1"),
    ],
)
def test_components_keep_explicit_profile_radii(
    factory: ComponentFactory, layer: str, width: float, port: str
) -> None:
    xs = kf.DCrossSection(
        kcl=gf.kcl,
        width=width,
        layer=gf.get_layer_info(layer),
        sections=[],
        radius=7,
        radius_min=3.5,
    )
    c = factory(cross_section=xs)
    assert c.ports[port].cross_section.base is xs.base
    assert c.ports[port].cross_section.radius == 7


def test_insets_hidden_and_explicit_ports() -> None:
    xs = gf.cross_section.cross_section(
        width=1.0, layer=(811, 0), sections=[((812, 0), 2.0, 3.0)]
    )
    c = gf.path.straight(10.0, npoints=11).extrude(
        xs,
        insets={1: (2.0, 3.0)},
        hidden=[0],
        ports={0: ("input", "output", "optical"), 1: ("a", "b", "electrical")},
    )
    assert c.area((811, 0)) == 0
    assert c.area((812, 0)) == 5
    assert c.ports["a"].center == (2.0, -2.5)
    assert c.ports["b"].center == (7.0, -2.5)
    assert c.ports["a"].cross_section.width == 1.0
    assert len(c.ports["a"].cross_section.get_sections()) == 1


def test_odd_auxiliary_port_is_recentered_on_grid() -> None:
    xs = gf.cross_section.cross_section(
        width=0.5, layer=(813, 0), sections=[((814, 0), 0.02, 0.251)]
    )
    c = gf.path.straight(10.0).extrude(xs, ports={1: ("a", "b", "electrical")})
    for port in c.ports:
        assert isinstance(port.cross_section, kf.DAsymmetricCrossSection)
        assert port.width == 0.231
        assert port.y / gf.kcl.dbu == pytest.approx(round(port.y / gf.kcl.dbu))
        assert len(port.cross_section.get_sections()) == 1


@pytest.mark.parametrize("output_port", [False, True])
def test_dynamic_taper_can_end_at_zero_without_a_port(output_port: bool) -> None:
    xs = gf.cross_section.cross_section(width=0.5, layer=(831, 0))
    ports = {0: (None, "out", "optical")} if output_port else {}
    c = gf.path.straight(10, npoints=11).extrude(
        xs, width_function=lambda t: t, ports=ports
    )
    assert c.area((831, 0)) == 5
    assert len(c.ports) == int(output_port)
    if output_port:
        assert c.ports["out"].width == 1
        assert c.ports["out"].center == (10, 0)


def _roundtrip_presets(directory: Path, reverse: bool = False) -> None:
    gf.gpdk.PDK.activate()
    factories = dict(gf.cross_section.cross_sections)
    factories.update(
        symmetric_bbox=partial(
            gf.cross_section.cross_section,
            width=0.501,
            layer=(821, 0),
            sections=[((821, 0), -2, 2)],
            bbox_layers=[(822, 0)],
            bbox_offsets=[3],
            radius=15,
            radius_min=7,
        ),
        asymmetric_bbox=partial(
            gf.cross_section.cross_section,
            width=None,
            sections=[((823, 0), -0.225, 0.226), ((823, 0), -2, 0.5), ((824, 0), 1, 2)],
            bbox_layers=[(822, 0)],
            bbox_offsets=[2],
            radius=20,
            radius_min=10,
        ),
        offset_core=partial(
            gf.cross_section.cross_section,
            width=0.451,
            offset=1,
            layer=(825, 0),
            sections=[((826, 0), -1, 0)],
        ),
    )
    assert gf.cross_section.metal_routing is gf.cross_section.metal3
    for name in reversed(factories) if reverse else factories:
        xs = factories[name]()
        c = gf.path.straight(10.0).extrude(xs)
        for suffix in ("gds", "oas"):
            filename = directory / f"{name}.{suffix}"
            c.write(filename)
            with temporary_kcl(f"read_{name}_{suffix}") as kcl:
                kcl.read(filename)
                restored = kcl[kcl.layout.top_cell().cell_index()].to_dtype()
                assert len(restored.ports) == len(c.ports)
                for port in c.ports:
                    back = restored.ports[port.name]
                    assert back.port_type == port.port_type
                    assert back.width == port.width
                    assert back.dcplx_trans == port.dcplx_trans
                    assert back.cross_section.base == port.cross_section.base
                    assert back.cross_section.radius == port.cross_section.radius
                    assert (
                        back.cross_section.radius_min == port.cross_section.radius_min
                    )
                    assert (
                        back.cross_section.bbox_sections
                        == port.cross_section.bbox_sections
                    )
                    assert (
                        back.cross_section.get_sections()
                        == port.cross_section.get_sections()
                    )
                for layer in c.layers:
                    region = kf.kdb.Region(c.begin_shapes_rec(gf.get_layer(layer)))
                    back = kf.kdb.Region(
                        restored.begin_shapes_rec(kcl.layout.layer(*layer))
                    )
                    assert (region ^ back).is_empty(), (name, layer)
    assert gf.cross_section.metal_routing().base is gf.cross_section.metal3().base
    assert gf.cross_section.metal3().name == "metal3"


@pytest.mark.parametrize("reverse", [False, True])
def test_all_presets_roundtrip_in_fresh_process(tmp_path: Path, reverse: bool) -> None:
    subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-c",
            "from pathlib import Path; import sys; from tests.test_cross_section_conversion import _roundtrip_presets; _roundtrip_presets(Path(sys.argv[1]), bool(int(sys.argv[2])))",
            str(tmp_path),
            str(int(reverse)),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
