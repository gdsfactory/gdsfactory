"""Profile and file-persistence requirements from kfactory #1054."""

from __future__ import annotations

import subprocess
from functools import partial
from pathlib import Path

import kfactory as kf
import pytest

import gdsfactory as gf
from gdsfactory._kcl import temporary_kcl
from gdsfactory.typings import ComponentFactory


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
