from __future__ import annotations

import pytest

import gdsfactory as gf
from gdsfactory.components.quantum.resonator import _cpw_cross_section


@pytest.mark.parametrize("factory", [gf.c.resonator_cpw, gf.c.resonator_lumped])
@pytest.mark.parametrize("layer", [(1, 0), (41, 0)])
@pytest.mark.parametrize("port_type", ["electrical", "optical"])
def test_resonator_port_types(
    factory: gf.typings.ComponentFactory, layer: gf.typings.LayerSpec, port_type: str
) -> None:
    """Explicit resonator port types override the layer's extrusion policy."""
    c = factory(layer_metal=layer, port_type=port_type)
    assert {p.name for p in c.ports} == {"input", "output"}
    assert all(p.port_type == port_type for p in c.ports)
    if factory is gf.c.resonator_lumped:
        netlist = c.get_netlist(on_dangling_port="ignore")
        assert "output" in netlist["ports"]
        assert len(netlist["nets"]) == 7


def test_resonator_radius_is_geometry() -> None:
    """Changing meander radii must not re-register the same physical profile."""
    for radius in (20.0, 30.0):
        lumped = gf.c.resonator_lumped(inductor_radius=radius)
        assert lumped.info["inductor_radius"] == radius
        cpw = gf.c.resonator_cpw(meander_pitch=2 * radius)
        assert cpw.ports["input"].width == 10


def test_cpw_sections() -> None:
    """The CPW consists of a center conductor and two disjoint gap strips."""
    xs = _cpw_cross_section(width=10, gap=6, layer_metal=(1, 0), layer_gap=(2, 0))
    sections = xs.get_sections()
    assert sorted((s.layer.layer, s.section_min, s.section_max) for s in sections) == [
        (1, -5, 5),
        (2, -11, -5),
        (2, 5, 11),
    ]


@pytest.mark.parametrize("factory", [gf.c.straight, gf.c.bend_circular])
def test_port_override_preserves_default_cell(
    factory: gf.typings.ComponentFactory,
) -> None:
    """Electrical overrides neither mutate nor replace cached optical cells."""
    optical = factory()
    electrical = factory(port_type="electrical")
    assert all(p.port_type == "optical" for p in optical.ports)
    assert all(p.port_type == "electrical" for p in electrical.ports)
    assert "port_type" not in optical.settings.model_dump()
    assert electrical.settings["port_type"] == "electrical"
