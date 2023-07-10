from __future__ import annotations

import gdsfactory as gf
from gdsfactory import Section
from gdsfactory.generic_tech import LAYER


def test_path_near_collinear() -> None:
    p = gf.path.smooth(points=[(0, 0), (0, 1000), (1, 10000)])
    c = p.extrude(cross_section="strip")
    assert c


def test_path_port_types() -> None:
    """Test path with different port types."""
    s = gf.Section(
        width=2.0,
        offset=-4,
        layer=LAYER.HEATER,
        port_names=["e1", "e2"],
        port_types=("electrical", "electrical"),
    )
    X = gf.CrossSection(
        width=0.5, offset=0, layer=LAYER.SLAB90, port_names=["o1", "o2"], sections=[s]
    )
    P = gf.path.straight(npoints=100, length=10)
    c = gf.path.extrude(P, X)
    assert c.ports["e1"].port_type == "electrical"
    assert c.ports["e2"].port_type == "electrical"
    assert c.ports["o1"].port_type == "optical"
    assert c.ports["o2"].port_type == "optical"


def test_extrude_transition() -> None:
    w1 = 1
    w2 = 5
    length = 10
    cs1 = gf.get_cross_section("strip", width=w1)
    cs2 = gf.get_cross_section("strip", width=w2)
    transition = gf.path.transition(cs1, cs2)
    p = gf.path.straight(length)
    c = gf.path.extrude(p, transition)
    # assert c.ports["o1"].cross_section == cs1
    # assert c.ports["o2"].cross_section == cs2
    assert c.ports["o1"].width == w1
    assert c.ports["o2"].width == w2

    expected_area = (w1 + w2) / 2 * length
    actual_area = c._cell.area(True)[(1, 0)]
    assert actual_area == expected_area


def test_transition_cross_section() -> None:
    w1 = 1
    w2 = 5
    length = 10
    cs1 = gf.get_cross_section("strip", width=w1)
    cs2 = gf.get_cross_section("strip", width=w2)
    transition = gf.path.transition(cs1, cs2, port_names=(None, None))

    c = gf.components.straight(length=length, cross_section=transition)
    assert c.ports["o1"].width == w1
    assert c.ports["o2"].width == w2


def dummy_cladded_wg_cs(intent_layer, core_layer, core_width, clad_layer, clad_width):
    sections = (
        Section(width=core_width, offset=0, layer=core_layer, name="core"),
        Section(width=clad_width, offset=0, layer=clad_layer, name="clad"),
    )
    return gf.cross_section.cross_section(
        width=core_width, sections=sections, layer=intent_layer
    )


def test_transition_cross_section_different_layers() -> None:
    core_width = 1
    w1 = 1
    w2 = 5
    length = 10

    intent_layer_1 = (852, 21)
    intent_layer_2 = (853, 21)

    # in platforms with multiple waveguide types, it is useful to use separate intent layers for the different cross sections
    # this will simulate a transition between waveguides with different intent layers (which i just made up arbitrarily for this test)
    # but shared physical layers
    cs1 = dummy_cladded_wg_cs(
        intent_layer=intent_layer_1,
        core_layer="WG",
        core_width=core_width,
        clad_layer="WGCLAD",
        clad_width=w1,
    )
    cs2 = dummy_cladded_wg_cs(
        intent_layer=intent_layer_2,
        core_layer="WG",
        core_width=core_width,
        clad_layer="WGCLAD",
        clad_width=w2,
    )
    transition = gf.path.transition(cs1, cs2, port_names=(None, None))

    c = gf.components.straight(length=length, cross_section=transition)
    assert c.ports["o1"].width == core_width
    assert c.ports["o2"].width == core_width
    assert c.ports["o1"].layer == intent_layer_1
    assert c.ports["o2"].layer == intent_layer_2

    # area of a trapezoid
    expected_area = (w1 + w2) / 2 * length
    # TODO: restore and replace after area() function is fixed
    # assert c.area() == expected_area
    assert c._cell.area(True)[gf.get_layer("WGCLAD")] == expected_area


def test_diagonal_extrude_consistent_naming():
    """This test intends to check that diagonal extrude components are properly serialized and get the same name on different platforms/environments."""
    p = gf.path.Path([(0, 0), (4.9932849, 6.328497)])
    c = p.extrude(cross_section="strip")
    # This name was generated at the time of writing the test. We expect it to be the same across other platforms.
    expected_name = "extrude_43964f7b"
    assert c.name == expected_name


if __name__ == "__main__":
    test_transition_cross_section()
