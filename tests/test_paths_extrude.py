from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

import gdsfactory as gf
from gdsfactory import Section
from gdsfactory.cross_section import CrossSection
from gdsfactory.generic_tech import LAYER
from gdsfactory.typings import LayerSpec


def test_path_near_collinear() -> None:
    p = gf.path.smooth(points=np.array([(0, 0), (0, 1000), (1, 10000)]))
    c = p.extrude(cross_section="strip")
    assert c


def test_path_port_types() -> None:
    """Test path with different port types."""
    s0 = gf.Section(width=0.5, offset=0, layer=LAYER.SLAB90, port_names=("o1", "o2"))
    s1 = gf.Section(
        width=2.0,
        offset=-4,
        layer=LAYER.HEATER,
        port_names=("e1", "e2"),
        port_types=("electrical", "electrical"),
    )
    X = gf.CrossSection(sections=(s0, s1))
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
    c = gf.path.extrude_transition(p, transition)

    assert c.ports["o1"].width == w1
    assert c.ports["o2"].width == w2

    expected_area = (w1 + w2) / 2 * length
    actual_area = c.area((1, 0))
    assert actual_area == expected_area


def test_extrude_transition_asymmetric() -> None:
    w1 = 2
    w2 = 6
    length = 10

    # Define a cubic polynomial width variation for asymmetric transition
    def polynomial(t: float, y1: float, y2: float) -> float:
        return (y2 - y1) * t**3 + y1

    cs1 = gf.get_cross_section("strip", width=w1)
    cs2 = gf.get_cross_section("strip", width=w2)
    transition = gf.path.transition_asymmetric(
        cs1, cs2, width_type1=polynomial, width_type2="linear"
    )
    # In order to have a transition other than linear, we need to sample more points along the path
    p = gf.path.straight(length, npoints=100)
    c = gf.path.extrude_transition(p, transition)

    assert c.ports["o1"].width == w1
    assert c.ports["o2"].width == w2

    # The expected area can be calculated in three parts
    # 1. The upper part is a trapezoid with bases w1/2 and w2/2, and height length
    expected_area = (w1 / 2 + w2 / 2) / 2 * length
    # 2. The lower part can be broken down into a rectangle of width w1/2 and length length
    expected_area += (w1 / 2) * length

    # 3. And a cubic curve area from w1/2 to w2/2 over length, which can be modeled as
    def f(x: float) -> float:
        return (w2 / 2 - w1 / 2) * (x / length) ** 3

    cubic_area, _ = quad(f, 0, length)
    expected_area += cubic_area

    actual_area = c.area((1, 0))
    assert actual_area == pytest.approx(expected_area, rel=1e-4)


def test_transition_cross_section() -> None:
    w1 = 1
    w2 = 5
    length = 10
    cs1 = gf.get_cross_section("strip", width=w1)
    cs2 = gf.get_cross_section("strip", width=w2)
    transition = gf.path.transition(cs1, cs2)

    p = gf.path.straight(length=length)
    c = gf.path.extrude_transition(p=p, transition=transition)

    assert c.ports["o1"].width == w1
    assert c.ports["o2"].width == w2


def test_transition_asymmetric_cross_section() -> None:
    w1 = 1
    w2 = 5
    length = 10
    cs1 = gf.get_cross_section("strip", width=w1)
    cs2 = gf.get_cross_section("strip", width=w2)
    transition = gf.path.transition_asymmetric(cs1, cs2)

    p = gf.path.straight(length=length)
    c = gf.path.extrude_transition(p=p, transition=transition)

    assert c.ports["o1"].width == w1
    assert c.ports["o2"].width == w2


def dummy_cladded_wg_cs(
    intent_layer: LayerSpec,
    core_layer: LayerSpec,
    core_width: float,
    clad_layer: LayerSpec,
    clad_width: float,
) -> CrossSection:
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

    intent_layer_1 = (1, 0)
    intent_layer_2 = (2, 0)

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
    transition = gf.path.transition(cs1, cs2)
    p = gf.path.straight(length=length)
    c = gf.path.extrude_transition(p=p, transition=transition)

    core_width = core_width
    intent_layer_1_ = gf.get_layer(intent_layer_1)
    intent_layer_2_ = gf.get_layer(intent_layer_2)

    assert c.ports["o1"].width == core_width
    assert c.ports["o2"].width == core_width

    assert c.ports["o1"].layer == intent_layer_1_
    assert c.ports["o2"].layer == intent_layer_2_

    # area of a trapezoid
    expected_area = (w1 + w2) / 2 * length
    assert c.area("WGCLAD") == expected_area


def test_transition_asymmetric_cross_section_different_layers() -> None:
    core_width = 1
    w1 = 1
    w2 = 5
    length = 10

    # Define a cubic polynomial width variation for asymmetric transition
    def polynomial(t: float, y1: float, y2: float) -> float:
        return (y2 - y1) * t**3 + y1

    intent_layer_1 = (1, 0)
    intent_layer_2 = (2, 0)

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
    transition = gf.path.transition_asymmetric(
        cs1, cs2, width_type1=polynomial, width_type2="linear"
    )
    # In order to have a transition other than linear, we need to sample more points along the path
    p = gf.path.straight(length=length, npoints=100)
    c = gf.path.extrude_transition(p=p, transition=transition)

    core_width = core_width
    intent_layer_1_ = gf.get_layer(intent_layer_1)
    intent_layer_2_ = gf.get_layer(intent_layer_2)

    assert c.ports["o1"].width == core_width
    assert c.ports["o2"].width == core_width

    assert c.ports["o1"].layer == intent_layer_1_
    assert c.ports["o2"].layer == intent_layer_2_

    # The expected area can be calculated in three parts
    # 1. The upper part is a trapezoid with bases w1/2 and w2/2, and height length
    expected_area = (w1 / 2 + w2 / 2) / 2 * length
    # 2. The lower part can be broken down into a rectangle of width w1/2 and length length
    expected_area += (w1 / 2) * length

    # 3. And a cubic curve area from w1/2 to w2/2 over length, which can be modeled as
    def f(x: float) -> float:
        return (w2 / 2 - w1 / 2) * (x / length) ** 3

    cubic_area, _ = quad(f, 0, length)
    expected_area += cubic_area

    assert c.area("WGCLAD") == pytest.approx(expected_area, rel=1e-4)


def test_extrude_port_centers() -> None:
    """Tests whether the ports created from CrossSections with multiple Sections are offset properly. Does not test the shear angle case."""
    s1_offset = 1
    s0 = gf.Section(layer="WG", width=0.5, offset=0, port_names=("o1", "o2"))
    s1 = gf.Section(layer="M1", width=0.5, offset=s1_offset, port_names=("e1", "e2"))
    xs = gf.CrossSection(sections=(s0, s1))
    s = gf.components.straight(cross_section=xs)

    assert s.ports["e1"].center[0] == s.ports["o1"].center[0]
    assert s.ports["e1"].center[1] == s.ports["o1"].center[1] - s1_offset, s.ports[
        "e1"
    ].center[1]

    assert s.ports["e2"].center[0] == s.ports["o2"].center[0]
    assert s.ports["e2"].center[1] == s.ports["o2"].center[1] - s1_offset


def test_extrude_component_along_path() -> None:
    p = gf.path.straight()
    p += gf.path.arc(10)
    p += gf.path.straight()

    # Define a cross-section with a via
    via = gf.cross_section.ComponentAlongPath(
        component=gf.c.rectangle(size=(1, 1), centered=True), spacing=5, padding=2
    )
    s = gf.Section(width=0.5, offset=0, layer=(1, 0), port_names=("in", "out"))
    x = gf.CrossSection(sections=(s,), components_along_path=(via,))

    # Combine the path with the cross-section
    c = gf.path.extrude(p, cross_section=x)
    assert c


def test_extrude_cross_section_list_of_sections() -> None:
    s = gf.Section(width=0.5, offset=0.5, layer="WG")
    xs = gf.CrossSection(sections=(s,))
    c = gf.c.straight(cross_section=xs)
    assert c


def test_extrude_cross_section_width() -> None:
    c = gf.path.extrude(gf.path.straight(length=10), cross_section="strip", width=2.4)
    assert c.ports["o1"].width == 2.4
