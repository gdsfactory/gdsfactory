from __future__ import annotations

import gdsfactory as gf
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


if __name__ == "__main__":
    test_transition_cross_section()
