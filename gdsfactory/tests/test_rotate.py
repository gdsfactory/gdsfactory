from __future__ import annotations

import numpy.testing as npt

import gdsfactory as gf


def test_rotate() -> None:
    c1 = gf.components.straight()
    c1r = c1.rotate()

    c2 = gf.components.straight()
    c2r = c2.rotate()

    assert c1.uid == c2.uid
    assert c1r.uid == c2r.uid


def test_rotate_port() -> None:
    port_center_original = (10, 0)
    port_center_expected = (0, 10)
    rotation = 90
    port_orientation_expected = rotation
    c1 = gf.Component()
    p1 = c1.add_port(
        "o1", center=port_center_original, width=5, layer="WG", orientation=0
    )
    p2 = p1.copy()
    p2.orientation = None
    c1.add_port("e1", port=p2)
    c = gf.Component()
    c1_ref = c << c1
    c1_ref.rotate(port_orientation_expected)
    port_center_actual = c1_ref["o1"].center
    port_orientation_actual = c1_ref["o1"].orientation
    npt.assert_almost_equal(port_center_actual, port_center_expected)
    assert port_orientation_actual == port_orientation_expected

    port_center_actual_no_orientation = c1_ref["e1"].center
    port_orientation_actual_no_orientation = c1_ref["e1"].orientation
    npt.assert_almost_equal(port_center_actual_no_orientation, port_center_expected)
    assert port_orientation_actual_no_orientation is None


if __name__ == "__main__":
    c1 = gf.components.straight()
    c1r = c1.rotate()

    c2 = gf.components.straight()
    c2r = c2.rotate()

    assert c1.uid == c2.uid
    assert c1r.uid == c2r.uid
    c2r.show()
