from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gdsdiff.gdsdiff import xor_polygons


def test_differences() -> None:
    straight = gf.partial(
        gf.components.straight,
        with_bbox=True,
        cladding_layers=None,
        add_pins=None,
        add_bbox=None,
    )
    c1 = straight(length=2)
    c2 = straight(length=3)
    c = xor_polygons(c1, c2, hash_geometry=False)
    area = c.area()
    assert area == 0.5, area


def test_no_differences() -> None:
    straight = gf.partial(
        gf.components.straight,
        with_bbox=True,
        cladding_layers=None,
        add_pins=None,
        add_bbox=None,
    )
    c1 = straight(length=2)
    c2 = straight(length=2)
    c = xor_polygons(c1, c2, hash_geometry=False)
    area = c.area()
    assert area == 0, area


if __name__ == "__main__":
    # test_no_differences()
    # test_differences()
    c1 = gf.components.mzi(length_x=3)
    c2 = gf.components.mzi(length_x=2)
    c = xor_polygons(c1, c2, hash_geometry=False)
    c.show(show_ports=True)
