from __future__ import annotations

import pytest

import gdsfactory as gf
from gdsfactory.add_pins import add_bbox, add_pins_siepic


@pytest.mark.skip("TODO")
def test_remap_layers_with_pins() -> None:
    c = gf.components.straight(
        length=1.221,
        width=0.5,
        add_pins=add_pins_siepic,
        add_bbox=add_bbox,
    )
    c2 = c.remap_layers({(1, 0): (34, 0), (1, 10): (1, 11)})
    p = 2  # 1 for no bbox
    assert len(c.polygons) == p, len(c.polygons)
    assert len(c2.polygons) == p, len(c2.polygons)
    assert len(c.paths) == 2, len(c.paths)
    assert len(c2.paths) == 2, len(c2.paths)
    assert (1, 0) in c.layers
    assert (34, 0) in c2.layers
    assert (1, 11) in c2.layers


def test_remap_layers() -> None:
    c1 = gf.Component()
    c1.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=(1, 0))
    n_polygons = 1

    c2 = c1.copy()
    c2 = c2.remap_layers({(1, 0): (2, 0)})

    c1_polygons = len(c1.get_polygons()[(1, 0)])
    c2_polygons = len(c2.get_polygons()[(2, 0)])

    assert c1_polygons == n_polygons, c1_polygons
    assert c2_polygons == n_polygons, c2_polygons
    assert (1, 0) in c1.layers, c1.layers
    assert (2, 0) in c2.layers, c2.layers


if __name__ == "__main__":
    # test_remap_layers()
    # c.show()

    c1 = gf.Component()
    c1.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=(1, 0))
    n_polygons = 1

    c2 = c1.copy()
    c2 = c2.remap_layers({(1, 0): (2, 0)})

    c1_polygons = len(c1.get_polygons()[(1, 0)])
    c2_polygons = len(c2.get_polygons()[(2, 0)])
    assert c1_polygons == n_polygons, c1_polygons  # FIXME! polygons get destroyed
    assert c2_polygons == n_polygons, c2_polygons
