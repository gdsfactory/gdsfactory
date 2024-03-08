from __future__ import annotations

import pytest

import gdsfactory as gf


@pytest.mark.skip("TODO")
def test_remap_layers_with_pins() -> None:
    c = gf.components.straight(
        length=1.221,
        width=0.5,
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
    c2 = c1.remap_layers({(1, 0): (2, 0)})

    n_polygons = len(c1.polygons)
    assert len(c1.polygons) == n_polygons, len(c1.polygons)
    assert len(c2.polygons) == n_polygons, len(c2.polygons)
    assert (2, 0) in c2.layers
