from __future__ import annotations

import gdsfactory as gf
from gdsfactory.add_pins import add_pins_siepic


def test_remap_layers():
    c = gf.components.straight(length=1, width=0.5, add_pins=add_pins_siepic)
    c2 = c.remap_layers({gf.LAYER.WG: gf.LAYER.WGN, gf.LAYER.PORT: gf.LAYER.PORTE})

    p = 2
    assert len(c.polygons) == p, len(c.polygons)
    assert len(c2.polygons) == p, len(c2.polygons)
    assert len(c.paths) == 2, len(c.paths)
    assert len(c2.paths) == 2, len(c2.paths)
    assert gf.LAYER.WG in c.layers
    assert gf.LAYER.WGN in c2.layers
    assert gf.LAYER.PORTE in c2.layers
    return c2


if __name__ == "__main__":
    c = test_remap_layers()
    # c.show()

    c = gf.components.straight(length=1, width=0.5, add_pins=add_pins_siepic)
    c2 = c.remap_layers({gf.LAYER.WG: gf.LAYER.WGN, gf.LAYER.PORT: gf.LAYER.PORTE})
    c2.show()
