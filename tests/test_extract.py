from __future__ import annotations

import gdsfactory as gf
from gdsfactory.add_pins import add_pins_siepic


def test_extract() -> None:
    c = gf.components.straight(
        length=11.124,
        width=0.5,
        bbox_layers=[gf.LAYER.WGCLAD],
        bbox_offsets=[3],
        with_bbox=True,
        cladding_layers=None,
        add_pins=add_pins_siepic,
        add_bbox=None,
    )
    c2 = c.extract(layers=[gf.LAYER.PORT])

    assert len(c.polygons) == 2, len(c.polygons)
    assert len(c2.polygons) == 0, len(c2.polygons)

    assert len(c.paths) == 2, len(c.paths)
    assert len(c2.paths) == 2, len(c2.paths)
    assert gf.LAYER.WGCLAD in c.layers
    assert gf.LAYER.PORT in c2.layers


if __name__ == "__main__":
    c = test_extract()
    c.show()

    c = gf.components.straight(
        length=10,
        width=0.5,
        bbox_layers=[gf.LAYER.WGCLAD],
        bbox_offsets=[3],
        with_bbox=True,
        cladding_layers=None,
        add_pins=add_pins_siepic,
        add_bbox=None,
    )
    c2 = c.extract(layers=[gf.LAYER.PORT])
