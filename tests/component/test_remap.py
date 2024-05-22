from __future__ import annotations

import gdsfactory as gf
from gdsfactory.generic_tech import LAYER


def test_remap_layers() -> None:
    c = gf.components.straight(length=1.221, width=0.5)
    c = gf.add_pins.add_pins_siepic(c, layer=LAYER.PORT, layer_label=LAYER.TEXT)

    c2 = c
    p = 2  # 1 for no bbox
    c2_polygons = c2.get_polygons()
    c2_paths = c2.get_paths(layer=LAYER.PORT)

    assert len(c2_polygons) == p, len(c2_polygons)
    assert len(c2_paths) == 2, len(c2_paths)
    assert LAYER.WG in c2.layers

    c2 = c2.copy()
    c2.remap_layers({LAYER.WG: LAYER.WGN, LAYER.PORT: LAYER.PORTE})
    assert LAYER.WGN in c2.layers, f"{LAYER.WGN} not in {c2.layers}"
    assert LAYER.PORTE in c2.layers, f"{LAYER.PORTE} not in {c2.layers}"
