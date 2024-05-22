from __future__ import annotations

import gdsfactory as gf
from gdsfactory.generic_tech import LAYER


def test_extract() -> None:
    xs = gf.cross_section.strip(
        width=0.5,
        bbox_layers=(LAYER.WGCLAD,),
        bbox_offsets=(3,),
    )

    c = gf.components.straight(
        length=11.124,
        cross_section=xs,
    )
    c2 = c.extract(layers=[LAYER.WGCLAD])
    p = 1
    c2_polygons = c2.get_polygons()
    assert len(c2_polygons) == p, len(c2_polygons)
    assert LAYER.WGCLAD in c2.layers, c2.layers


if __name__ == "__main__":
    test_extract()
