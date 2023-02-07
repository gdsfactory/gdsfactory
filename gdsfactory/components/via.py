from __future__ import annotations

from typing import Optional, Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell
def via(
    size: Tuple[float, float] = (0.7, 0.7),
    spacing: Tuple[float, float] = (2.0, 2.0),
    enclosure: float = 1.0,
    layer: LayerSpec = "VIAC",
    bbox_layers: Optional[Tuple[Tuple[int, int], ...]] = None,
    bbox_offset: float = 0,
) -> Component:
    """Rectangular via.

    Defaults to a square via.

    Args:
        size: in x, y direction.
        spacing: pitch_x, pitch_y.
        enclosure: inclusion of via.
        layer: via layer.
        bbox_layers: layers for the bounding box.
        bbox_offset: in um.

    .. code::

        enclosure
        _________________________________________
        |<--->                                  |
        |                      size[0]          |
        |                      <----->          |
        |      ______          ______           |
        |     |      |        |      |          |
        |     |      |        |      |  size[1] |
        |     |______|        |______|          |
        |      <------------->                  |
        |         spacing[0]                    |
        |_______________________________________|
    """
    c = Component()
    c.info["spacing"] = spacing
    c.info["enclosure"] = enclosure
    c.info["size"] = size

    width, height = size
    a = width / 2
    b = height / 2
    c.add_polygon([(-a, -b), (a, -b), (a, b), (-a, b)], layer=layer)

    bbox_layers = bbox_layers or []
    a = (width + bbox_offset) / 2
    b = (height + bbox_offset) / 2
    for layer in bbox_layers:
        c.add_polygon([(-a, -b), (a, -b), (a, b), (-a, b)], layer=layer)

    return c


viac = gf.partial(via, layer="VIAC")
via1 = gf.partial(via, layer="VIA1", enclosure=2)
via2 = gf.partial(via, layer="VIA2")


if __name__ == "__main__":
    c = via()
    # c.pprint()
    print(c)
    c.show(show_ports=True)
