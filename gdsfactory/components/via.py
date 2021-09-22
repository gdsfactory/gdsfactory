from typing import Optional, Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.tech import LAYER


@gf.cell
def via(
    size: Tuple[float, float] = (0.7, 0.7),
    spacing: Tuple[float, float] = (2.0, 2.0),
    enclosure: float = 1.0,
    layer: Tuple[int, int] = LAYER.VIAC,
    layers_cladding: Optional[Tuple[Tuple[int, int], ...]] = None,
    cladding_offset: float = 0,
) -> Component:
    """Rectangular via. Defaults to a square via.

    Args:
        size: in x, y direction
        spacing: pitch_x, pitch_y
        enclosure: inclusion of via
        layer: via layer
        layers_cladding:
        cladding_offset

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

    layers_cladding = layers_cladding or []
    a = (width + cladding_offset) / 2
    b = (height + cladding_offset) / 2
    for layer in layers_cladding:
        c.add_polygon([(-a, -b), (a, -b), (a, b), (-a, b)], layer=layer)

    return c


viac = gf.partial(via, layer=LAYER.VIAC)
via1 = gf.partial(via, layer=LAYER.VIA1, enclosure=2)
via2 = gf.partial(via, layer=LAYER.VIA2)


if __name__ == "__main__":

    c = via()
    # c.pprint
    print(c)
    c.show()
