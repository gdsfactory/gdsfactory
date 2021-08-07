from typing import Optional, Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.tech import LAYER


@gf.cell
def via(
    width: float = 0.7,
    height: Optional[float] = None,
    pitch: float = 2.0,
    pitch_x: Optional[float] = None,
    pitch_y: Optional[float] = None,
    enclosure: float = 1.0,
    layer: Tuple[int, int] = LAYER.VIA1,
) -> Component:
    """Rectangular via. Defaults to a square via.

    Args:
        width: in x direction
        height: in y direction, defaults to width
        pitch:
        pitch_x: Optional x pitch
        pitch_y: Optional y pitch
        enclosure: inclusion of via
        layer: via layer
    """
    height = height or width
    c = Component()
    c.info["pitch"] = pitch
    c.info["pitch_x"] = pitch_x or pitch
    c.info["pitch_y"] = pitch_y or pitch
    c.info["enclosure"] = enclosure
    c.info["width"] = width
    c.info["height"] = height

    a = width / 2
    b = height / 2

    c.add_polygon([(-a, -b), (a, -b), (a, b), (-a, b)], layer=layer)

    return c


@gf.cell
def via1(**kwargs) -> Component:
    return via(layer=LAYER.VIA1, **kwargs)


@gf.cell
def via2(enclosure: float = 2.0, **kwargs) -> Component:
    return via(layer=LAYER.VIA2, enclosure=enclosure, **kwargs)


@gf.cell
def via3(**kwargs) -> Component:
    return via(layer=LAYER.VIA3, **kwargs)


if __name__ == "__main__":

    c = via()
    # c.pprint()
    print(c)
    c.show()
