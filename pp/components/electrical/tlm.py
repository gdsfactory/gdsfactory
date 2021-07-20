from typing import Optional, Tuple

from numpy import floor

import pp
from pp.component import Component
from pp.layers import LAYER
from pp.types import ComponentOrFactory, Layer


@pp.cell_with_validator
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
        width:
        height: Defaults to width
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


@pp.cell_with_validator
def via1(**kwargs) -> Component:
    return via(layer=LAYER.VIA1, **kwargs)


@pp.cell_with_validator
def via2(enclosure: float = 2.0, **kwargs) -> Component:
    return via(layer=LAYER.VIA2, enclosure=enclosure, **kwargs)


@pp.cell_with_validator
def via3(**kwargs) -> Component:
    return via(layer=LAYER.VIA3, **kwargs)


@pp.cell_with_validator
def tlm(
    width: float = 11.0,
    height: Optional[float] = None,
    layers: Tuple[Layer, ...] = (LAYER.M1, LAYER.M2, LAYER.M3),
    vias: Tuple[ComponentOrFactory, ...] = (via2, via3),
    port_orientation: int = 180,
) -> Component:
    """Rectangular transition thru metal layers

    Args:
        width: width
        height: defaults to width
        layers: layers on which to draw rectangles
        vias: vias to use to fill the rectangles
        port_orientation: 180: W0, 0: E0, 90: N0, 270: S0
    """
    height = height or width

    a = width / 2
    b = height / 2
    rect_pts = [(-a, -b), (a, -b), (a, b), (-a, b)]

    c = Component()
    c.height = height

    # Add metal rectangles
    for layer in layers:
        c.add_polygon(rect_pts, layer=layer)

    # Add vias
    for via in vias:
        via = via() if callable(via) else via

        w = via.info["width"]
        h = via.info["height"]
        g = via.info["enclosure"]
        pitch_x = via.info["pitch_x"]
        pitch_y = via.info["pitch_y"]

        nb_vias_x = (width - w - 2 * g) / pitch_x + 1
        nb_vias_y = (height - h - 2 * g) / pitch_y + 1

        nb_vias_x = int(floor(nb_vias_x)) or 1
        nb_vias_y = int(floor(nb_vias_y)) or 1

        cw = (width - (nb_vias_x - 1) * pitch_x - w) / 2
        ch = (height - (nb_vias_y - 1) * pitch_y - h) / 2

        x0 = -a + cw + w / 2
        y0 = -b + ch + h / 2

        for i in range(nb_vias_x):
            for j in range(nb_vias_y):
                c.add(via.ref(position=(x0 + i * pitch_x, y0 + j * pitch_y)))

    if port_orientation == 0:
        port_name = "E0"
        port_width = height
    elif port_orientation == 180:
        port_name = "W0"
        port_width = height
    elif port_orientation == 90:
        port_name = "N0"
        port_width = width
    elif port_orientation == 270:
        port_name = "S0"
        port_width = width
    else:
        raise ValueError(
            f"Invalid port_orientation = {port_orientation} not in [0, 90, 180, 270]"
        )
    c.add_port(name=port_name, width=port_width, orientation=port_orientation)

    return c


if __name__ == "__main__":

    # c = via()
    c = tlm()
    # c.pprint()
    print(c)
    c.show()
