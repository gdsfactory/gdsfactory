from typing import Optional, Tuple

from numpy import floor

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.via import via2, via3
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentOrFactory, Layer


@gf.cell
def via_stack(
    width: float = 11.0,
    height: Optional[float] = None,
    layers: Tuple[Layer, ...] = (LAYER.M1, LAYER.M2, LAYER.M3),
    vias: Tuple[ComponentOrFactory, ...] = (via2, via3),
    port_orientation: int = 180,
) -> Component:
    """Rectangular via_stack

    Args:
        width: in x direction
        height: in y direction, defaults to width
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
    c.add_port(
        name=port_name, width=port_width, orientation=port_orientation, port_type="dc"
    )

    return c


if __name__ == "__main__":
    c = via_stack()
    print(c)
    c.show()
