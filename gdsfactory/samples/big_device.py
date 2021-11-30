from typing import Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory import LAYER, Port
from gdsfactory.component import Component


@gf.cell
def big_device(
    size: Tuple[float, float] = (400.0, 400.0),
    nports: int = 16,
    spacing: float = 15.0,
    layer: Tuple[int, int] = LAYER.WG,
    wg_width: float = 0.5,
) -> Component:
    """Big component with N ports on each side

    Args:
        size:
        nports: number of ports
        spacing:
        layer:
        wg_width: waveguide width
    """
    component = gf.Component()
    p0 = np.array((0, 0))

    w, h = size
    dx = w / 2
    dy = h / 2
    N = nports

    points = [[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]]
    component.add_polygon(points, layer=layer)
    for i in range(N):
        port = Port(
            name=f"W{i}",
            midpoint=p0 + (-dx, (i - N / 2) * spacing),
            orientation=180,
            layer=layer,
            width=wg_width,
        )
        component.add_port(port)

    for i in range(N):
        port = Port(
            name=f"E{i}",
            midpoint=p0 + (dx, (i - N / 2) * spacing),
            orientation=0,
            layer=layer,
            width=wg_width,
        )
        component.add_port(port)

    for i in range(N):
        port = Port(
            name=f"N{i}",
            midpoint=p0 + ((i - N / 2) * spacing, dy),
            orientation=90,
            layer=layer,
            width=wg_width,
        )
        component.add_port(port)

    for i in range(N):
        port = Port(
            name=f"S{i}",
            midpoint=p0 + ((i - N / 2) * spacing, -dy),
            orientation=-90,
            layer=layer,
            width=wg_width,
        )
        component.add_port(port)
    return component


if __name__ == "__main__":
    c = big_device()
    c.show()
