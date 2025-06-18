from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.typings import Float2, LayerSpecs


@gf.cell_with_module_name
def fiducial_squares(
    layers: LayerSpecs = ("WG",), size: Float2 = (5, 5), offset: float = 0.14
) -> gf.Component:
    """Returns fiducials with two squares.

    Args:
        layers: list of layers to draw the squares.
        size: size of each square in um.
        offset: space between squares in x and y.
    """
    c = gf.Component()

    dx, dy = (np.array(size) + np.array([offset, offset])) / 2

    for layer in layers:
        r = c << gf.c.rectangle(size=size, layer=layer, centered=True)
        r.move((dx, dy))

    for layer in layers:
        r = c << gf.c.rectangle(size=size, layer=layer, centered=True)
        r.move((-dx, -dy))

    return c


if __name__ == "__main__":
    c = fiducial_squares()
    c.show()
