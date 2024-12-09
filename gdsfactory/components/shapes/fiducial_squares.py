from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.typings import Float2, Layers


@gf.cell
def fiducial_squares(
    layers: Layers = ((1, 0),), size: Float2 = (5, 5), offset: float = 0.14
) -> gf.Component:
    """Returns fiducials with two squares.

    Args:
        layers: list of layers.
        size: in um.
        offset: in um.
    """
    c = gf.Component()
    for layer in layers:
        r = c << gf.c.rectangle(size=size, layer=layer)

    for layer in layers:
        r = c << gf.c.rectangle(size=size, layer=layer)
        r.dmove(-np.array(size) - np.array([offset, offset]))

    return c


if __name__ == "__main__":
    c = fiducial_squares()
    c.show()
