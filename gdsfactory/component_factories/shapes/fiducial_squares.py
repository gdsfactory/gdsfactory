from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.typings import Float2, LayerSpecs


def fiducial_squares_factory(
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
