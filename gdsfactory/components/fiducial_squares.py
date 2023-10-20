from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.typings import Float2, LayerSpec, LayerSpecs


@gf.cell
def fiducial_squares(
    layer: LayerSpec = "WG",
    layers: LayerSpecs | None = None,
    size: Float2 = (5.0, 5.0),
    offset: float = 0.14,
) -> gf.Component:
    """Returns fiducials with two squares.

    Args:
        layer: layer for geometry.
        layers: optional list of layers to duplicate the geometry.
        size: in um.
        offset: between squares in um.
    """
    c = gf.Component()
    layers = layers or [layer]

    for layer in layers:
        layer = gf.get_layer(layer)
        rectangle = gf.components.rectangle(size=size, layer=layer)
        c << rectangle
        r2 = c << rectangle
        r2.move(-np.array(size) - np.array([offset, offset]))

    return c


if __name__ == "__main__":
    c = fiducial_squares()
    c.show()
