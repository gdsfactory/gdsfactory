"""Write GDS with sample errors."""

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.typings import Float2, Layer

from gpdk import LAYER

layer = LAYER.WG
layer1 = LAYER.WG


@gf.cell
def _width_min(size: Float2 = (0.1, 0.1)) -> Component:
    return gf.components.rectangle(size=size, layer=layer)


@gf.cell
def _area_min() -> Component:
    size = (0.2, 0.2)
    return gf.components.rectangle(size=size, layer=layer)


@gf.cell
def _gap_min(gap: float = 0.1) -> Component:
    c = gf.Component()
    r1 = c << gf.components.rectangle(size=(1, 1), layer=layer)
    r2 = c << gf.components.rectangle(size=(1, 1), layer=layer)
    r1.xmax = 0
    r2.xmin = gap
    return c


@gf.cell
def _separation(
    gap: float = 0.1, layer1: Layer = (47, 0), layer2: Layer = (41, 0)
) -> Component:
    c = gf.Component()
    r1 = c << gf.components.rectangle(size=(1, 1), layer=layer1)
    r2 = c << gf.components.rectangle(size=(1, 1), layer=layer2)
    r1.xmax = 0
    r2.xmin = gap
    return c


@gf.cell
def _enclosing(
    enclosing: float = 0.1, layer1: Layer = (40, 0), layer2: Layer = (41, 0)
) -> Component:
    """Layer1 must be enclosed by layer2 by value.

    checks if layer1 encloses (is bigger than) layer2 by value
    """
    w1 = 1
    w2 = w1 + enclosing
    c = gf.Component()
    _ = c << gf.components.rectangle(size=(w1, w1), layer=layer1, centered=True)
    r2 = c << gf.components.rectangle(size=(w2, w2), layer=layer2, centered=True)
    r2.movex(0.5)
    return c


@gf.cell
def _sample_drc_errors() -> Component:
    components = [_width_min(), _separation(), _enclosing()]
    components += [_width_min()] * 100

    for gap in np.linspace(0.1, 0.2, 5):
        components.append(_gap_min(gap=gap))

    c = gf.Component()
    _ = c << gf.pack(components, spacing=1)[0]

    return c
