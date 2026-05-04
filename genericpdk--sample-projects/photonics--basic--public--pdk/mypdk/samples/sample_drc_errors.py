"""Write GDS with sample errors."""

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.typings import Layer

from mypdk import LAYER

layer = LAYER.WG
layer1 = LAYER.WG


@gf.cell
def _width_min(width=0.1) -> Component:
    """Minimum width for a waveguide."""
    size = (width, width)
    return gf.components.rectangle(size=size, layer=layer)


@gf.cell
def _area_min() -> Component:
    """Minimum area for a waveguide."""
    size = (0.2, 0.2)
    return gf.components.rectangle(size=size, layer=layer)


@gf.cell
def _gap_min(gap: float = 0.1) -> Component:
    """Minimum gap between two waveguides."""
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
    """Minimum separation between two layers."""
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
def _snapping_error(gap: float = 1e-3) -> Component:
    """Snapping error."""
    c = gf.Component()
    r1 = c << gf.components.rectangle(size=(1, 1), layer=layer)
    r2 = c << gf.components.rectangle(size=(1, 1), layer=layer)
    r1.xmax = 0
    r2.xmin = gap
    return c


@gf.cell
def _sample_drc_errors() -> Component:
    """Write GDS with sample errors."""
    components = [_enclosing()]
    components += [_width_min(width=0.1)] * 100

    min_gap = 0.1
    max_gap = 0.2

    for gap in np.linspace(min_gap, max_gap, 5):
        components.append(_gap_min(gap=gap))

    c = gf.pack(components, spacing=2)[0]
    return c
