from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import Float2, Layer

layer = gf.LAYER.WG


@gf.cell
def width_min(size: Float2 = (0.1, 0.1)) -> Component:
    return gf.components.rectangle(size=size, layer=layer)


@gf.cell
def area_min() -> Component:
    size = (0.2, 0.2)
    return gf.components.rectangle(size=size, layer=layer)


@gf.cell
def gap_min(gap: float = 0.1) -> Component:
    c = gf.Component()
    r1 = c << gf.components.rectangle(size=(1, 1), layer=layer)
    r2 = c << gf.components.rectangle(size=(1, 1), layer=layer)
    r1.xmax = 0
    r2.xmin = gap
    return c


@gf.cell
def separation(
    gap: float = 0.1, layer1: Layer = gf.LAYER.HEATER, layer2: Layer = gf.LAYER.M1
) -> Component:
    c = gf.Component()
    r1 = c << gf.components.rectangle(size=(1, 1), layer=layer1)
    r2 = c << gf.components.rectangle(size=(1, 1), layer=layer2)
    r1.xmax = 0
    r2.xmin = gap
    return c


@gf.cell
def enclosing(
    enclosing: float = 0.1, layer1: Layer = gf.LAYER.VIAC, layer2: Layer = gf.LAYER.M1
) -> Component:
    """Layer1 must be enclosed by layer2 by value.

    checks if layer1 encloses (is bigger than) layer2 by value
    """
    w1 = 1
    w2 = w1 + enclosing
    c = gf.Component()
    c << gf.components.rectangle(size=(w1, w1), layer=layer1, centered=True)
    r2 = c << gf.components.rectangle(size=(w2, w2), layer=layer2, centered=True)
    r2.movex(0.5)
    return c


@gf.cell
def snapping_error(gap: float = 1e-3) -> Component:
    c = gf.Component()
    r1 = c << gf.components.rectangle(size=(1, 1), layer=layer)
    r2 = c << gf.components.rectangle(size=(1, 1), layer=layer)
    r1.xmax = 0
    r2.xmin = gap
    return c


@gf.cell
def errors() -> Component:
    components = [width_min(), gap_min(), separation(), enclosing()]
    c = gf.pack(components, spacing=1.5)
    return c[0]


if __name__ == "__main__":
    # c = width_min()
    # c.write_gds("wmin.gds")
    # c = gap_min()
    # c.write_gds("gmin.gds")
    # c = snapping_error()
    # c.write_gds("snap.gds")

    c = errors()
    c = gf.add_padding_container(c, layers=(gf.LAYER.FLOORPLAN,), default=5)
    c.write_gds("errors.gds")
    c.show(show_ports=True)
