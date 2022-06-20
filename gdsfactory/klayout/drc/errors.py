import gdsfactory as gf
from gdsfactory.component import Component

layer = gf.LAYER.WG


@gf.cell
def width_min(size=(0.1, 0.1)) -> Component:
    return gf.components.rectangle(size=size, layer=layer)


@gf.cell
def gap_min(gap=0.1) -> Component:
    c = gf.Component()
    r1 = c << gf.components.rectangle(size=(1, 1), layer=layer)
    r2 = c << gf.components.rectangle(size=(1, 1), layer=layer)
    r1.xmax = 0
    r2.xmin = gap
    return c


@gf.cell
def separation(gap=0.1, layer1=gf.LAYER.HEATER, layer2=gf.LAYER.M1) -> Component:
    c = gf.Component()
    r1 = c << gf.components.rectangle(size=(1, 1), layer=layer1)
    r2 = c << gf.components.rectangle(size=(1, 1), layer=layer2)
    r1.xmax = 0
    r2.xmin = gap
    return c


@gf.cell
def enclosing(enclosing=0.1, layer1=gf.LAYER.M1, layer2=gf.LAYER.VIAC) -> Component:
    """Layer1 must be enclosed by layer2 by value.
    checks if layer1 encloses (is bigger than) layer2 by value
    """
    w1 = 1
    w2 = w1 - enclosing
    c = gf.Component()
    c << gf.components.rectangle(size=(w1, w1), layer=layer1, centered=True)
    c << gf.components.rectangle(size=(w2, w2), layer=layer2, centered=True)
    return c


@gf.cell
def snapping_error(gap=1e-3) -> Component:
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
    c.write_gds("errors.gds")
    c.show()
