import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.circle import circle
from gdsfactory.types import Layer


@gf.cell
def marker_vertical_te(width: float = 11.0, layer: Layer = (203, 0)) -> Component:
    c = Component()
    c << circle(radius=width / 2, layer=layer)
    return c


@gf.cell
def marker_vertical_tm(width: float = 11.0, layer: Layer = (204, 0)) -> Component:
    c = Component()
    c << circle(radius=width / 2, layer=layer)
    return c


if __name__ == "__main__":
    c = marker_vertical_te()
    c.show()
