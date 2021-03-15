from typing import Any, Iterable, List, Tuple

from numpy import floor

from pp.cell import cell
from pp.component import Component
from pp.layers import LAYER


@cell
def via(
    width: float = 0.7,
    height: float = 0.7,
    period: float = 2.0,
    clearance: float = 1.0,
    layer: Tuple[int, int] = LAYER.VIA1,
) -> Component:
    """Rectangular via"""
    c = Component()
    c.info["period"] = period
    c.info["clearance"] = clearance
    c.info["width"] = width
    c.info["height"] = height

    a = width / 2
    b = height / 2

    c.add_polygon([(-a, -b), (a, -b), (a, b), (-a, b)], layer=layer)

    return c


@cell
def via1(**kwargs) -> Component:
    return via(layer=LAYER.VIA1, **kwargs)


@cell
def via2(**kwargs) -> Component:
    return via(layer=LAYER.VIA2, **kwargs)


@cell
def via3(**kwargs) -> Component:
    return via(layer=LAYER.VIA3, **kwargs)


@cell
def tlm(
    width: float = 11.0,
    height: float = 11.0,
    layers: List[Tuple[int, int]] = (LAYER.M1, LAYER.M2, LAYER.M3),
    vias: Iterable[Any] = (via2, via3),
) -> Component:
    """Rectangular transition thru metal layers

    Args:
        name: component name
        width, height: rectangle parameters
        layers: layers on which to draw rectangles
        vias: vias to use to fill the rectangles
    """

    # assert len(layers) - 1 == len(vias), "tlm: There should be N layers for N-1 vias"

    a = width / 2
    b = height / 2
    rect_pts = [(-a, -b), (a, -b), (a, b), (-a, b)]

    c = Component()
    # Add metal rectangles
    for layer in layers:
        c.add_polygon(rect_pts, layer=layer)

    # Add vias
    for via in vias:
        via = via() if callable(via) else via

        w = via.info["width"]
        h = via.info["height"]
        g = via.info["clearance"]
        period = via.info["period"]

        nb_vias_x = (width - w - 2 * g) / period + 1
        nb_vias_y = (height - h - 2 * g) / period + 1

        nb_vias_x = int(floor(nb_vias_x))
        nb_vias_y = int(floor(nb_vias_y))

        cw = (width - (nb_vias_x - 1) * period - w) / 2
        ch = (height - (nb_vias_y - 1) * period - h) / 2

        x0 = -a + cw + w / 2
        y0 = -b + ch + h / 2

        for i in range(nb_vias_x):
            for j in range(nb_vias_y):
                c.add(via.ref(position=(x0 + i * period, y0 + j * period)))

    return c


if __name__ == "__main__":

    # c = via()
    c = tlm()
    c.pprint()
    c.show()
