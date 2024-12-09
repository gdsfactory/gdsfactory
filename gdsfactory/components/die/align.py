from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from gdsfactory.typings import ComponentSpec, LayerSpec


@gf.cell
def align_wafer(
    width: float = 10.0,
    spacing: float = 10.0,
    cross_length: float = 80.0,
    layer: LayerSpec = "WG",
    layer_cladding: tuple[int, int] | None = None,
    square_corner: str = "bottom_left",
) -> Component:
    """Returns cross inside a frame to align wafer.

    Args:
        width: in um.
        spacing: in um.
        cross_length: for the cross.
        layer: for the cross.
        layer_cladding: optional.
        square_corner: bottom_left, bottom_right, top_right, top_left.
    """
    layer = gf.get_layer(layer)
    c = Component()
    cross = gf.components.cross(length=cross_length, width=width, layer=layer)
    c.add_ref(cross)

    b = cross_length / 2 + spacing + width / 2
    w = width

    rh = rectangle(size=(2 * b + w, w), layer=layer, centered=True)
    rv = rectangle(size=(w, 2 * b), layer=layer, centered=True)

    rtop = c.add_ref(rh)
    rbot = c.add_ref(rh)

    rtop.dmovey(+b)
    rbot.dmovey(-b)

    rl = c.add_ref(rv)
    rr = c.add_ref(rv)
    rl.dmovex(-b)
    rr.dmovex(+b)

    wsq = (cross_length + 2 * spacing) / 4
    square_mark = c << rectangle(size=(wsq, wsq), layer=layer, centered=True)
    a = width / 2 + wsq / 2 + spacing

    corner_to_position = {
        "bottom_left": (-a, -a),
        "bottom_right": (a, -a),
        "top_right": (a, a),
        "top_left": (-a, a),
    }

    square_mark.dmove(corner_to_position[square_corner])

    if layer_cladding:
        rc_tile_excl = rectangle(
            size=(2 * (b + spacing), 2 * (b + spacing)),
            layer=layer_cladding,
            centered=True,
        )
        c.add_ref(rc_tile_excl)

    return c


@gf.cell
def add_frame(
    component: ComponentSpec = rectangle,
    width: float = 10.0,
    spacing: float = 10.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns component with a frame around it.

    Args:
        component: Component to frame.
        width: of the frame.
        spacing: of component to frame.
        layer: frame layer.
    """
    c = Component()
    layer = gf.get_layer(layer)
    component = gf.get_component(component)
    cref = c.add_ref(component)
    cref.dx = 0
    cref.dy = 0
    y = max([component.dxsize, component.dysize]) / 2 + spacing + width / 2
    x = y
    w = width

    rh = rectangle(size=(2 * y + w, w), layer=layer, centered=True)
    rtop = c.add_ref(rh)
    rbot = c.add_ref(rh)
    rtop.dmovey(+y)
    rbot.dmovey(-y)

    rv = rectangle(size=(w, 2 * y), layer=layer, centered=True)
    rl = c.add_ref(rv)
    rr = c.add_ref(rv)
    rl.dmovex(-x)
    rr.dmovex(+x)
    c.flatten()
    return c


if __name__ == "__main__":
    c = gf.components.straight()
    c = add_frame(component=c)
    # c = align_wafer()
    c.show()
