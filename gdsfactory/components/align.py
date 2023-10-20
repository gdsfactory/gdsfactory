from __future__ import annotations

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from gdsfactory.typings import ComponentSpec, LayerSpec, LayerSpecs


@cell
def align_wafer(
    width: float = 10.0,
    spacing: float = 10.0,
    cross_length: float = 80.0,
    layer: LayerSpec = "WG",
    layers: LayerSpecs | None = None,
    layer_cladding: tuple[int, int] | None = None,
    square_corner: str = "bottom_left",
) -> Component:
    """Returns cross inside a frame to align wafer.

    Args:
        width: in um.
        spacing: in um.
        cross_length: for the cross.
        layer: for the cross.
        layers: Optional. List of layers for cross.
        layer_cladding: optional.
        square_corner: bottom_left, bottom_right, top_right, top_left.
    """
    c = Component()
    layers = layers or [layer]

    for layer in layers:
        layer = gf.get_layer(layer)
        cross = gf.components.cross(length=cross_length, width=width, layer=layer)
        c.add_ref(cross)

        b = cross_length / 2 + spacing + width / 2
        w = width

        rh = rectangle(size=(2 * b + w, w), layer=layer, centered=True)
        rtop = c.add_ref(rh)
        rbot = c.add_ref(rh)
        rtop.movey(+b)
        rbot.movey(-b)

        rv = rectangle(size=(w, 2 * b), layer=layer, centered=True)
        rl = c.add_ref(rv)
        rr = c.add_ref(rv)
        rl.movex(-b)
        rr.movex(+b)

        wsq = (cross_length + 2 * spacing) / 4
        square_mark = c << rectangle(size=(wsq, wsq), layer=layer, centered=True)
        a = width / 2 + wsq / 2 + spacing

        corner_to_position = {
            "bottom_left": (-a, -a),
            "bottom_right": (a, -a),
            "top_right": (a, a),
            "top_left": (-a, a),
        }

        square_mark.move(corner_to_position[square_corner])

        if layer_cladding:
            rc_tile_excl = rectangle(
                size=(2 * (b + spacing), 2 * (b + spacing)),
                layer=layer_cladding,
                centered=True,
            )
            c.add_ref(rc_tile_excl)

    return c


@cell
def add_frame(
    component: ComponentSpec = rectangle,
    width: float = 10.0,
    spacing: float = 10.0,
    layer: LayerSpec = "WG",
    layers: LayerSpecs | None = None,
) -> Component:
    """Returns component with a frame around it.

    Args:
        component: Component to frame.
        width: of the frame.
        spacing: of component to frame.
        layer: frame layer.
        layers: Optional. List of layers for geometry.
    """
    c = Component()
    component = gf.get_component(component)
    layers = layers or [layer]

    for layer in layers:
        layer = gf.get_layer(layer)
        cref = c.add_ref(component)
        cref.move(-c.size_info.center)
        y = (
            max([component.size_info.height, component.size_info.width]) / 2
            + spacing
            + width / 2
        )
        x = y
        w = width

        rh = rectangle(size=(2 * y + w, w), layer=layer, centered=True)
        rtop = c.add_ref(rh)
        rbot = c.add_ref(rh)
        rtop.movey(+y)
        rbot.movey(-y)

        rv = rectangle(size=(w, 2 * y), layer=layer, centered=True)
        rl = c.add_ref(rv)
        rr = c.add_ref(rv)
        rl.movex(-x)
        rr.movex(+x)
        c.absorb(cref)
    return c


if __name__ == "__main__":
    layers = {(1, 0), (2, 0)}

    # c = gf.components.straight()
    # c = add_frame(component=c)
    c = align_wafer(layers=layers)
    c = add_frame(c, layers=layers)
    c.show(show_ports=True)
