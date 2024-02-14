from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import Callable, LayerSpec, LayerSpecs


@gf.cell
def cross(
    length: float = 10.0,
    width: float = 3.0,
    layer: LayerSpec = "WG",
    layers: LayerSpecs | None = None,
    port_type: str | None = None,
    post_process: Callable | None = None,
) -> Component:
    """Returns a cross from two rectangles of length and width.

    Args:
        length: float Length of the cross from one end to the other.
        width: float Width of the arms of the cross.
        layer: layer for geometry.
        layers: Optional. List of layers for geometry.
        port_type: None, optical, electrical.
        post_process: Optional. Function to post process the component.
    """
    c = gf.Component()
    layers = layers or [layer]

    for layer in layers:
        layer = gf.get_layer(layer)
        R = gf.components.rectangle(size=(width, length), layer=layer)
        r1 = c.add_ref(R).rotate(90)
        r2 = c.add_ref(R)
        r1.center = (0, 0)
        r2.center = (0, 0)

        if port_type:
            c.add_port(
                1,
                width=width,
                layer=layer,
                orientation=0,
                center=(+length / 2, 0),
                port_type=port_type,
            )
            c.add_port(
                2,
                width=width,
                layer=layer,
                orientation=180,
                center=(-length / 2, 0),
                port_type=port_type,
            )
            c.add_port(
                3,
                width=width,
                layer=layer,
                orientation=90,
                center=(0, length / 2),
                port_type=port_type,
            )
            c.add_port(
                4,
                width=width,
                layer=layer,
                orientation=270,
                center=(0, -length / 2),
                port_type=port_type,
            )
        c.auto_rename_ports()
    if post_process:
        post_process(c)
    return c


if __name__ == "__main__":
    layers = {(1, 0), (2, 0)}
    c = cross(layers=layers)
    c.show(show_ports=True)
    # c.pprint_ports()
    # cc = gf.routing.add_fiber_array(component=c)
    # cc.show(show_ports=True)
