from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell
def cross(
    length: float = 10.0,
    width: float = 3.0,
    layer: LayerSpec = "WG",
    port_type: str | None = None,
) -> Component:
    """Returns a cross from two rectangles of length and width.

    Args:
        length: float Length of the cross from one end to the other.
        width: float Width of the arms of the cross.
        layer: layer for geometry.
        port_type: None, optical, electrical.
    """
    layer = gf.get_layer(layer)
    c = gf.Component()
    R = gf.components.rectangle(size=(width, length), layer=layer)
    r1 = c.add_ref(R).drotate(90)
    r2 = c.add_ref(R)
    r1.dcenter = (0, 0)  # type: ignore
    r2.dcenter = (0, 0)  # type: ignore
    c.flatten()

    if port_type:
        prefix = "o" if port_type == "optical" else "e"
        c.add_port(
            f"{prefix}1",
            width=width,
            layer=layer,
            orientation=0,
            center=(+length / 2, 0),
            port_type=port_type,
        )
        c.add_port(
            f"{prefix}2",
            width=width,
            layer=layer,
            orientation=180,
            center=(-length / 2, 0),
            port_type=port_type,
        )
        c.add_port(
            f"{prefix}3",
            width=width,
            layer=layer,
            orientation=90,
            center=(0, length / 2),
            port_type=port_type,
        )
        c.add_port(
            f"{prefix}4",
            width=width,
            layer=layer,
            orientation=270,
            center=(0, -length / 2),
            port_type=port_type,
        )
        c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = cross(port_type="optical")
    c.show()
    # c.pprint_ports()
    # cc = gf.routing.add_fiber_array(component=c)
    # cc.show( )
