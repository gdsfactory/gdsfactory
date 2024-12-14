from __future__ import annotations

import gdsfactory as gf
from gdsfactory import Port
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, Size


@gf.cell
def big_device(
    size: Size = (400.0, 400.0),
    nports: int = 16,
    spacing: float = 15.0,
    port_type: str = "electrical",
    cross_section: CrossSectionSpec = "metal_routing",
) -> Component:
    """Big component with N ports on each side.

    Args:
        size: x, y.
        nports: number of ports.
        spacing: in um.
        port_type: optical, electrical, rf, etc.
        cross_section: spec.
    """
    component = gf.Component()

    w, h = size
    dx = w / 2
    dy = h / 2
    n = nports

    xs = gf.get_cross_section(cross_section)
    layer = xs.layer
    assert layer is not None
    width = xs.width
    assert isinstance(width, float)

    points = [(dx, dy), (dx, -dy), (-dx, -dy), (-dx, dy)]
    component.add_polygon(points, layer=layer)
    ports: list[Port] = []

    for i in range(n):
        port = Port(
            name=f"W{i}",
            center=(-dx, (i - n / 2) * spacing),
            orientation=180,
            port_type=port_type,
            cross_section=xs,
            layer=layer,
            width=width,
        )
        ports.append(port)

    for i in range(n):
        port = Port(
            name=f"E{i}",
            center=(dx, (i - n / 2) * spacing),
            orientation=0,
            port_type=port_type,
            cross_section=xs,
            layer=layer,
            width=width,
        )
        ports.append(port)

    for i in range(n):
        port = Port(
            name=f"N{i}",
            center=((i - n / 2) * spacing, dy),
            orientation=90,
            port_type=port_type,
            cross_section=xs,
            layer=layer,
            width=width,
        )
        ports.append(port)

    for i in range(n):
        port = Port(
            name=f"S{i}",
            center=((i - n / 2) * spacing, -dy),
            orientation=-90,
            port_type=port_type,
            cross_section=xs,
            layer=layer,
            width=width,
        )
        ports.append(port)

    component.add_ports(ports)
    component.auto_rename_ports()
    return component


if __name__ == "__main__":
    import gdsfactory as gf

    c = big_device()
    c = gf.routing.add_pads_top(c, fanout_length=None)
    c.show()
