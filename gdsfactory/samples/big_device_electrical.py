from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory import Port
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec


@gf.cell
def big_device(
    size: tuple[float, float] = (400.0, 400.0),
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
    p0 = np.array((0, 0))

    w, h = size
    dx = w / 2
    dy = h / 2
    N = nports

    xs = gf.get_cross_section(cross_section)
    layer = xs.layer
    width = xs.width
    port_settings = dict(
        port_type=port_type, cross_section=xs, layer=layer, width=width
    )

    points = [[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]]
    component.add_polygon(points, layer=layer)
    ports = []

    for i in range(N):
        port = Port(
            name=f"W{i}",
            center=p0 + (-dx, (i - N / 2) * spacing),
            orientation=180,
            **port_settings,
        )
        ports.append(port)

    for i in range(N):
        port = Port(
            name=f"E{i}",
            center=p0 + (dx, (i - N / 2) * spacing),
            orientation=0,
            **port_settings,
        )
        ports.append(port)

    for i in range(N):
        port = Port(
            name=f"N{i}",
            center=p0 + ((i - N / 2) * spacing, dy),
            orientation=90,
            **port_settings,
        )
        ports.append(port)

    for i in range(N):
        port = Port(
            name=f"S{i}",
            center=p0 + ((i - N / 2) * spacing, -dy),
            orientation=-90,
            **port_settings,
        )
        ports.append(port)

    component.add_ports(ports)
    component.auto_rename_ports()
    return component


if __name__ == "__main__":
    import gdsfactory as gf

    c = big_device()
    c = gf.routing.add_pads_top(c, fanout_length=500)
    c.show()
