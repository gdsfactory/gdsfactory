from __future__ import annotations

from typing import Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory import LAYER, Port
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec


@gf.cell
def big_device(
    size: Tuple[float, float] = (400.0, 400.0),
    nports: int = 16,
    spacing: float = 15.0,
    layer: Tuple[int, int] = LAYER.WG,
    wg_width: float = 0.5,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Big component with N ports on each side.

    Args:
        size: x, y.
        nports: number of ports.
        spacing: in um.
        layer: spec.
        wg_width: waveguide width in um.
        cross_section: spec.

    """
    component = gf.Component()
    p0 = np.array((0, 0))

    w, h = size
    dx = w / 2
    dy = h / 2
    N = nports

    points = [[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]]
    component.add_polygon(points, layer=layer)

    for i in range(N):
        port = Port(
            name=f"W{i}",
            center=p0 + (-dx, (i - N / 2) * spacing),
            orientation=180,
            layer=layer,
            width=wg_width,
        )
        component.add_port(port)

    for i in range(N):
        port = Port(
            name=f"E{i}",
            center=p0 + (dx, (i - N / 2) * spacing),
            orientation=0,
            layer=layer,
            width=wg_width,
        )
        component.add_port(port)

    for i in range(N):
        port = Port(
            name=f"N{i}",
            center=p0 + ((i - N / 2) * spacing, dy),
            orientation=90,
            layer=layer,
            width=wg_width,
        )
        component.add_port(port)

    for i in range(N):
        port = Port(
            name=f"S{i}",
            center=p0 + ((i - N / 2) * spacing, -dy),
            orientation=-90,
            layer=layer,
            width=wg_width,
        )
        component.add_port(port)

    component = gf.add_pins.add_pins_inside1nm(component)
    component.auto_rename_ports()
    xs = gf.get_cross_section(cross_section)
    if xs.add_pins:
        xs.add_pins(component)
    return component


if __name__ == "__main__":
    import gdsfactory as gf

    pdk = gf.pdk.get_active_pdk()
    pdk.gds_write_settings.flatten_invalid_refs = False
    c = big_device()
    c = gf.routing.add_fiber_array(c)
    # c.write_gds("./test.gds")
    c.show(show_ports=False)
