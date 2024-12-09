from __future__ import annotations

from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory import Port
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, Size


@gf.cell
def big_device(
    size: Size = (400.0, 400.0),
    nports: int = 16,
    spacing: float = 15.0,
    port_type: str = "optical",
    cross_section: CrossSectionSpec = "strip",
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
    n = nports

    xs = gf.get_cross_section(cross_section)
    layer = xs.layer
    assert layer is not None
    width = xs.width

    points = [(dx, dy), (dx, -dy), (-dx, -dy), (-dx, dy)]
    component.add_polygon(points, layer=layer)
    ports: list[Port] = []

    create_port_with_port_settings = partial(
        Port, port_type=port_type, cross_section=xs, layer=layer, width=width
    )

    for i in range(n):
        port = create_port_with_port_settings(
            name=f"W{i}",
            center=tuple(p0 + (-dx, (i - n / 2) * spacing)),
            orientation=180,
        )
        ports.append(port)

    for i in range(n):
        port = create_port_with_port_settings(
            name=f"E{i}",
            center=tuple(p0 + (dx, (i - n / 2) * spacing)),
            orientation=0,
        )
        ports.append(port)

    for i in range(n):
        port = create_port_with_port_settings(
            name=f"N{i}",
            center=tuple(p0 + ((i - n / 2) * spacing, dy)),
            orientation=90,
        )
        ports.append(port)

    for i in range(n):
        port = create_port_with_port_settings(
            name=f"S{i}",
            center=tuple(p0 + ((i - n / 2) * spacing, -dy)),
            orientation=-90,
        )
        ports.append(port)

    component.add_ports(ports)
    component.auto_rename_ports()
    return component


if __name__ == "__main__":
    import gdsfactory as gf

    # pdk = gf.pdk.get_active_pdk()
    # pdk.gds_write_settings.flatten_invalid_refs = False
    c = big_device()
    c = gf.routing.add_fiber_array(c)
    c.show()
    # lyrdb = c.connectivity_check()
    # filepath = gf.config.home / "errors.lyrdb"
    # lyrdb.save(filepath)
    # gf.show(c, lyrdb=filepath)
    # c = c.flatten_invalid_refs()
    # c.write_gds("./test.gds")
