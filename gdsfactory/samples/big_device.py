from __future__ import annotations

from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.routing.add_fiber_array import add_fiber_array
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
    from gdsfactory.pdk import get_layer

    component = gf.Component()
    p0 = np.array((0, 0), dtype=np.float64)

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

    create_port_with_port_settings = partial(
        component.add_port,
        port_type=port_type,
        cross_section=xs,
        layer=get_layer(layer),
        width=width,
    )

    for i in range(n):
        center = tuple(p0 + (-dx, (i - n / 2) * spacing))
        create_port_with_port_settings(
            name=f"W{i}",
            center=(center[0], center[1]),
            orientation=180,
        )

    for i in range(n):
        center = tuple(p0 + (dx, (i - n / 2) * spacing))
        create_port_with_port_settings(
            name=f"E{i}",
            center=(center[0], center[1]),
            orientation=0,
        )

    for i in range(n):
        center = tuple(p0 + ((i - n / 2) * spacing, dy))
        create_port_with_port_settings(
            name=f"N{i}",
            center=(center[0], center[1]),
            orientation=90,
        )

    for i in range(n):
        center = tuple(p0 + ((i - n / 2) * spacing, -dy))
        create_port_with_port_settings(
            name=f"S{i}",
            center=(center[0], center[1]),
            orientation=-90,
        )

    component.auto_rename_ports()
    return component


if __name__ == "__main__":
    import gdsfactory as gf

    # pdk = gf.pdk.get_active_pdk()
    # pdk.gds_write_settings.flatten_invalid_refs = False
    c = big_device()
    c = add_fiber_array(c)
    c.show()
    # lyrdb = c.connectivity_check()
    # filepath = gf.config.home / "errors.lyrdb"
    # lyrdb.save(filepath)
    # gf.show(c, lyrdb=filepath)
    # c = c.flatten_invalid_refs()
    # c.write_gds("./test.gds")
