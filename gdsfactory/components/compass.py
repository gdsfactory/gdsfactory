from __future__ import annotations

from typing import Optional, Tuple

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.typings import Ints, LayerSpec


@cell
def compass(
    size: Tuple[float, float] = (4.0, 2.0),
    layer: LayerSpec = "WG",
    port_type: Optional[str] = "placement",
    port_inclusion: float = 0.0,
    port_orientations: Optional[Ints] = (180, 90, 0, -90),
) -> Component:
    """Rectangle with ports on each edge (north, south, east, and west).

    Args:
        size: rectangle size.
        layer: tuple (int, int).
        port_type: optical, electrical.
        port_inclusion: from edge.
        port_orientations: list of port_orientations to add. None add one port only.
    """
    c = gf.Component()
    dx, dy = size

    points = [
        [-dx / 2.0, -dy / 2.0],
        [-dx / 2.0, dy / 2],
        [dx / 2, dy / 2],
        [dx / 2, -dy / 2.0],
    ]

    c.add_polygon(points, layer=layer)

    if port_type:
        if 180 in port_orientations:
            c.add_port(
                name="e1",
                center=(-dx / 2 + port_inclusion, 0),
                width=dy,
                orientation=180,
                layer=layer,
                port_type=port_type,
            )
        if 90 in port_orientations:
            c.add_port(
                name="e2",
                center=(0, dy / 2 - port_inclusion),
                width=dx,
                orientation=90,
                layer=layer,
                port_type=port_type,
            )
        if 0 in port_orientations:
            c.add_port(
                name="e3",
                center=(dx / 2 - port_inclusion, 0),
                width=dy,
                orientation=0,
                layer=layer,
                port_type=port_type,
            )
        if -90 in port_orientations:
            c.add_port(
                name="e4",
                center=(0, -dy / 2 + port_inclusion),
                width=dx,
                orientation=-90,
                layer=layer,
                port_type=port_type,
            )
        if port_orientations is None:
            c.add_port(
                name="pad",
                center=(0, 0),
                width=dy,
                orientation=None,
                layer=layer,
                port_type=port_type,
            )

        c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = compass(size=(1, 2), layer="WG", port_type="optical", port_inclusion=0.5)
    c.show(show_ports=True)
