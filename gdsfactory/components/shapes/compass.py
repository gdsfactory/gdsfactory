from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.config import valid_port_orientations
from gdsfactory.snap import snap_to_grid2x
from gdsfactory.typings import Ints, LayerSpec, Size


@gf.cell_with_module_name
def compass(
    size: Size = (4.0, 2.0),
    layer: LayerSpec = "WG",
    port_type: str | None = "electrical",
    port_inclusion: float = 0.0,
    port_orientations: Ints | None = (180, 90, 0, -90),
    auto_rename_ports: bool = True,
) -> Component:
    """Rectangle with ports on each edge (north, south, east, and west).

    Args:
        size: rectangle size.
        layer: tuple (int, int).
        port_type: optical, electrical.
        port_inclusion: from edge.
        port_orientations: list of port_orientations to add. None does not add ports.
        auto_rename_ports: auto rename ports.
    """
    c = gf.Component()
    dx, dy = snap_to_grid2x(size)
    ports = port_orientations if port_orientations else ()

    if dx <= 0 or dy <= 0:
        raise ValueError(f"dx={dx} and dy={dy} must be > 0")

    half_dx, half_dy = dx * 0.5, dy * 0.5
    points = [
        (-half_dx, -half_dy),
        (-half_dx, half_dy),
        (half_dx, half_dy),
        (half_dx, -half_dy),
    ]
    c.add_polygon(points, layer=layer)

    if port_type and ports:
        port_set = set(ports)
        not_valid = [po for po in port_set if po not in valid_port_orientations]
        if not_valid:
            raise ValueError(
                f"port_orientations contain invalid entries: {not_valid}, must be in {valid_port_orientations}"
            )
        # Add at most one port per unique orientation:
        if 180 in port_set:
            c.add_port(
                name="e1",
                center=(-half_dx + port_inclusion, 0),
                width=dy,
                orientation=180,
                layer=layer,
                port_type=port_type,
            )
        if 90 in port_set:
            c.add_port(
                name="e2",
                center=(0, half_dy - port_inclusion),
                width=dx,
                orientation=90,
                layer=layer,
                port_type=port_type,
            )
        if 0 in port_set:
            c.add_port(
                name="e3",
                center=(half_dx - port_inclusion, 0),
                width=dy,
                orientation=0,
                layer=layer,
                port_type=port_type,
            )
        if -90 in port_set or 270 in port_set:
            c.add_port(
                name="e4",
                center=(0, -half_dy + port_inclusion),
                width=dx,
                orientation=-90,
                layer=layer,
                port_type=port_type,
            )
        if auto_rename_ports:
            c.auto_rename_ports()
    return c


if __name__ == "__main__":
    # c = compass(size=(10, 4), port_type="electrical")
    c = compass(port_orientations=[270])
    c.pprint_ports()
    c.show()
