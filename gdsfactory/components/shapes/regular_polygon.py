from __future__ import annotations

from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def regular_polygon(
    sides: int = 6,
    side_length: float = 10,
    layer: LayerSpec = "WG",
    port_type: str | None = "placement",
) -> Component:
    """Returns a regular N-sided polygon, with ports on each edge.

    Args:
        sides: number of sides for the polygon.
        side_length: of the edges.
        layer: Specific layer to put polygon geometry on.
        port_type: optical, electrical.
    """
    c = Component()
    angle_step = 2 * np.pi / sides
    radius = side_length / (2 * np.sin(np.pi / sides))

    # Rotate the polygon to make one facet flat
    rotation_angle = np.pi / 2 - angle_step / 2
    points = [
        (
            radius * np.cos(i * angle_step + rotation_angle),
            radius * np.sin(i * angle_step + rotation_angle),
        )
        for i in range(sides)
    ]
    c.add_polygon(points, layer=layer)
    a = side_length / (2 * np.tan(np.pi / sides))

    if port_type:
        for side_index in range(sides):
            angle = 270 + side_index * 360 / sides
            center = (a * np.cos(np.radians(angle)), a * np.sin(np.radians(angle)))
            c.add_port(
                name=f"o{side_index + 1}",
                center=center,
                width=side_length,
                layer=layer,
                port_type=port_type,
                orientation=angle,
            )

    c.auto_rename_ports()
    return c


hexagon = partial(regular_polygon, sides=6)
octagon = partial(regular_polygon, sides=8)


if __name__ == "__main__":
    # c = regular_polygon(sides=8, side_length=20)
    # c = rectangle(size=(3, 2), centered=True, layer=(2, 3))
    c = octagon(sides=8)
    c.show()
