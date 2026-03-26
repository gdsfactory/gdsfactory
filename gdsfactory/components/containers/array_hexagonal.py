from __future__ import annotations

__all__ = ["array_hexagonal"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec


@gf.cell_with_module_name
def array_hexagonal(
    component: ComponentSpec = "circle",
    columns: int = 10,
    rows: int = 10,
    pitch: float = 25.0,
    centered: bool = True,
    add_ports: bool = True,
) -> Component:
    """Returns a hexagonal close-packed array of components.

    Even rows are placed normally, odd rows are offset by pitch/2.
    Row spacing is pitch * sqrt(3)/2.

    Args:
        component: component to replicate.
        columns: number of columns.
        rows: number of rows.
        pitch: spacing between adjacent elements.
        centered: center the array around the origin.
        add_ports: add ports from each element.
    """
    c = Component()
    comp = gf.get_component(component)
    row_spacing = pitch * np.sqrt(3) / 2

    for row in range(rows):
        x_offset = pitch / 2 if row % 2 else 0.0
        for col in range(columns):
            ref = c.add_ref(comp)
            x = col * pitch + x_offset
            y = row * row_spacing
            ref.move((x, y))

            if add_ports and comp.ports:
                for port in comp.ports:
                    name = f"{port.name}_{row + 1}_{col + 1}"
                    c.add_port(name, port=port.copy(ref.trans))

    if centered:
        c.center = (0, 0)

    return c


if __name__ == "__main__":
    c = array_hexagonal()
    c.show()
