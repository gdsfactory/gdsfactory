from __future__ import annotations

__all__ = ["array_hexagonal"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec


@gf.cell_with_module_name(tags=["containers"])
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
    return cf.array_hexagonal(
        component=component,
        columns=columns,
        rows=rows,
        pitch=pitch,
        centered=centered,
        add_ports=add_ports,
    )


if __name__ == "__main__":
    c = array_hexagonal()
    c.show()
