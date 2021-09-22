from typing import Tuple

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.straight import straight


@cell
def array(
    component: gf.types.ComponentOrFactory = straight,
    spacing: Tuple[float, float] = (150.0, 150.0),
    columns: int = 6,
    rows: int = 1,
) -> Component:
    """Returns an array of components.

    Args:
        component: to replicate
        n: number of components
        pitch: float
        axis: x or y
        rotation: in degrees
        h_mirror: horizontal mirror
        v_mirror: vertical mirror
    """
    c = Component()
    component = component() if callable(component) else component
    c.add_array(component, columns=columns, rows=rows, spacing=spacing)

    for col in range(columns):
        for row in range(rows):
            for port in component.ports.values():
                name = f"{port.name}_{row+1}_{col+1}"
                c.add_port(name, port=port)
                c.ports[name].move((col * spacing[0], row * spacing[1]))
    return c


if __name__ == "__main__":

    c2 = array()
    c2.show(show_ports=True)
