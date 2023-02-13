from __future__ import annotations

from typing import Tuple

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.straight import straight


@cell
def array(
    component: gf.typings.ComponentSpec = straight,
    spacing: Tuple[float, float] = (150.0, 150.0),
    columns: int = 6,
    rows: int = 1,
    add_ports: bool = True,
) -> Component:
    """Returns an array of components.

    Args:
        component: to replicate.
        spacing: x, y spacing.
        columns: in x.
        rows: in y.
        add_ports: add ports from component into the array.

    Raises:
        ValueError: If columns > 1 and spacing[0] = 0.
        ValueError: If rows > 1 and spacing[1] = 0.

    .. code::

        2 rows x 4 columns
         ___        ___       ___          ___
        |   |      |   |     |   |        |   |
        |___|      |___|     |___|        |___|

         ___        ___       ___          ___
        |   |      |   |     |   |        |   |
        |___|      |___|     |___|        |___|
    """
    if rows > 1 and spacing[1] == 0:
        raise ValueError(f"rows = {rows} > 1 require spacing[1] > 0")

    if columns > 1 and spacing[0] == 0:
        raise ValueError(f"columns = {columns} > 1 require spacing[0] > 0")

    c = Component()
    component = gf.get_component(component)
    c.add_array(component, columns=columns, rows=rows, spacing=spacing)

    if add_ports and component.ports:
        for col in range(columns):
            for row in range(rows):
                for port in component.ports.values():
                    name = f"{port.name}_{row+1}_{col+1}"
                    c.add_port(name, port=port)
                    c.ports[name].move((col * spacing[0], row * spacing[1]))
    return c


if __name__ == "__main__":
    from gdsfactory.components.pad import pad

    # c2 = array(rows=2, columns=2, spacing=(100, 100))
    c2 = array(pad, rows=2, spacing=(200, 200), columns=1)

    # c3 = c2.copy()

    # nports = len(c2.get_ports_list(orientation=0))
    # assert nports == 2, nports
    # c2.show(show_ports=True)
    # c2.show(show_subports=True)
    c2.show(show_ports=True)
