from __future__ import annotations

__all__ = ["array_polar"]

import numpy as np
from kfactory.conf import CheckInstances

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec


@gf.cell(with_module_name=True, check_instances=CheckInstances.IGNORE)
def array_polar(
    component: ComponentSpec = "C",
    n_items: int = 6,
    radius: float = 50.0,
    start_angle: float = 0.0,
    end_angle: float = 360.0,
    rotate_items: bool = True,
    add_ports: bool = True,
) -> Component:
    """Returns a polar/circular array of components.

    Places component refs at equal angular intervals around a circle.

    Args:
        component: component to replicate.
        n_items: number of items in the array.
        radius: radius of the circle.
        start_angle: starting angle in degrees.
        end_angle: ending angle in degrees.
        rotate_items: if True, rotate each item to point radially outward.
        add_ports: add ports from each element.
    """
    c = Component()
    comp = gf.get_component(component)

    if abs(end_angle - start_angle) >= 360.0:
        angles = np.linspace(start_angle, end_angle, n_items, endpoint=False)
    else:
        angles = np.linspace(start_angle, end_angle, n_items, endpoint=True)

    for i, angle_deg in enumerate(angles):
        angle_rad = np.radians(angle_deg)
        x = radius * np.cos(angle_rad)
        y = radius * np.sin(angle_rad)

        ref = c.add_ref(comp)
        if rotate_items:
            ref.rotate(angle_deg)
        ref.move((x, y))

        if add_ports and comp.ports:
            for port in comp.ports:
                name = f"{port.name}_{i + 1}"
                c.add_port(name, port=port.copy(ref.trans))

    return c


if __name__ == "__main__":
    c = array_polar()
    c.show()
