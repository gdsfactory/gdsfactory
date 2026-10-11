from __future__ import annotations

__all__ = ["array_polar"]

from kfactory.conf import CheckInstances

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec


@gf.cell(
    with_module_name=True,
    check_instances=CheckInstances.IGNORE,
    tags=["containers"],
)
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
    return cf.array_polar(
        component=component,
        n_items=n_items,
        radius=radius,
        start_angle=start_angle,
        end_angle=end_angle,
        rotate_items=rotate_items,
        add_ports=add_ports,
    )


if __name__ == "__main__":
    c = array_polar()
    c.show()
