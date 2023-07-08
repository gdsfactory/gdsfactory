"""Lets create a crossing component with two references to other components (crossing_arm).

- add references to a component (one of the arm references rotated)
- add ports from the child references into the parent cell
- use Component.auto_rename_ports() to rename ports according to their location

"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import Layer


@gf.cell
def crossing_arm(
    wg_width: float = 0.5,
    r1: float = 3.0,
    r2: float = 1.1,
    taper_width: float = 1.2,
    taper_length: float = 3.4,
    layer_wg: Layer = (1, 0),
    layer_slab: Layer = (2, 0),
) -> Component:
    """Crossing arm."""
    c = gf.Component()
    c << gf.components.ellipse(radii=(r1, r2), layer=layer_slab)

    xmax = taper_length + taper_width / 2
    h = wg_width / 2
    taper_points = [
        (-xmax, h),
        (-taper_width / 2, taper_width / 2),
        (taper_width / 2, taper_width / 2),
        (xmax, h),
        (xmax, -h),
        (taper_width / 2, -taper_width / 2),
        (-taper_width / 2, -taper_width / 2),
        (-xmax, -h),
    ]

    c.add_polygon(taper_points, layer=layer_wg)

    c.add_port(
        name="o1", center=(-xmax, 0), orientation=180, width=wg_width, layer=layer_wg
    )

    c.add_port(
        name="o2", center=(xmax, 0), orientation=0, width=wg_width, layer=layer_wg
    )
    return c


@gf.cell  # This decorator will generate a good name for the component
def crossing() -> Component:
    c = gf.Component()
    arm = crossing_arm()

    # Create two arm references. One has a 90Deg rotation
    arm_h = arm.ref(position=(0, 0))
    arm_v = arm.ref(position=(0, 0), rotation=90)

    # Add each arm to the component
    # Also add the ports
    port_id = 0
    for a in [arm_h, arm_v]:
        c.add(a)
        for p in a.ports.values():
            # Here we don't care too much about the name we give to the ports
            # since they will be renamed. We just want the names to be unique
            c.add_port(name=f"{port_id}", port=p)
            port_id += 1

    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = crossing()
    c.show(show_ports=True)
