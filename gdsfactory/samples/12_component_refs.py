"""Lets create a crossing component with two references to other components (crossing_arm).

- add references to a component (one of the arm references rotated)
- add ports from the child references into the parent cell
- use Component.auto_rename_ports() to rename ports according to their location

"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentFactory, Layer


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
    """Crossing arm.

    Args:
        wg_width: waveguide width.
        r1: radius of the ellipse.
        r2: radius of the ellipse.
        taper_width: width of the taper.
        taper_length: length of the taper.
        layer_wg: waveguide layer.
        layer_slab: slab layer.
    """
    c = gf.Component()
    _ = c << gf.components.ellipse(radii=(r1, r2), layer=layer_slab)

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


@gf.cell
def crossing(
    arm: ComponentFactory = crossing_arm,
    cross_section: str = "strip",
) -> Component:
    """Waveguide crossing.

    Args:
        arm: arm spec.
        cross_section: spec.
    """
    x = gf.get_cross_section(cross_section)
    c = Component()
    arm_c = gf.get_component(arm)
    port_id = 0
    for rotation in [0, 90]:
        ref = c << arm_c
        ref.rotate(rotation)
        for p in ref.ports:
            c.add_port(name=str(port_id), port=p)
            port_id += 1

    c.auto_rename_ports()
    x.add_bbox(c)
    c.flatten()
    return c


if __name__ == "__main__":
    c = crossing()
    c.show()
