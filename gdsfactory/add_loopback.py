"""Add reference for a grating coupler array."""
from typing import List

import gdsfactory as gf
from gdsfactory.component import ComponentReference
from gdsfactory.port import Port
from gdsfactory.routing.manhattan import round_corners
from gdsfactory.types import ComponentFactory


def gen_loopback(
    start_port: Port,
    end_port: Port,
    gc: ComponentFactory,
    grating_separation: float = 127.0,
    gc_rotation: int = -90,
    gc_port_name: str = "W0",
    bend_radius_loopback: float = 10.0,
    bend_factory: ComponentFactory = gf.components.bend_euler,
    straight_factory: ComponentFactory = gf.components.straight,
    y_bot_align_route: None = None,
) -> List[ComponentReference]:
    """
    Add a loopback (grating coupler align reference) to a start port and and end port
    Input grating generated on the left of start_port
    Output grating generated on the right of end_port

    .. code::

        __________________________________________
        | separation  |            |              |
        |             |            |              |
       GC          start_port  end_port          GC

    """

    gc = gc() if callable(gc) else gc

    if hasattr(start_port, "y"):
        y0 = start_port.y
    else:
        y0 = start_port[1]

    if hasattr(start_port, "x"):
        x0 = start_port.x - grating_separation
    else:
        x0 = start_port[0] - grating_separation

    if hasattr(end_port, "x"):
        x1 = end_port.x + grating_separation
    else:
        x1 = end_port[0] + grating_separation

    gca1, gca2 = [
        gc.ref(position=(x, y0), rotation=gc_rotation, port_id=gc_port_name)
        for x in [x0, x1]
    ]

    gsi = gc.size_info
    p0 = gca1.ports[gc_port_name].position
    p1 = gca2.ports[gc_port_name].position
    bend90 = bend_factory(radius=bend_radius_loopback)

    if hasattr(bend90, "dx"):
        a = abs(bend90.dy)
    else:
        a = bend_radius_loopback + 0.5
    b = max(2 * a, grating_separation / 2)
    y_bot_align_route = (
        y_bot_align_route if y_bot_align_route is not None else -gsi.width - 5.0
    )

    points = [
        p0,
        p0 + (0, a),
        p0 + (b, a),
        p0 + (b, y_bot_align_route),
        p1 + (-b, y_bot_align_route),
        p1 + (-b, a),
        p1 + (0, a),
        p1,
    ]
    route = round_corners(
        points=points, bend_factory=bend90, straight_factory=straight_factory
    )
    elements = [gca1, gca2]
    elements.extend(route.references)
    return elements


@gf.cell
def straight_with_loopback() -> gf.Component:
    c = gf.Component("straight_with_loopback")
    wg = c << gf.components.straight()
    c.add(
        gen_loopback(
            wg.ports["W0"], wg.ports["E0"], gc=gf.components.grating_coupler_te
        )
    )
    return c


if __name__ == "__main__":
    component = straight_with_loopback()
    component.show()
