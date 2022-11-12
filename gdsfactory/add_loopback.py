"""Add reference for a grating coupler array."""
from typing import List, Optional

import gdsfactory as gf
from gdsfactory.component import ComponentReference
from gdsfactory.port import Port
from gdsfactory.routing.manhattan import round_corners
from gdsfactory.types import ComponentSpec


def add_loopback(
    port1: Port,
    port2: Port,
    grating: ComponentSpec,
    grating_separation: float = 127.0,
    grating_rotation: int = -90,
    grating_port_name: str = "o1",
    bend: ComponentSpec = gf.components.bend_euler,
    south_waveguide_spacing: Optional[float] = None,
    inside: bool = True,
    **kwargs
) -> List[ComponentReference]:
    """Return loopback (grating coupler align reference) references.

    Input grating generated on the left of port1
    Output grating generated on the right of port2

    Args:
        port1: start port.
        port2: end port.
        grating: fiber coupler.
        grating_separation: grating pitch.
        grating_rotation: in degrees.
        grating_port_name: fiber port name for grating coupler.
        bend: bend spec.
        south_waveguide_spacing: spacing from loopback to grating_coupler.ymin
        inside: add loopback inside.
        kwargs: cross_section settings.

    .. code::

    inside = True
         ______                     ______
        |<-separation  |     |      |     |
        |      |       |     |      |     |
       GC      |    port1  port2    |    GC___
               |                    |       |
               |                    |       | south_waveguide_spacing
               |____________________|      _|_

    inside = False
                ______                    _______
               |     |                   |      |
               |     |       |     |     |      |
               |     GC   port1  port2   GC     |      ___
               |                                |       |
               |                                |       | south_waveguide_spacing
               |________________________________|      _|_
    """
    gc = gf.get_component(grating)

    y0 = port1.y if hasattr(port1, "y") else port1[1]
    if hasattr(port1, "x"):
        x0 = port1.x - grating_separation
    else:
        x0 = port1[0] - grating_separation

    if hasattr(port2, "x"):
        x1 = port2.x + grating_separation
    else:
        x1 = port2[0] + grating_separation

    gca1, gca2 = (
        gc.ref(position=(x, y0), rotation=grating_rotation, port_id=grating_port_name)
        for x in [x0, x1]
    )

    gsi = gc.size_info
    p0 = gca1.ports[grating_port_name].center
    p1 = gca2.ports[grating_port_name].center
    bend90 = bend(**kwargs)

    a = abs(bend90.info["dy"]) if hasattr(bend90, "dx") else bend90.xsize + 0.5
    b = max(2 * a, grating_separation / 2)
    b = b if inside else -b

    south_waveguide_spacing = (
        south_waveguide_spacing
        if south_waveguide_spacing is not None
        else -gsi.width - 5.0
    )

    points = [
        p0,
        p0 + (0, a),
        p0 + (b, a),
        p0 + (b, south_waveguide_spacing),
        p1 + (-b, south_waveguide_spacing),
        p1 + (-b, a),
        p1 + (0, a),
        p1,
    ]
    route = round_corners(points=points, bend=bend90, **kwargs)
    elements = [gca1, gca2]
    elements.extend(route.references)
    return elements


if __name__ == "__main__":
    c = gf.Component("straight_with_loopback")
    wg = c << gf.components.straight()
    c.add(
        add_loopback(
            wg.ports["o1"],
            wg.ports["o2"],
            grating=gf.components.grating_coupler_te,
            inside=False,
        )
    )
    c.show(show_ports=True)
