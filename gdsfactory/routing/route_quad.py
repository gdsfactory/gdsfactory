"""Route for electrical based on phidl.routing.route_quad."""
from __future__ import annotations

from typing import Optional

import numpy as np

import gdsfactory as gf
from gdsfactory.port import Port


def _get_rotated_basis(theta):
    """Returns basis vectors rotated CCW by theta (in degrees)."""
    theta = np.radians(theta)
    e1 = np.array([np.cos(theta), np.sin(theta)])
    e2 = np.array([-1 * np.sin(theta), np.cos(theta)])
    return e1, e2


@gf.cell
def route_quad(
    port1: Port,
    port2: Port,
    width1: Optional[float] = None,
    width2: Optional[float] = None,
    layer: gf.typings.LayerSpec = "M1",
) -> gf.Component:
    """Routes a basic quadrilateral polygon directly between two ports.

    Args:
        port1: Port to start route.
        port2 : Port objects to end route.
        width1: Width of quadrilateral at ports. If None, uses port widths.
        width2: Width of quadrilateral at ports. If None, uses port widths.
        layer: Layer to put the route on.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component()
        pad1 = c << gf.components.pad(size=(50, 50))
        pad2 = c << gf.components.pad(size=(10, 10))
        pad2.movex(100)
        pad2.movey(50)
        route_gnd = c << gf.routing.route_quad(
            pad1.ports["e2"],
            pad2.ports["e4"],
            width1=None,
            width2=None,
        )
        c.show()
        c.plot()

    """

    def get_port_edges(port, width):
        _, e1 = _get_rotated_basis(port.orientation)
        pt1 = port.center + e1 * width / 2
        pt2 = port.center - e1 * width / 2
        return pt1, pt2

    if width1 is None:
        width1 = port1.width
    if width2 is None:
        width2 = port2.width
    vertices = np.array(get_port_edges(port1, width1) + get_port_edges(port2, width2))
    center = np.mean(vertices, axis=0)
    displacements = vertices - center
    # sort vertices by angle from center of quadrilateral to make convex polygon
    angles = np.array([np.arctan2(disp[0], disp[1]) for disp in displacements])
    vertices = [vert for _, vert in sorted(zip(angles, vertices), key=lambda x: x[0])]

    component = gf.Component()
    component.add_polygon(points=vertices, layer=layer)
    component.add_port(
        name="e1",
        center=port1.center,
        orientation=port1.orientation + 180,
        width=width1,
        layer=layer,
    )
    component.add_port(
        name="e2",
        center=port2.center,
        orientation=port2.orientation + 180,
        width=width2,
        layer=layer,
    )
    return component


if __name__ == "__main__":
    # c = gf.Component()
    # mzi = c << gf.components.mzi_phase_shifter()
    # pads = c << gf.components.array(component=gf.components.pad, columns=3)
    # pads.ymin = mzi.ymax + 30

    # pads.movex(-pads.size_info.sc[0])
    # mzi.movex(-mzi.size_info.sc[0])

    # route_gnd = c << route_quad(
    #     mzi.ports["e1"],
    #     pads.ports["e4_1_1"],
    #     width1=None,
    #     width2=None,
    # )

    # c.show(show_ports=True)

    c = gf.Component("route")
    pad1 = c << gf.components.pad(size=(50, 50))
    pad2 = c << gf.components.pad(size=(10, 10))
    pad2.movex(100)
    pad2.movey(50)
    route_gnd = c << route_quad(
        pad1.ports["e2"],
        pad2.ports["e4"],
        width1=None,
        width2=None,
    )
    c.show()
