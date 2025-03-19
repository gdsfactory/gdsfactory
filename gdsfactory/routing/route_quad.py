"""Route for electrical based on phidl.routing.route_quad."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

import gdsfactory as gf
from gdsfactory.typings import Port


def _get_rotated_basis(
    theta: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Returns basis vectors rotated CCW by theta (in degrees)."""
    theta = np.radians(theta)
    e1 = np.array([np.cos(theta), np.sin(theta)])
    e2 = np.array([-1 * np.sin(theta), np.cos(theta)])
    return e1, e2


def route_quad(
    component: gf.Component,
    port1: Port,
    port2: Port,
    width1: float | None = None,
    width2: float | None = None,
    layer: gf.typings.LayerSpec = "M1",
    manhattan_target_step: float | None = None,
) -> None:
    """Routes a basic quadrilateral polygon directly between two ports.

    Args:
        component: Component to add the route to.
        port1: Port to start route.
        port2 : Port objects to end route.
        width1: Width of quadrilateral at ports. If None, uses port widths.
        width2: Width of quadrilateral at ports. If None, uses port widths.
        layer: Layer to put the route on.
        manhattan_target_step: if not none, min step to manhattanize the polygon

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component()
        pad1 = c << gf.components.pad(size=(50, 50))
        pad2 = c << gf.components.pad(size=(10, 10))
        pad2.movex(100)
        pad2.movey(50)
        gf.routing.route_quad(
            c,
            pad1.ports["e2"],
            pad2.ports["e4"],
            width1=None,
            width2=None,
        )
        c.plot()

    """

    def get_port_edges(
        port: Port, width: float
    ) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
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
    sorted_vertices: npt.NDArray[np.floating[Any]] = np.array(
        [vert for _, vert in sorted(zip(angles, vertices), key=lambda x: x[0])],
        dtype=np.float64,
    )

    if manhattan_target_step:
        component.add_polygon(
            sorted_vertices,
            layer=layer,
        )
    else:
        component.add_polygon(points=sorted_vertices, layer=layer)


if __name__ == "__main__":
    from gdsfactory.components import pad

    c = gf.Component()
    pad1 = c << pad(size=(50, 50))
    pad2 = c << pad(size=(10, 10))
    pad2.movex(100)
    pad2.movey(50)
    route_quad(
        c,
        pad1.ports["e2"],
        pad2.ports["e4"],
        width1=None,
        width2=None,
        manhattan_target_step=0.1,
    )

    # c = gf.Component(name="route")
    # pad1 = c << gf.components.pad(size=(50, 50))
    # pad2 = c << gf.components.pad(size=(10, 10))
    # pad2.movex(100)
    # pad2.movey(50)
    # route_gnd = c << route_quad(
    #     pad1.ports["e2"],
    #     pad2.ports["e4"],
    #     width1=None,
    #     width2=None,
    #     manhattan_min_step=0.1,
    # )
    c.show()
    # test_manhattan_route_quad()
