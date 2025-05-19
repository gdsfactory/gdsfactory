"""Route for electrical based on phidl.routing.route_quad."""

from __future__ import annotations

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

    def get_port_edges(port: Port, width: float):
        _, e1 = _get_rotated_basis(port.orientation)
        off = e1 * (width * 0.5)
        # np.add/np.subtract will use fast vectorized ops
        return port.center + off, port.center - off

    if width1 is None:
        width1 = port1.width
    if width2 is None:
        width2 = port2.width

    v1, v2 = get_port_edges(port1, width1)
    v3, v4 = get_port_edges(port2, width2)
    vertices = np.vstack([v1, v2, v3, v4])
    center = np.mean(vertices, axis=0)
    displacements = vertices - center
    # --- Optimize: Avoid Python sorted, use numpy for speed ---
    # Angles now use arctan2(y, x)
    angles = np.arctan2(displacements[:, 1], displacements[:, 0])
    idx = np.argsort(angles)
    sorted_vertices = vertices[idx]

    component.add_polygon(sorted_vertices, layer=layer)  # always add sorted quad


def _get_rotated_basis(angle: float):
    """Fast helper, used by route_quad"""
    radians = np.deg2rad(angle)
    c, s = np.cos(radians), np.sin(radians)
    return np.array([c, s]), np.array([-s, c])


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
