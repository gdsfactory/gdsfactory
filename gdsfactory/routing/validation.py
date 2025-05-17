import math
from warnings import warn

import numpy as np
import shapely.geometry as sg
from shapely import intersection_all

import gdsfactory as gf
from gdsfactory.config import CONF
from gdsfactory.port import Port
from gdsfactory.routing.utils import RouteWarning


def make_error_traces(
    component: gf.Component, ports1: list[Port], ports2: list[Port], message: str
) -> None:
    """Creates a set of error traces showing the intended connectivity between ports1 and ports2.

    The specified message will be included in the RouteWarning that is raised.

    Args:
        component: the Component to add the error traces to.
        ports1: the list of starting ports.
        ports2: the list of ending ports.
        message: a message to include in the RouteWarning that is raised.

    Returns:
        A list of Routes (the error traces).
    """
    import gdsfactory as gf

    warn(message, RouteWarning, stacklevel=3)
    for port1, port2 in zip(ports1, ports2):
        path = gf.path.Path(np.array([port1.center, port2.center]))
        error_component = gf.path.extrude(path, layer=CONF.layer_error_path, width=1)
        _ = component << error_component


def is_invalid_bundle_topology(ports1: list[Port], ports2: list[Port]) -> bool:
    """Returns True if the bundle is topologically unroutable without introducing crossings.

    Args:
        ports1: the starting ports of the bundle.
        ports2: the ending ports of the bundle.

    Returns:
        True if the bundle is unroutable. False otherwise.
    """
    angle_tolerance = 1e-10

    # Early exit: only one route or missing orientation, always valid
    if len(ports1) < 2:
        return False

    # Use generator to short-circuit the 'any' check on orientations
    if any(not getattr(p, "orientation", None) for p in ports1 + ports2):
        return False

    n = len(ports1)
    # Prebuild all necessary data in single iteration and minimize attribute lookups
    ports_facing = []
    lines = []

    for i in range(n):
        p1 = ports1[i]
        p2 = ports2[i]
        c1 = p1.center
        c2 = p2.center
        dx_line = c2[0] - c1[0]
        dy_line = c2[1] - c1[1]

        theta1 = math.radians(p1.orientation)
        theta2 = math.radians(p2.orientation)

        dx_p1 = math.cos(theta1)
        dy_p1 = math.sin(theta1)
        dx_p2 = math.cos(theta2)
        dy_p2 = math.sin(theta2)

        # Direct computation (vector dot products)
        dot1 = dx_line * dx_p1 + dy_line * dy_p1
        dot2 = -dx_line * dx_p2 + -dy_line * dy_p2
        # ('both_facing' is positive if both ports face the line or away; negative if not)
        both_facing = dot1 * dot2
        ports_facing.append(both_facing)

        lines.append(sg.LineString([c1, c2]))

    intersections = intersection_all(lines)

    if intersections.is_empty and all(s < -angle_tolerance for s in ports_facing):
        return True
    elif not intersections.is_empty and all(s > angle_tolerance for s in ports_facing):
        return True

    # Other cases are treated as potentially valid (return False)
    return False
