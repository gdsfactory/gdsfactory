from warnings import warn

import numpy as np

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

    warn(message, RouteWarning)
    for port1, port2 in zip(ports1, ports2):
        path = gf.path.Path(np.array([port1.dcenter, port2.dcenter]))
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
    # draw lines between ports1 and ports2
    # if no lines intersect and EITHER
    # A. all are within 90 degrees of port angle (non-negative dot product) or
    # B. all are further than 90 degrees of port angle (negative dot product)
    # OR all lines intersect and all ports1 > 90, ports2 < 90, or vice versa
    # the topology is valid
    # (actually, the bundle can contain 2 groups-- one of each, and still maintain valid, as long as there are no crossings between them)
    import shapely.geometry as sg
    from shapely import intersection_all

    # this is not really quite angle, but a threshold to check if dot products are effectively above/below zero, excluding numerical errors
    angle_tolerance = 1e-10

    if len(ports1) < 2:
        # if there's only one route, the bundle topology is always valid
        return False
    if any(p.orientation is None for p in ports1 + ports2):
        # don't check if the ports do not have orientation
        return False

    lines = [sg.LineString([p1.dcenter, p2.dcenter]) for p1, p2 in zip(ports1, ports2)]

    # Positive if BOTH ports are EITHER facing towards OR away from the vector of the outgoing line between them
    # Zero if either is orthogonal
    # Negative if one is facing and the other not
    ports_facing = []
    for p1, p2 in zip(ports1, ports2):
        dy_line = p2.dcenter[1] - p1.dcenter[1]
        dx_line = p2.dcenter[0] - p1.dcenter[0]

        dy_p1 = np.sin(np.deg2rad(p1.orientation))
        dx_p1 = np.cos(np.deg2rad(p1.orientation))

        dy_p2 = np.sin(np.deg2rad(p2.orientation))
        dx_p2 = np.cos(np.deg2rad(p2.orientation))
        dot1 = np.vdot([dx_line, dy_line], [dx_p1, dy_p1])
        dot2 = np.vdot([-dx_line, -dy_line], [dx_p2, dy_p2])
        both_facing = dot1 * dot2
        # print(both_facing)
        ports_facing.append(both_facing)

    intersections = intersection_all(lines)
    # print(intersections)
    if intersections.is_empty and all(s < -angle_tolerance for s in ports_facing):
        return True
    elif not intersections.is_empty and all(s > angle_tolerance for s in ports_facing):
        return True

    # NOTE: there are more complicated cases we are ignoring for now and giving "the benefit of the doubt"
    # i.e. if ports2 is perpendicular to ports1 and located somewhere laterally in between ports1
    # or some cases where ports are not properly ordered
    # for now we call these cases potentially valid, but we could be stricter in the future
    return False
