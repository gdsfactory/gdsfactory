from warnings import warn

import numpy as np

from gdsfactory.component_reference import ComponentReference
from gdsfactory.config import CONF
from gdsfactory.port import Port
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import Route


def validate_connections(
    ports1: list[Port], ports2: list[Port], routes: list[Route]
) -> list[Route]:
    """
    Validates that a set of Routes indeed connects the port-pairs listed in ports1 and ports2. If the Routes form valid connections between ports1 and ports2, the original Routes will be returned. If not, a RouteWarning will be raised, and a set of error traces will be returned instead.

    Args:
        ports1: the list of starting ports.
        ports2: the list of ending ports.
        routes: the list of Route objects, purportedly between ports1 and ports2.

    Returns:
        A list of Routes. If the input routes are valid, they will be returned as-is. Otherwise, a list of error traces will be returned and a RouteWarning will be raised.
    """
    connections_expected = {_connection_tuple(p1, p2) for p1, p2 in zip(ports1, ports2)}

    for route in routes:
        connection = _connection_tuple(*route.ports)
        if connection not in connections_expected:
            return make_error_traces(
                ports1=ports1,
                ports2=ports2,
                message="Unable to route bundle! Please check the ordering of your ports to ensure it is a possible topology.",
            )
    return routes


def make_error_traces(
    ports1: list[Port], ports2: list[Port], message: str
) -> list[Route]:
    """
    Creates a set of error traces showing the intended connectivity between ports1 and ports2. The specified message will be included in the RouteWarning that is raised.

    Args:
        ports1: the list of starting ports.
        ports2: the list of ending ports.
        message: a message to include in the RouteWarning that is raised.

    Returns:
        A list of Routes (the error traces).
    """
    import gdsfactory as gf
    from gdsfactory.routing.manhattan import RouteWarning

    warn(message, RouteWarning)
    error_routes = []
    for port1, port2 in zip(ports1, ports2):
        path = gf.path.Path([port1.center, port2.center])
        error_component = gf.path.extrude(path, layer=CONF.layer_error_path, width=1)
        error_ref = ComponentReference(error_component)
        error_route = Route(
            references=[error_ref], ports=list(error_ref.ports.values()), length=np.nan
        )
        error_routes.append(error_route)
    return error_routes


def _connection_tuple(port1: Port, port2: Port) -> tuple:
    c1 = snap_to_grid(port1.center)
    c2 = snap_to_grid(port2.center)
    return (tuple(c1), tuple(c2))


def is_invalid_bundle_topology(ports1: list[Port], ports2: list[Port]) -> bool:
    """Returns True if the bundle is topologically unroutable without introducing crossings.

    Args:
        ports1: the starting ports of the bundle
        ports2: the ending ports of the bundle

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
    ANGLE_TOLERANCE = 1e-10

    if len(ports1) < 2:
        # if there's only one route, the bundle topology is always valid
        return False
    if any(p.orientation is None for p in ports1 + ports2):
        # don't check if the ports do not have orientation
        return False

    lines = [sg.LineString([p1.center, p2.center]) for p1, p2 in zip(ports1, ports2)]

    # Positive if BOTH ports are EITHER facing towards OR away from the vector of the outgoing line between them
    # Zero if either is orthogonal
    # Negative if one is facing and the other not
    ports_facing = []
    for p1, p2 in zip(ports1, ports2):
        dy_line = p2.center[1] - p1.center[1]
        dx_line = p2.center[0] - p1.center[0]

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
    if intersections.is_empty and all(s < -ANGLE_TOLERANCE for s in ports_facing):
        return True
    elif not intersections.is_empty and all(s > ANGLE_TOLERANCE for s in ports_facing):
        return True

    # NOTE: there are more complicated cases we are ignoring for now and giving "the benefit of the doubt"
    # i.e. if ports2 is perpendicular to ports1 and located somewhere laterally in between ports1
    # or some cases where ports are not properly ordered
    # for now we call these cases potentially valid, but we could be stricter in the future
    return False
