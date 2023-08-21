from warnings import warn

import numpy as np

from gdsfactory.component_reference import ComponentReference
from gdsfactory.port import Port
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import Route


def validate_connections(
    ports1: list[Port], ports2: list[Port], routes: list[Route]
) -> list[Route]:
    """
    Validates that a set of Routes indeed connects the port-pairs listed in ports1 and ports2. If the Routes form valid connections between ports1 and ports2, the original Routes will be returned. If not, a RouteWarning will be raised, and a set of error traces will be returned instead.

    Args:
        ports1: the list of starting ports
        ports2: the list of ending ports
        routes: the list of Route objects, purportedly between ports1 and ports2

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
        ports1: the list of starting ports
        ports2: the list of ending ports
        message: a message to include in the RouteWarning that is raised

    Returns:
        A list of Routes (the error traces).
    """
    import gdsfactory as gf
    from gdsfactory.routing.manhattan import RouteWarning

    warn(message, RouteWarning)
    error_routes = []
    for port1, port2 in zip(ports1, ports2):
        path = gf.path.Path([port1.center, port2.center])
        error_component = gf.path.extrude(path, layer="ERROR_PATH", width=1)
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
