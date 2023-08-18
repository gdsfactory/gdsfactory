from warnings import warn

import numpy as np

from gdsfactory.component_reference import ComponentReference
from gdsfactory.port import Port
from gdsfactory.typings import Route


def validate_connections(
    ports1: list[Port], ports2: list[Port], routes: list[Route]
) -> list[Route]:
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
    return (tuple(port1.center), tuple(port2.center))
