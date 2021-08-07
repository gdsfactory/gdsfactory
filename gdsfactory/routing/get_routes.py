import warnings
from typing import Iterable

from gdsfactory.port import Port
from gdsfactory.routing.get_route_sbend import get_route_sbend

# from gdsfactory.routing.routing import route_basic
from gdsfactory.routing.routing import route_manhattan
from gdsfactory.routing.sort_ports import sort_ports as sort_ports_function
from gdsfactory.types import Routes


def get_routes(
    ports1: Iterable[Port],
    ports2: Iterable[Port],
    sort_ports: bool = True,
    **kwargs,
) -> Routes:
    """Returns Routes with all the manhattan routes.
    if a manhattan route does not fit it adds an Sbend

    Notice that it does not route bundles of ports. Use get_bundle instead.
    Temporary solution until round_corners supports Sbend routing

    Args:
        ports1: list of src ports
        ports2: list of dst ports
        with_sort_ports: sort_ports
        **kwargs: for routing
    """
    warnings.warn(
        "get_routes is a temporary solution until get_bundle supports Sbend routing "
        "get_routes does not route bundles of ports, use get_bundle instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    references = []
    lengths = []

    if sort_ports:
        ports1, ports2 = sort_ports_function(ports1, ports2)

    for port1, port2 in zip(ports1, ports2):
        try:
            route = route_manhattan(port1, port2, **kwargs)
            references.extend(route.references)
            lengths.append(route.length)
        except ValueError:
            # route = route_basic(port1=port1, port2=port2, **kwargs)
            # route_ref = path.ref()
            route = get_route_sbend(port1, port2, **kwargs)
            references.extend(route.references)
            lengths.append(route.length)

    return Routes(references=references, lengths=lengths)


if __name__ == "__main__":

    import gdsfactory as gf

    c = gf.Component("test_get_bundle_sort_ports")
    pitch = 5.0
    ys_left = [0, 10, 20]
    N = len(ys_left)
    ys_right = [25 + i * pitch for i in range(N)]

    p1 = [gf.Port(f"L_{i}", (0, ys_left[i]), 0.5, 0) for i in range(N)]
    p2 = [gf.Port(f"R_{i}", (20, ys_right[i]), 0.5, 180) for i in range(N)]

    p1.reverse()
    routes = get_routes(p1, p2)
    c.add(routes.references)
    c.show()

    # route_references = gf.routing.get_bundle(right_ports, left_ports, bend_radius=5)
    # c.add(route_references)

    # for p1, p2 in zip(right_ports, left_ports):
    #     path = gf.path.smooth(
    #         [
    #             p1.midpoint,
    #             p1.get_extended_midpoint(),
    #             [p1.get_extended_midpoint()[0], p2.get_extended_midpoint()[1]],
    #             p2.get_extended_midpoint(),
    #             p2.midpoint,
    #         ]
    #     )
    #     route = gf.path.extrude(path, cross_section=gf.cross_section.strip)
    #     c.add(route.ref())
