from typing import Iterable

from pp.port import Port
from pp.routing.get_route_sbend import get_route_sbend

# from pp.routing.routing import route_basic
from pp.routing.routing import route_manhattan
from pp.routing.sort_ports import sort_ports
from pp.types import Routes


def get_routes(
    ports1: Iterable[Port],
    ports2: Iterable[Port],
    with_sort_ports: bool = True,
    **kwargs,
) -> Routes:
    """Returns Routes with all the manhattan routes.
    if it does not fit a manhattan route, adds an Sbend

    Args:
        ports1: list of src ports
        ports2: list of dst ports
        with_sort_ports: sort_ports
        **kwargs: for routing
    """
    references = []
    lengths = []

    if with_sort_ports:
        ports1, ports2 = sort_ports(ports1, ports2)

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

    import pp

    c = pp.Component("test_get_bundle_sort_ports")
    pitch = 5.0
    ys_left = [0, 10, 20]
    N = len(ys_left)
    ys_right = [25 + i * pitch for i in range(N)]

    p1 = [pp.Port(f"L_{i}", (0, ys_left[i]), 0.5, 0) for i in range(N)]
    p2 = [pp.Port(f"R_{i}", (20, ys_right[i]), 0.5, 180) for i in range(N)]

    p1.reverse()
    routes = get_routes(p1, p2)
    c.add(routes.references)
    c.show()

    # route_references = pp.routing.get_bundle(right_ports, left_ports, bend_radius=5)
    # c.add(route_references)

    # for p1, p2 in zip(right_ports, left_ports):
    #     path = pp.path.smooth(
    #         [
    #             p1.midpoint,
    #             p1.get_extended_midpoint(),
    #             [p1.get_extended_midpoint()[0], p2.get_extended_midpoint()[1]],
    #             p2.get_extended_midpoint(),
    #             p2.midpoint,
    #         ]
    #     )
    #     route = pp.path.extrude(path, cross_section=pp.cross_section.strip)
    #     c.add(route.ref())
