from typing import List

from gdsfactory.components.bend_s import bend_s
from gdsfactory.port import Port
from gdsfactory.routing.sort_ports import sort_ports as sort_ports_function
from gdsfactory.types import Route


def get_bundle_sbend(
    ports1: Port, ports2: Port, sort_ports: bool = True, **kwargs
) -> List[Route]:
    """Returns a list of routes from ports1 to ports2.

    Args:
        ports1: start ports.
        ports2: end ports.
        sort_ports: sort ports.
        kwargs: cross_section settings.

    Returns:
        list of routes.

    """
    if sort_ports:
        ports1, ports2 = sort_ports_function(ports1, ports2)

    routes = []

    for p1, p2 in zip(ports1, ports2):
        ysize = p2.center[1] - p1.center[1]
        xsize = p2.center[0] - p1.center[0]
        bend = bend_s(size=(xsize, ysize), **kwargs)
        sbend = bend.ref()
        sbend.connect("o1", p1)
        routes.append(
            Route(
                references=[sbend],
                ports=tuple(sbend.get_ports_list()),
                length=bend.info["length"],
            )
        )

    return routes


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component("test_get_route_sbend")
    pitch = 2.0
    ys_left = [0, 10, 20]
    N = len(ys_left)
    y0 = -10
    ys_right = [(i - N / 2) * pitch + y0 for i in range(N)]

    layer = (1, 0)
    right_ports = [
        gf.Port(
            f"R_{i}", center=(0, ys_right[i]), width=0.5, orientation=180, layer=layer
        )
        for i in range(N)
    ]
    left_ports = [
        gf.Port(
            f"L_{i}", center=(-50, ys_left[i]), width=0.5, orientation=0, layer=layer
        )
        for i in range(N)
    ]
    left_ports.reverse()

    routes = gf.routing.get_bundle(right_ports, left_ports, with_sbend=False)
    for route in routes:
        c.add(route.references)
    c.show(show_ports=True)
