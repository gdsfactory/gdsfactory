from __future__ import annotations

from gdsfactory.components.bend_s import bend_s
from gdsfactory.port import Port
from gdsfactory.typings import Route


def get_route_sbend(port1: Port, port2: Port, **kwargs) -> Route:
    """Returns an Sbend Route to connect two ports.

    Args:
        port1: start port.
        port2: end port.

    keyword Args:
        nb_points: number of points.
        with_cladding_box: square bounding box to avoid DRC errors.
        cross_section: function.
        kwargs: cross_section settings.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component("demo_route_sbend")
        mmi1 = c << gf.components.mmi1x2()
        mmi2 = c << gf.components.mmi1x2()
        mmi2.movex(50)
        mmi2.movey(5)
        route = gf.routing.get_route_sbend(mmi1.ports['o2'], mmi2.ports['o1'])
        c.add(route.references)
        c.show()
        c.plot()

    """
    ysize = port2.center[1] - port1.center[1]
    xsize = port2.center[0] - port1.center[0]
    size = (xsize, ysize)

    bend = bend_s(size=size, **kwargs)

    bend_ref = bend.ref()
    bend_ref.connect(list(bend_ref.ports.keys())[0], port1)

    orthogonality_error = abs(abs(port1.orientation - port2.orientation) - 180)
    if orthogonality_error > 0.1:
        raise ValueError(
            f"Ports need to have orthogonal orientation {orthogonality_error}\n"
            f"port1 = {port1.orientation} deg and port2 = {port2.orientation}"
        )

    return Route(
        references=[bend_ref],
        length=bend.info["length"],
        ports=(port1, port2),
    )


if __name__ == "__main__":
    # import gdsfactory as gf
    # from gdsfactory.routing.sort_ports import sort_ports

    # c = gf.Component("test_get_route_sbend")
    # pitch = 2.0
    # ys_left = [0, 10, 20]
    # N = len(ys_left)
    # ys_right = [(i - N / 2) * pitch for i in range(N)]

    # right_ports = [
    #     gf.Port(f"R_{i}", (0, ys_right[i]), width=0.5, orientation=180, layer=(1, 0))
    #     for i in range(N)
    # ]
    # left_ports = [
    #     gf.Port(f"L_{i}", (-50, ys_left[i]), width=0.5, orientation=0, layer=(1, 0))
    #     for i in range(N)
    # ]
    # left_ports.reverse()
    # right_ports, left_ports = sort_ports(right_ports, left_ports)

    # for p1, p2 in zip(right_ports, left_ports):
    #     route = get_route_sbend(p1, p2, layer=(2, 0))
    #     c.add(route.references)

    # c.show(show_ports=True)

    import gdsfactory as gf

    c = gf.Component("demo_route_sbend")
    mmi1 = c << gf.components.mmi1x2()
    mmi2 = c << gf.components.mmi1x2()
    mmi2.movex(50)
    mmi2.movey(5)
    route = gf.routing.get_route_sbend(mmi1.ports["o2"], mmi2.ports["o1"])
    c.add(route.references)
    c.show()
