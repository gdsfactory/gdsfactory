from gdsfactory.components.bend_s import bend_s
from gdsfactory.port import Port
from gdsfactory.routing.sort_ports import sort_ports
from gdsfactory.types import Route


def get_route_sbend(port1: Port, port2: Port, **kwargs) -> Route:
    """Returns an Sbend Route to connect two ports.

    Args:
        port1: start port
        port2: end port
        **kwargs
            nb_points: number of points
            cross_section_factory
            **waveguide_settings
    """
    height = port2.midpoint[1] - port1.midpoint[1]
    length = port2.midpoint[0] - port1.midpoint[0]
    bend = bend_s(height=height, length=length, **kwargs)
    bend_ref = bend.ref()
    bend_ref.connect("W0", port1)
    return Route(
        references=[bend_ref], length=bend.info["length"], ports=(port1, port2)
    )


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component("test_get_route_sbend")
    pitch = 2.0
    ys_left = [0, 10, 20]
    N = len(ys_left)
    ys_right = [(i - N / 2) * pitch for i in range(N)]

    right_ports = [gf.Port(f"R_{i}", (0, ys_right[i]), 0.5, 180) for i in range(N)]
    left_ports = [gf.Port(f"L_{i}", (-50, ys_left[i]), 0.5, 0) for i in range(N)]
    left_ports.reverse()
    right_ports, left_ports = sort_ports(right_ports, left_ports)

    for p1, p2 in zip(right_ports, left_ports):
        route = get_route_sbend(p1, p2, waveguide="nitride")
        c.add(route.references)

    c.show()
