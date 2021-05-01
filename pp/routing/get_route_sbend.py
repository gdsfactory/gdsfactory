from pp.component import Component
from pp.components.bend_s import bend_s
from pp.port import Port
from pp.routing.sort_ports import sort_ports


def get_route_sbend(p1: Port, p2: Port, **kwargs) -> Component:
    """Returns an Sbend to connect two ports.

    Args:
        p1: start port
        p2: end port
        **kwargs
            nb_points: number of points
            cross_section_factory
            **cross_section_settings
    """
    height = p2.midpoint[1] - p1.midpoint[1]
    length = p2.midpoint[0] - p1.midpoint[0]
    return bend_s(height=height, length=length, **kwargs)


if __name__ == "__main__":
    import pp

    c = pp.Component("test_get_route_sbend")
    pitch = 2.0
    ys_left = [0, 10, 20]
    N = len(ys_left)
    ys_right = [(i - N / 2) * pitch for i in range(N)]

    right_ports = [pp.Port(f"R_{i}", (0, ys_right[i]), 0.5, 180) for i in range(N)]
    left_ports = [pp.Port(f"L_{i}", (-50, ys_left[i]), 0.5, 0) for i in range(N)]
    left_ports.reverse()
    right_ports, left_ports = sort_ports(right_ports, left_ports)

    for p1, p2 in zip(right_ports, left_ports):
        sbend = c << get_route_sbend(p1, p2)
        sbend.connect("W0", p1)

    c.show()
