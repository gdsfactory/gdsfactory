from gdsfactory.components.bend_s import bend_s
from gdsfactory.port import Port
from gdsfactory.routing.sort_ports import sort_ports as sort_ports_function
from gdsfactory.types import Routes


def get_bundle_sbend(
    ports1: Port, ports2: Port, sort_ports: bool = True, **kwargs
) -> Routes:
    """Returns a Dict with the routes from ports1 to ports2

    Args:
        ports1: start ports
        ports2: end ports
        **kwargs
            nb_points: number of points
            cross_section_factory
            **waveguide_settings

    Returns:
        references: List of route references
        lengths: list of floats
        bend_radius: list of min bend_radius

    """
    if sort_ports:
        ports1, ports2 = sort_ports_function(ports1, ports2)

    references = []
    lengths = []
    bend_radius = []

    for p1, p2 in zip(ports1, ports2):
        height = p2.midpoint[1] - p1.midpoint[1]
        length = p2.midpoint[0] - p1.midpoint[0]
        bend = bend_s(height=height, length=length, **kwargs)
        sbend = bend.ref()
        sbend.connect("W0", p1)
        references.append(sbend)
        lengths.append(bend.info["length"])
        bend_radius.append(bend.info["min_bend_radius"])

    return Routes(references=references, lengths=lengths, bend_radius=bend_radius)


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

    routes = get_bundle_sbend(right_ports, left_ports)
    c.add(routes.references)
    c.show()
