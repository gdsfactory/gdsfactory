from gdsfactory.component import Component, ComponentReference
from gdsfactory.hash_points import hash_points
from gdsfactory.port import Port


def get_route_electrical_shortest_path(port1: Port, port2: Port) -> ComponentReference:
    """Returns polygon reference that connects two ports
    with a polygon that takes the shortest path

    Args:
        port1: source port
        port2: destination port
    """
    points = [port1.midpoint, port2.midpoint]
    name = f"zz_conn_{hash_points(points)}"
    c = Component(name=name)
    layer = port1.layer
    p1x0 = port1.endpoints[0][0]
    p1y0 = port1.endpoints[0][1]
    p1x1 = port1.endpoints[1][0]
    p1y1 = port1.endpoints[1][1]

    x = [p1x0, p1x1]
    y = [p1y0, p1y1]
    p1y0 = min(y)
    p1y1 = max(y)
    p1x0 = min(x)
    p1x1 = max(x)

    p2x0 = port2.endpoints[0][0]
    p2y0 = port2.endpoints[0][1]
    p2x1 = port2.endpoints[1][0]
    p2y1 = port2.endpoints[1][1]

    x = [p2x0, p2x1]
    y = [p2y0, p2y1]
    p2y0 = min(y)
    p2y1 = max(y)
    p2x0 = min(x)
    p2x1 = max(x)

    if port1.orientation in [90, 270]:
        c.add_polygon(
            ([(p1x0, p1y0), (p1x1, p1y0), (p2x1, p2y1), (p2x0, p2y0)]), layer=layer
        )
    elif port1.orientation == 180:
        c.add_polygon(
            ([(p1x0, p1y1), (p1x1, p1y0), (p2x1, p2y0), (p2x0, p2y1)]), layer=layer
        )
    elif port1.orientation == 0:
        c.add_polygon(
            ([(p1x0, p1y0), (p1x1, p1y1), (p2x1, p2y0), (p2x0, p2y1)]), layer=layer
        )

    return c.ref()


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.components.pad import pad_array

    c = Component("mzi_with_pads")
    mzi = gf.components.mzi_phase_shifter()
    pads = pad_array(columns=3, orientation=270)
    p = c << pads
    c << mzi
    p.move((-150, 200))
    ports_pads = list(p.ports.values())
    ports_mzi = mzi.get_ports_list()

    for p1, p2 in zip(ports_pads, ports_mzi):
        c.add(get_route_electrical_shortest_path(p1, p2))
    c.show()
