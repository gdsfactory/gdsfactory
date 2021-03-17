from pp.component import Component, ComponentReference
from pp.hash_points import hash_points
from pp.port import Port


def get_route_electrical_shortest_path(port1: Port, port2: Port) -> ComponentReference:
    """Returns polygon reference that connects two ports
    with a polygon that takes the shortest path

    Args:
        port1: src
        port2: dst
    """
    points = [port1.midpoint, port2.midpoint]
    name = f"zz_conn_{hash_points(points)}"
    c = Component(name=name)
    layer = port1.layer
    p1x0 = port1.endpoints[0][0]
    p1y0 = port1.endpoints[0][1]
    p1x1 = port1.endpoints[1][0]
    p1y1 = port1.endpoints[1][1]

    p2x0 = port2.endpoints[0][0]
    p2y0 = port2.endpoints[0][1]
    p2x1 = port2.endpoints[1][0]
    p2y1 = port2.endpoints[1][1]

    if port1.orientation in [90, 270]:
        c.add_polygon(
            ([(p1x1, p1y0), (p1x0, p1y1), (p2x1, p2y1), (p2x0, p2y0)]), layer=layer
        )
    else:
        c.add_polygon(
            ([(p1x0, p1y1), (p1x1, p1y0), (p2x1, p2y1), (p2x0, p2y0)]), layer=layer
        )
    return c.ref()


if __name__ == "__main__":
    import pp
    from pp.components.electrical.pad import pad_array

    c = Component("mzi_with_pads")
    mzi = pp.components.mzi2x2(with_elec_connections=True)
    pads = pad_array(n=3, port_list=["S"])
    p = c << pads
    c << mzi
    p.move((-150, 200))
    ports_pads = list(p.ports.values())
    ports_mzi = mzi.get_ports_list(port_type="dc")

    for p1, p2 in zip(ports_pads, ports_mzi):
        c.add(get_route_electrical_shortest_path(p1, p2))
    c.show()
