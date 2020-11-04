import pp


def test_link_electrical_ports():
    """FIXME use connect_bundle instead"""
    c = pp.Component("demo_connect_bundle_small_electrical")
    c1 = c << pp.c.pad()
    c2 = c << pp.c.pad()
    c2.move((200, 100))
    route = pp.routing.link_electrical_ports(
        [c1.ports["E"]], [c2.ports["W"]], route_filter=pp.routing.connect_elec_waypoints
    )
    c.add(route)
    print(route[0].parent.length)
    # assert np.isclose(route[0].parent.length, 200.0)

    route = pp.routing.link_electrical_ports(
        [c1.ports["S"]], [c2.ports["E"]], route_filter=pp.routing.connect_elec_waypoints
    )
    c.add(route)
    print(route[0].parent.length)
    # assert np.isclose(route[0].parent.length, 320.02)
    return c


if __name__ == "__main__":
    c = test_link_electrical_ports()
    pp.show(c)
