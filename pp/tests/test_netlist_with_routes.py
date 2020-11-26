import pp


def test_netlist_with_routes():
    """ Needs FIX
    routes are not connected using connect,
    but we still need to add them to the netlist
    """
    c = pp.Component()
    w = c << pp.c.waveguide()
    b = c << pp.c.bend_circular()
    w.xmax = 0
    b.xmin = 10

    route = pp.routing.connect_bundle(w.ports["E0"], b.ports["W0"])
    c.add(route)
    print(c.get_netlist().connections)

    # print(c.get_netlist().pretty())
    # print((c.get_netlist().connections.pretty()))
    # print(len(c.get_netlist().connections))
    # print(len(c.get_dependencies()))
    # assert len(c.get_dependencies()) == 5
    # assert len(c.get_netlist().connections) == 18
    return c


if __name__ == "__main__":
    c = test_netlist_with_routes()
    pp.show(c)
