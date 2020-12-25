import pp


@pp.cell
def test_netlist_with_routes():
    """
    """
    c = pp.Component()
    w = c << pp.c.waveguide(length=3)
    b = c << pp.c.bend_circular()
    w.xmax = 0
    b.xmin = 10

    routes = pp.routing.connect_bundle(w.ports["E0"], b.ports["W0"])
    c.add(routes)
    n = c.get_netlist()
    connections = n["connections"]

    # print(routes[0].get_settings())
    # print(c.get_netlist().connections)
    # print(c.get_netlist().instances)
    # print(len(c.get_netlist().connections))

    assert len(c.get_dependencies()) == 3
    assert len(connections) == 2  # 2 components + 1 flat netlist
    return c


if __name__ == "__main__":
    c = test_netlist_with_routes()
    pp.show(c)
