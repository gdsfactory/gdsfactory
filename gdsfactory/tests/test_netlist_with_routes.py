import gdsfactory
from gdsfactory.component import Component


@gdsfactory.cell
def test_netlist_with_routes() -> Component:
    """ """
    c = gdsfactory.Component()
    w = c << gdsfactory.components.straight(length=3)
    b = c << gdsfactory.components.bend_circular()
    w.xmax = 0
    b.xmin = 10

    routes = gdsfactory.routing.get_bundle(w.ports["E0"], b.ports["W0"])
    for route in routes:
        c.add(route.references)
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
    c.show()
