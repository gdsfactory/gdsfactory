import pp
from pp.component import Component


@pp.cell
def test_get_bundle_optical2() -> Component:
    """Test route length"""
    c = pp.Component()

    w = c << pp.c.waveguide_array(n_waveguides=4, spacing=200)
    d = c << pp.c.nxn(west=4, east=1)
    d.y = w.y
    d.xmin = w.xmax + 200

    ports1 = [
        w.ports["E1"],
        w.ports["E0"],
    ]
    ports2 = [
        d.ports["W1"],
        d.ports["W0"],
    ]

    routes = pp.routing.link_optical_ports(
        ports1, ports2, sort_ports=True, bend_radius=10
    )

    lengths = [486.028, 287.361]

    for route, length in zip(routes, lengths):
        c.add(route["references"])
        print(route["length"])
        # assert np.isclose(route["length"], length, atol=0.1)

    return c


if __name__ == "__main__":
    c = test_get_bundle_optical2()
    r = 31.4
    route_length = 35 + 77.6 + 35 + r + 35 + 190.6 + 35 + r + 15
    print("approx length = ", route_length, "um")
    c.show()
