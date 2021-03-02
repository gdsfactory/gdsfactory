import pp
from pp.component import Component


def test_get_bundle_optical3() -> Component:
    """Connect 4 waveguides into a 4x1 component."""
    c = pp.Component()

    w = c << pp.c.waveguide_array(n_waveguides=4, spacing=200)
    d = c << pp.c.nxn(west=4, east=1)
    d.y = w.y
    d.xmin = w.xmax + 200

    ports1 = w.get_ports_list(prefix="E")
    ports2 = d.get_ports_list(prefix="W")

    routes = pp.routing.link_optical_ports(ports1, ports2, sort_ports=True)

    lengths = [
        486.028,
        287.361,
        287.361,
        486.028,
    ]

    for route, length in zip(routes, lengths):
        print(route["length"])
        c.add(route["references"])
        # assert np.isclose(route["length"], length)
    return c


if __name__ == "__main__":
    c = test_get_bundle_optical3()
    c.show()
