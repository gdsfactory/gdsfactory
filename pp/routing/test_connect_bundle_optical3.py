import numpy as np

import pp


def test_connect_bundle_optical3():
    """ connect 4 waveguides into a 4x1 component """
    c = pp.Component()

    w = c << pp.c.waveguide_array(n_waveguides=4, spacing=200)
    d = c << pp.c.nxn(west=4, east=1)
    d.y = w.y
    d.xmin = w.xmax + 200

    ports1 = w.get_ports_list(prefix="E")
    ports2 = d.get_ports_list(prefix="W")

    r = pp.routing.link_optical_ports(ports1, ports2, sort_ports=True)
    # print(r[0].parent.length)
    # print(r[1].parent.length)
    # print(r[2].parent.length)
    # print(r[3].parent.length)

    assert np.isclose(r[0].parent.length, 489.4159265358979)
    assert np.isclose(r[3].parent.length, 489.4159265358979)

    assert np.isclose(r[1].parent.length, 290.74892653589797)
    assert np.isclose(r[2].parent.length, 290.74892653589797)
    c.add(r)
    return c


if __name__ == "__main__":
    c = test_connect_bundle_optical3()
    pp.show(c)
