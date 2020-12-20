import numpy as np

import pp


def test_connect_bundle_u_direct_different_x():
    """ u direct with different x """
    c = pp.Component()

    w = c << pp.c.waveguide_array(n_waveguides=4, spacing=200)
    d = c << pp.c.nxn()
    d.y = w.y
    d.xmin = w.xmax + 200

    ports1 = w.get_ports_list(prefix="E")
    ports2 = d.get_ports_list(prefix="E")

    ports1 = [
        w.ports["E0"],
        w.ports["E1"],
    ]
    ports2 = [
        d.ports["E1"],
        d.ports["E0"],
    ]

    r = pp.routing.connect_bundle(ports1, ports2, sort_ports=True)
    # print(r[0].parent.length)
    # print(r[1].parent.length)
    assert np.isclose(r[0].parent.length, 316.9359265358979)
    assert np.isclose(r[1].parent.length, 529.268926535898)
    c.add(r)
    return c


if __name__ == "__main__":
    c = test_connect_bundle_u_direct_different_x()
    pp.show(c)
