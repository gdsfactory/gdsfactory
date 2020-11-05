import numpy as np
import pp


def test_link_optical_ports_no_grouping():
    c = pp.Component()

    w = c << pp.c.waveguide_array(n_waveguides=4, spacing=200)
    d = c << pp.c.nxn()
    d.y = w.y
    d.xmin = w.xmax + 200

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

    r = pp.routing.link_optical_ports_no_grouping(ports1, ports2, sort_ports=True)
    c.add(r)
    # print(r[0].parent.length)
    assert np.isclose(r[0].parent.length, 489.41592653589794)
    return c


if __name__ == "__main__":
    c = test_link_optical_ports_no_grouping()
    pp.show(c)
