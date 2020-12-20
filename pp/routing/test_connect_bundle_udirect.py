import numpy as np

import pp


def test_connect_u_direct():
    w = h = 10
    c = pp.Component()
    pad_south = pp.c.pad_array(port_list=["S"], spacing=(15, 0), width=w, height=h)
    pt = c << pad_south
    pb = c << pad_south
    pb.rotate(90)
    pt.rotate(90)
    pb.move((0, -100))

    pbports = pb.get_ports_list()
    ptports = pt.get_ports_list()

    pbports.reverse()

    r = pp.routing.connect_bundle(pbports, ptports)
    # r = pp.routing.link_ports(pbports, ptports) # does not work
    c.add(r)
    # print(r[0].parent.length)
    # print(r[1].parent.length)
    # print(r[2].parent.length)
    # print(r[3].parent.length)
    # print(r[4].parent.length)
    # print(r[5].parent.length)
    assert np.isclose(r[0].parent.length, 36.435926535897934)
    assert np.isclose(r[1].parent.length, 76.43592653589793)
    assert np.isclose(r[2].parent.length, 116.43592653589793)
    assert np.isclose(r[3].parent.length, 156.43592653589792)
    assert np.isclose(r[4].parent.length, 196.43592653589795)
    assert np.isclose(r[5].parent.length, 236.43592653589795)
    return c


if __name__ == "__main__":
    c = test_connect_u_direct()
    pp.show(c)
