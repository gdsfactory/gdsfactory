import numpy as np

import pp
from pp import Port
from pp.cell import cell
from pp.routing.get_bundle import get_bundle


@cell
def test_get_bundle():

    xs_top = [-100, -90, -80, 0, 10, 20, 40, 50, 80, 90, 100, 105, 110, 115]

    pitch = 127.0
    N = len(xs_top)
    xs_bottom = [(i - N / 2) * pitch for i in range(N)]

    top_ports = [Port("top_{}".format(i), (xs_top[i], 0), 0.5, 270) for i in range(N)]

    bottom_ports = [
        Port("bottom_{}".format(i), (xs_bottom[i], -400), 0.5, 90) for i in range(N)
    ]

    c = pp.Component()
    routes = get_bundle(top_ports, bottom_ports)
    lengths = [
        1180.416,
        1063.416,
        946.416,
        899.416,
        782.416,
        665.416,
        558.416,
        441.416,
        438.416,
        555.416,
        672.416,
        794.416,
        916.416,
        1038.416,
    ]

    for route, length in zip(routes, lengths):
        # print(route["length"])
        c.add(route["references"])
        assert np.isclose(route["length"], length)

    return c


@cell
def test_connect_corner(N=6):
    d = 10.0
    sep = 5.0
    c = pp.Component()

    a = 100.0
    ports_A_TR = [
        Port("A_TR_{}".format(i), (d, a / 2 + i * sep), 0.5, 0) for i in range(N)
    ]
    ports_A_TL = [
        Port("A_TL_{}".format(i), (-d, a / 2 + i * sep), 0.5, 180) for i in range(N)
    ]
    ports_A_BR = [
        Port("A_BR_{}".format(i), (d, -a / 2 - i * sep), 0.5, 0) for i in range(N)
    ]
    ports_A_BL = [
        Port("A_BL_{}".format(i), (-d, -a / 2 - i * sep), 0.5, 180) for i in range(N)
    ]

    ports_A = [ports_A_TR, ports_A_TL, ports_A_BR, ports_A_BL]

    ports_B_TR = [
        Port("B_TR_{}".format(i), (a / 2 + i * sep, d), 0.5, 90) for i in range(N)
    ]
    ports_B_TL = [
        Port("B_TL_{}".format(i), (-a / 2 - i * sep, d), 0.5, 90) for i in range(N)
    ]
    ports_B_BR = [
        Port("B_BR_{}".format(i), (a / 2 + i * sep, -d), 0.5, 270) for i in range(N)
    ]
    ports_B_BL = [
        Port("B_BL_{}".format(i), (-a / 2 - i * sep, -d), 0.5, 270) for i in range(N)
    ]

    ports_B = [ports_B_TR, ports_B_TL, ports_B_BR, ports_B_BL]

    lengths = [
        75.708,
        85.708,
        95.708,
        105.708,
        115.708,
        125.708,
        75.708,
        85.708,
        95.708,
        105.708,
        115.708,
        125.708,
        125.708,
        115.708,
        105.708,
        95.708,
        85.708,
        75.708,
        125.708,
        115.708,
        105.708,
        95.708,
        85.708,
        75.708,
    ]
    i = 0

    for ports1, ports2 in zip(ports_A, ports_B):
        routes = get_bundle(ports1, ports2)
        for route in routes:
            c.add(route["references"])
            length = lengths[i]
            i += 1
            assert np.isclose(
                route["length"], length
            ), f"{route['length']} should be {length}"

    return c


@cell
def test_get_bundle_udirect(dy=200, angle=270):
    xs1 = [-100, -90, -80, -55, -35, 24, 0] + [200, 210, 240]
    axis = "X" if angle in [0, 180] else "Y"

    pitch = 10.0
    N = len(xs1)
    xs2 = [50 + i * pitch for i in range(N)]

    if axis == "X":
        ports1 = [Port("top_{}".format(i), (0, xs1[i]), 0.5, angle) for i in range(N)]

        ports2 = [
            Port("bottom_{}".format(i), (dy, xs2[i]), 0.5, angle) for i in range(N)
        ]

    else:
        ports1 = [Port("top_{}".format(i), (xs1[i], 0), 0.5, angle) for i in range(N)]

        ports2 = [
            Port("bottom_{}".format(i), (xs2[i], dy), 0.5, angle) for i in range(N)
        ]

    c = pp.Component(name="get_bundle")
    routes = get_bundle(ports1, ports2)
    lengths = [0] * len(routes)
    for route, length in zip(routes, lengths):
        print(route["length"])
        c.add(route["references"])
        assert np.isclose(route["length"], length)
    return c


@cell
def test_get_bundle_u_indirect(dy=-200, angle=180):

    xs1 = [-100, -90, -80, -55, -35] + [200, 210, 240]

    axis = "X" if angle in [0, 180] else "Y"

    pitch = 10.0
    N = len(xs1)
    xs2 = [50 + i * pitch for i in range(N)]

    a1 = angle
    a2 = a1 + 180

    if axis == "X":
        ports1 = [Port("top_{}".format(i), (0, xs1[i]), 0.5, a1) for i in range(N)]

        ports2 = [Port("bottom_{}".format(i), (dy, xs2[i]), 0.5, a2) for i in range(N)]

    else:
        ports1 = [Port("top_{}".format(i), (xs1[i], 0), 0.5, a1) for i in range(N)]

        ports2 = [Port("bottom_{}".format(i), (xs2[i], dy), 0.5, a2) for i in range(N)]

    c = pp.Component()
    routes = get_bundle(ports1, ports2)
    lengths = [
        341.416,
        341.416,
        341.416,
        326.416,
        316.416,
        291.416,
        291.416,
        311.416,
    ]
    for route, length in zip(routes, lengths):
        # print(route["length"])
        c.add(route["references"])
        assert np.isclose(route["length"], length)
    return c


@cell
def test_facing_ports():

    dy = 200.0
    xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

    pitch = 10.0
    N = len(xs1)
    xs2 = [-20 + i * pitch for i in range(N // 2)]
    xs2 += [400 + i * pitch for i in range(N // 2)]

    a1 = 90
    a2 = a1 + 180

    ports1 = [Port("top_{}".format(i), (xs1[i], 0), 0.5, a1) for i in range(N)]
    ports2 = [Port("bottom_{}".format(i), (xs2[i], dy), 0.5, a2) for i in range(N)]

    c = pp.Component()
    routes = get_bundle(ports1, ports2)
    lengths = [
        671.416,
        481.416,
        291.416,
        291.416,
        291.416,
        276.416,
        626.416,
        401.416,
        401.416,
        381.416,
        251.416,
        391.416,
    ]
    for route, length in zip(routes, lengths):
        # print(route["length"])
        c.add(route["references"])
        assert np.isclose(route["length"], length)
    return c


if __name__ == "__main__":

    # c = test_get_bundle()
    # c = test_connect_corner()
    c = test_get_bundle_u_indirect()
    # c = test_facing_ports()
    c.show()
