import pp
from pp import Port
from pp.routing.connect_bundle import connect_bundle


def test_connect_bundle():

    xs_top = [-100, -90, -80, 0, 10, 20, 40, 50, 80, 90, 100, 105, 110, 115]

    pitch = 127.0
    N = len(xs_top)
    xs_bottom = [(i - N / 2) * pitch for i in range(N)]

    top_ports = [Port("top_{}".format(i), (xs_top[i], 0), 0.5, 270) for i in range(N)]

    bottom_ports = [
        Port("bottom_{}".format(i), (xs_bottom[i], -400), 0.5, 90) for i in range(N)
    ]

    top_cell = pp.Component(name="connect_bundle")
    elements = connect_bundle(top_ports, bottom_ports)
    for e in elements:
        top_cell.add(e)
    top_cell.name = "connect_bundle"

    return top_cell


@pp.cell
def test_connect_corner(N=6, config="A"):

    d = 10.0

    sep = 5.0
    top_cell = pp.Component(name="connect_bundle_corners")

    if config in ["A", "B"]:
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
            Port("A_BL_{}".format(i), (-d, -a / 2 - i * sep), 0.5, 180)
            for i in range(N)
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
            Port("B_BL_{}".format(i), (-a / 2 - i * sep, -d), 0.5, 270)
            for i in range(N)
        ]

        ports_B = [ports_B_TR, ports_B_TL, ports_B_BR, ports_B_BL]

    elif config in ["C", "D"]:
        a = N * sep + 2 * d
        ports_A_TR = [
            Port("A_TR_{}".format(i), (a, d + i * sep), 0.5, 0) for i in range(N)
        ]
        ports_A_TL = [
            Port("A_TL_{}".format(i), (-a, d + i * sep), 0.5, 180) for i in range(N)
        ]
        ports_A_BR = [
            Port("A_BR_{}".format(i), (a, -d - i * sep), 0.5, 0) for i in range(N)
        ]
        ports_A_BL = [
            Port("A_BL_{}".format(i), (-a, -d - i * sep), 0.5, 180) for i in range(N)
        ]

        ports_A = [ports_A_TR, ports_A_TL, ports_A_BR, ports_A_BL]

        ports_B_TR = [
            Port("B_TR_{}".format(i), (d + i * sep, a), 0.5, 90) for i in range(N)
        ]
        ports_B_TL = [
            Port("B_TL_{}".format(i), (-d - i * sep, a), 0.5, 90) for i in range(N)
        ]
        ports_B_BR = [
            Port("B_BR_{}".format(i), (d + i * sep, -a), 0.5, 270) for i in range(N)
        ]
        ports_B_BL = [
            Port("B_BL_{}".format(i), (-d - i * sep, -a), 0.5, 270) for i in range(N)
        ]

        ports_B = [ports_B_TR, ports_B_TL, ports_B_BR, ports_B_BL]

    if config in ["A", "C"]:
        for ports1, ports2 in zip(ports_A, ports_B):
            elements = connect_bundle(ports1, ports2)
            top_cell.add(elements)

    elif config in ["B", "D"]:
        for ports1, ports2 in zip(ports_A, ports_B):
            elements = connect_bundle(ports2, ports1)
            top_cell.add(elements)

    return top_cell


@pp.cell
def test_connect_bundle_udirect(dy=200, angle=270):

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

    top_cell = pp.Component(name="connect_bundle")
    elements = connect_bundle(ports1, ports2)
    for e in elements:
        top_cell.add(e)

    return top_cell


@pp.cell
def test_connect_bundle_u_indirect(dy=-200, angle=180):

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

    top_cell = pp.Component()
    elements = connect_bundle(ports1, ports2)
    for e in elements:
        top_cell.add(e)

    return top_cell


@pp.cell
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

    top_cell = pp.Component()
    elements = connect_bundle(ports1, ports2)
    # elements = link_ports_path_length_match(ports1, ports2)
    top_cell.add(elements)

    return top_cell


if __name__ == "__main__":
    import pp

    c = test_facing_ports()
    pp.show(c)
