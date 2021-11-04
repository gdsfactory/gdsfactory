import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory import Port
from gdsfactory.component import Component
from gdsfactory.difftest import difftest
from gdsfactory.routing.get_bundle import get_bundle


def test_get_bundle(data_regression: DataRegressionFixture, check: bool = True):
    xs_top = [-100, -90, -80, 0, 10, 20, 40, 50, 80, 90, 100, 105, 110, 115]
    pitch = 127.0
    N = len(xs_top)
    xs_bottom = [(i - N / 2) * pitch for i in range(N)]

    top_ports = [Port(f"top_{i}", (xs_top[i], 0), 0.5, 270) for i in range(N)]

    bot_ports = [Port(f"bot_{i}", (xs_bottom[i], -400), 0.5, 90) for i in range(N)]

    c = gf.Component("test_get_bundle")
    routes = get_bundle(
        top_ports, bot_ports, start_straight_length=5, end_straight_length=10
    )
    lengths = {}
    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)
        difftest(c)
    return c


@pytest.mark.parametrize("config", ["A", "C"])
def test_connect_corner(
    config: str, data_regression: DataRegressionFixture, check: bool = True, N=6
) -> Component:
    d = 10.0
    sep = 5.0
    c = Component(name=f"test_connect_corner_{config}")

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

    lengths = {}
    i = 0
    if config in ["A", "C"]:
        for ports1, ports2 in zip(ports_A, ports_B):
            routes = get_bundle(ports1, ports2)
            for route in routes:
                c.add(route.references)
                lengths[i] = route.length
                i += 1

    elif config in ["B", "D"]:
        for ports1, ports2 in zip(ports_A, ports_B):
            routes = get_bundle(ports2, ports1)
            for route in routes:
                c.add(route.references)
                lengths[i] = route.length
                i += 1

    if check:
        data_regression.check(lengths)
        difftest(c)
    return c


def test_get_bundle_udirect(
    data_regression: DataRegressionFixture, check: bool = True, dy=200, angle=270
):
    xs1 = [-100, -90, -80, -55, -35, 24, 0] + [200, 210, 240]
    axis = "X" if angle in [0, 180] else "Y"

    pitch = 10.0
    N = len(xs1)
    xs2 = [70 + i * pitch for i in range(N)]

    if axis == "X":
        ports1 = [Port(f"top_{i}", (0, xs1[i]), 0.5, angle) for i in range(N)]
        ports2 = [Port(f"bot_{i}", (dy, xs2[i]), 0.5, angle) for i in range(N)]

    else:
        ports1 = [Port(f"top_{i}", (xs1[i], 0), 0.5, angle) for i in range(N)]
        ports2 = [Port(f"bot_{i}", (xs2[i], dy), 0.5, angle) for i in range(N)]

    c = gf.Component(name="test_get_bundle_udirect")
    routes = get_bundle(
        ports1, ports2, bend=gf.components.bend_circular, end_straight_length=30
    )
    lengths = {}
    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)
        difftest(c)

    return c


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_get_bundle_u_indirect(
    data_regression: DataRegressionFixture, angle, check: bool = True, dy=-200
):

    xs1 = [-100, -90, -80, -55, -35] + [200, 210, 240]

    axis = "X" if angle in [0, 180] else "Y"

    pitch = 10.0
    N = len(xs1)
    xs2 = [50 + i * pitch for i in range(N)]

    a1 = angle
    a2 = a1 + 180

    if axis == "X":
        ports1 = [Port("top_{}".format(i), (0, xs1[i]), 0.5, a1) for i in range(N)]

        ports2 = [Port("bot_{}".format(i), (dy, xs2[i]), 0.5, a2) for i in range(N)]

    else:
        ports1 = [Port("top_{}".format(i), (xs1[i], 0), 0.5, a1) for i in range(N)]

        ports2 = [Port("bot_{}".format(i), (xs2[i], dy), 0.5, a2) for i in range(N)]

    c = gf.Component(f"test_get_bundle_u_indirect_{angle}_{dy}")

    routes = get_bundle(
        ports1,
        ports2,
        bend=gf.components.bend_circular,
        end_straight_length=15,
        start_straight_length=5,
    )
    lengths = {}
    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)
        difftest(c)

    return c


def test_facing_ports(
    data_regression: DataRegressionFixture,
    check: bool = True,
):

    dy = 200.0
    xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

    pitch = 10.0
    N = len(xs1)
    xs2 = [-20 + i * pitch for i in range(N // 2)]
    xs2 += [400 + i * pitch for i in range(N // 2)]

    a1 = 90
    a2 = a1 + 180

    ports1 = [Port(f"top_{i}", (xs1[i], 0), 0.5, a1) for i in range(N)]
    ports2 = [Port(f"bottom_{i}", (xs2[i], dy), 0.5, a2) for i in range(N)]

    c = gf.Component("test_facing_ports")
    routes = get_bundle(ports1, ports2)
    lengths = {}
    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)
        difftest(c)

    return c


if __name__ == "__main__":

    # c = test_get_bundle(None, check=False)
    # c = test_connect_corner(None, config="A", check=False)
    # c = test_connect_corner(None, config="C", check=False) # FIXME
    # c = test_get_bundle_udirect(None, check=False)
    c = test_get_bundle_u_indirect(None, check=False, angle=90)
    # c = test_get_bundle_u_indirect(None, angle=0, check=False)
    # c = test_facing_ports(None, check=False)
    c.show()
