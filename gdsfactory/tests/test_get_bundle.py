import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory import Port
from gdsfactory.component import Component
from gdsfactory.difftest import difftest
from gdsfactory.routing.get_bundle import get_bundle


def test_get_bundle(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:
    xs_top = [-100, -90, -80, 0, 10, 20, 40, 50, 80, 90, 100, 105, 110, 115]
    pitch = 127.0
    layer = (1, 0)
    N = len(xs_top)
    xs_bottom = [(i - N / 2) * pitch for i in range(N)]
    layer = (1, 0)

    top_ports = [
        Port(f"top_{i}", center=(xs_top[i], 0), width=0.5, orientation=270, layer=layer)
        for i in range(N)
    ]

    bot_ports = [
        Port(
            f"bot_{i}",
            center=(xs_bottom[i], -400),
            width=0.5,
            orientation=90,
            layer=layer,
        )
        for i in range(N)
    ]

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
    layer = (1, 0)
    c = Component(name=f"test_connect_corner_{config}")

    if config in {"A", "B"}:
        a = 100.0
        ports_A_TR = [
            Port(
                f"A_TR_{i}",
                center=(d, a / 2 + i * sep),
                width=0.5,
                orientation=0,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_A_TL = [
            Port(
                f"A_TL_{i}",
                center=(-d, a / 2 + i * sep),
                width=0.5,
                orientation=180,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_A_BR = [
            Port(
                f"A_BR_{i}",
                center=(d, -a / 2 - i * sep),
                width=0.5,
                orientation=0,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_A_BL = [
            Port(
                f"A_BL_{i}",
                center=(-d, -a / 2 - i * sep),
                width=0.5,
                orientation=180,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_A = [ports_A_TR, ports_A_TL, ports_A_BR, ports_A_BL]

        ports_B_TR = [
            Port(
                f"B_TR_{i}",
                center=(a / 2 + i * sep, d),
                width=0.5,
                orientation=90,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_B_TL = [
            Port(
                f"B_TL_{i}",
                center=(-a / 2 - i * sep, d),
                width=0.5,
                orientation=90,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_B_BR = [
            Port(
                f"B_BR_{i}",
                center=(a / 2 + i * sep, -d),
                width=0.5,
                orientation=270,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_B_BL = [
            Port(
                f"B_BL_{i}",
                center=(-a / 2 - i * sep, -d),
                width=0.5,
                orientation=270,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_B = [ports_B_TR, ports_B_TL, ports_B_BR, ports_B_BL]

    elif config in {"C", "D"}:
        a = N * sep + 2 * d
        ports_A_TR = [
            Port(
                f"A_TR_{i}",
                center=(a, d + i * sep),
                width=0.5,
                orientation=0,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_A_TL = [
            Port(
                f"A_TL_{i}",
                center=(-a, d + i * sep),
                width=0.5,
                orientation=180,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_A_BR = [
            Port(
                f"A_BR_{i}",
                center=(a, -d - i * sep),
                width=0.5,
                orientation=0,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_A_BL = [
            Port(
                f"A_BL_{i}",
                center=(-a, -d - i * sep),
                width=0.5,
                orientation=180,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_A = [ports_A_TR, ports_A_TL, ports_A_BR, ports_A_BL]

        ports_B_TR = [
            Port(
                f"B_TR_{i}",
                center=(d + i * sep, a),
                width=0.5,
                orientation=90,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_B_TL = [
            Port(
                f"B_TL_{i}",
                center=(-d - i * sep, a),
                width=0.5,
                orientation=90,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_B_BR = [
            Port(
                f"B_BR_{i}",
                center=(d + i * sep, -a),
                width=0.5,
                orientation=270,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_B_BL = [
            Port(
                f"B_BL_{i}",
                center=(-d - i * sep, -a),
                width=0.5,
                orientation=270,
                layer=layer,
            )
            for i in range(N)
        ]

        ports_B = [ports_B_TR, ports_B_TL, ports_B_BR, ports_B_BL]

    lengths = {}
    i = 0
    for ports1, ports2 in zip(ports_A, ports_B):
        if config in {"A", "C"}:
            routes = get_bundle(ports1, ports2)
            for route in routes:
                c.add(route.references)
                lengths[i] = route.length
                i += 1

        elif config in {"B", "D"}:
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
) -> Component:
    xs1 = [-100, -90, -80, -55, -35, 24, 0] + [200, 210, 240]
    axis = "X" if angle in [0, 180] else "Y"

    pitch = 10.0
    N = len(xs1)
    xs2 = [70 + i * pitch for i in range(N)]
    layer = (1, 0)

    if axis == "X":
        ports1 = [
            Port(
                f"top_{i}",
                center=(0, xs1[i]),
                width=0.5,
                orientation=angle,
                layer=layer,
            )
            for i in range(N)
        ]
        ports2 = [
            Port(
                f"bot_{i}",
                center=(dy, xs2[i]),
                width=0.5,
                orientation=angle,
                layer=layer,
            )
            for i in range(N)
        ]

    else:
        ports1 = [
            Port(
                f"top_{i}",
                center=(xs1[i], 0),
                width=0.5,
                orientation=angle,
                layer=layer,
            )
            for i in range(N)
        ]
        ports2 = [
            Port(
                f"bot_{i}",
                center=(xs2[i], dy),
                width=0.5,
                orientation=angle,
                layer=layer,
            )
            for i in range(N)
        ]

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
    data_regression: DataRegressionFixture, angle: int, check: bool = True, dy=-200
) -> Component:

    xs1 = [-100, -90, -80, -55, -35] + [200, 210, 240]

    axis = "X" if angle in {0, 180} else "Y"

    layer = (1, 0)
    pitch = 10.0
    N = len(xs1)
    xs2 = [50 + i * pitch for i in range(N)]

    a1 = angle
    a2 = a1 + 180

    if axis == "X":
        ports1 = [
            Port(f"top_{i}", center=(0, xs1[i]), width=0.5, orientation=a1, layer=layer)
            for i in range(N)
        ]
        ports2 = [
            Port(
                f"bot_{i}",
                center=(dy, xs2[i]),
                width=0.5,
                orientation=a2,
                layer=layer,
            )
            for i in range(N)
        ]

    else:
        ports1 = [
            Port(f"top_{i}", center=(xs1[i], 0), width=0.5, orientation=a1, layer=layer)
            for i in range(N)
        ]
        ports2 = [
            Port(
                f"bot_{i}",
                center=(xs2[i], dy),
                width=0.5,
                orientation=a2,
                layer=layer,
            )
            for i in range(N)
        ]

    c = gf.Component(f"test_get_bundle_u_indirect_{angle}_{dy}")

    routes = get_bundle(
        ports1,
        ports2,
        bend=gf.components.bend_circular,
        end_straight_length=15,
        start_straight_length=5,
        radius=5,
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
) -> Component:

    dy = 200.0
    xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

    pitch = 10.0
    N = len(xs1)
    xs2 = [-20 + i * pitch for i in range(N // 2)]
    xs2 += [400 + i * pitch for i in range(N // 2)]

    a1 = 90
    a2 = a1 + 180
    layer = (1, 0)

    ports1 = [
        Port(f"top_{i}", center=(xs1[i], +0), width=0.5, orientation=a1, layer=layer)
        for i in range(N)
    ]
    ports2 = [
        Port(f"bot_{i}", center=(xs2[i], dy), width=0.5, orientation=a2, layer=layer)
        for i in range(N)
    ]

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
    c = test_connect_corner(config="A", data_regression=None, check=False)
    # c = test_get_bundle_udirect(None, check=False)
    # c = test_get_bundle_u_indirect(None, check=False, angle=90)
    # c = test_get_bundle_u_indirect(None, angle=0, check=False)
    # c = test_facing_ports(None, check=False)
    c.show(show_ports=True)
