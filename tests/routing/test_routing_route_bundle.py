from __future__ import annotations

import numpy as np
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory import Port
from gdsfactory.component import Component
from gdsfactory.difftest import difftest
from gdsfactory.routing.route_bundle import route_bundle
from gdsfactory.typings import AngleInDegrees, Delta


def test_route_bundle(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    if check:
        pitch = 127.0
        xs_top = [-100, -90, -80, 0, 10, 20, 40, 50, 80, 90, 100, 105, 110, 115]
        N = len(xs_top)
        xs_bottom = [(i - N / 2) * pitch for i in range(N)]
        layer = (1, 0)

        top_ports = [
            Port(
                f"top_{i}",
                center=(xs_top[i], 0),
                width=0.5,
                orientation=270,
                layer=layer,
            )
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

        c = gf.Component("test_route_bundle")
        routes = route_bundle(
            c,
            top_ports,
            bot_ports,
            start_straight_length=5,
            end_straight_length=10,
            cross_section="strip",
        )
        lengths = {i: route.length for i, route in enumerate(routes)}
        if data_regression:
            data_regression.check(lengths)  # type: ignore
            difftest(c)


@pytest.mark.parametrize("config", ["A", "C"])
def test_connect_corner(
    config: str, data_regression: DataRegressionFixture, check: bool = True, n: int = 6
) -> None:
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
            for i in range(n)
        ]

        ports_A_TL = [
            Port(
                f"A_TL_{i}",
                center=(-d, a / 2 + i * sep),
                width=0.5,
                orientation=180,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_A_BR = [
            Port(
                f"A_BR_{i}",
                center=(d, -a / 2 - i * sep),
                width=0.5,
                orientation=0,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_A_BL = [
            Port(
                f"A_BL_{i}",
                center=(-d, -a / 2 - i * sep),
                width=0.5,
                orientation=180,
                layer=layer,
            )
            for i in range(n)
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
            for i in range(n)
        ]

        ports_B_TL = [
            Port(
                f"B_TL_{i}",
                center=(-a / 2 - i * sep, d),
                width=0.5,
                orientation=90,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_B_BR = [
            Port(
                f"B_BR_{i}",
                center=(a / 2 + i * sep, -d),
                width=0.5,
                orientation=270,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_B_BL = [
            Port(
                f"B_BL_{i}",
                center=(-a / 2 - i * sep, -d),
                width=0.5,
                orientation=270,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_B = [ports_B_TR, ports_B_TL, ports_B_BR, ports_B_BL]

    elif config in {"C", "D"}:
        a = n * sep + 2 * d
        ports_A_TR = [
            Port(
                f"A_TR_{i}",
                center=(a, d + i * sep),
                width=0.5,
                orientation=0,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_A_TL = [
            Port(
                f"A_TL_{i}",
                center=(-a, d + i * sep),
                width=0.5,
                orientation=180,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_A_BR = [
            Port(
                f"A_BR_{i}",
                center=(a, -d - i * sep),
                width=0.5,
                orientation=0,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_A_BL = [
            Port(
                f"A_BL_{i}",
                center=(-a, -d - i * sep),
                width=0.5,
                orientation=180,
                layer=layer,
            )
            for i in range(n)
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
            for i in range(n)
        ]

        ports_B_TL = [
            Port(
                f"B_TL_{i}",
                center=(-d - i * sep, a),
                width=0.5,
                orientation=90,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_B_BR = [
            Port(
                f"B_BR_{i}",
                center=(d + i * sep, -a),
                width=0.5,
                orientation=270,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_B_BL = [
            Port(
                f"B_BL_{i}",
                center=(-d - i * sep, -a),
                width=0.5,
                orientation=270,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_B = [ports_B_TR, ports_B_TL, ports_B_BR, ports_B_BL]

    lengths = {}
    i = 0
    for ports1, ports2 in zip(ports_A, ports_B):  # type: ignore
        if config in {"A", "C"}:
            routes = route_bundle(c, ports1, ports2, cross_section="strip")
            for route in routes:
                lengths[i] = route.length
                i += 1

        elif config in {"B", "D"}:
            routes = route_bundle(c, ports2, ports1, cross_section="strip")
            for route in routes:
                lengths[i] = route.length
                i += 1

    if check:
        data_regression.check(lengths)  # type: ignore
        difftest(c)


def test_route_bundle_udirect(
    data_regression: DataRegressionFixture,
    check: bool = True,
    dy: Delta = 200,
    angle: float = 270,
) -> None:
    xs1 = [-100, -90, -80, -55, -35, 24, 0] + [200, 210, 240]
    axis = "X" if angle in {0, 180} else "Y"

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

    c = gf.Component("test_route_bundle_udirect")
    routes = route_bundle(
        c,
        ports1,
        ports2,
        bend=gf.components.bend_circular,
        end_straight_length=30,
        sort_ports=True,
        cross_section="strip",
    )
    lengths = {i: route.length for i, route in enumerate(routes)}

    if check:
        data_regression.check(lengths)  # type: ignore
        difftest(c)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_route_bundle_u_indirect(
    data_regression: DataRegressionFixture,
    angle: AngleInDegrees,
    check: bool = True,
    dy: Delta = -200,
) -> None:
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

    c = gf.Component(f"test_route_bundle_u_indirect_{angle}_{dy}")

    routes = route_bundle(
        c,
        ports1,
        ports2,
        bend=gf.components.bend_circular,
        end_straight_length=15,
        start_straight_length=5,
        radius=5,
        cross_section="strip",
    )
    lengths = {i: route.length for i, route in enumerate(routes)}
    if check:
        data_regression.check(lengths)  # type: ignore
        difftest(c)


def test_facing_ports(
    data_regression: DataRegressionFixture,
    check: bool = True,
) -> None:
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
        Port(f"bot_{i}", center=(xs2[i], 200), width=0.5, orientation=a2, layer=layer)
        for i in range(N)
    ]

    c = gf.Component("test_facing_ports")
    routes = route_bundle(c, ports1, ports2, cross_section="strip")

    lengths = {i: route.length for i, route in enumerate(routes)}
    if check:
        data_regression.check(lengths)  # type: ignore
        difftest(c)


def test_route_bundle_small() -> None:
    c = gf.Component()
    c1 = c << gf.components.mmi2x2()
    c2 = c << gf.components.mmi2x2()
    c2.dmove((100, 40))
    routes = route_bundle(
        c,
        [c1.ports["o3"], c1.ports["o4"]],
        [c2.ports["o2"], c2.ports["o1"]],
        separation=5.0,
        cross_section="strip",
        sort_ports=True,
    )
    for route in routes:
        assert np.isclose(route.length, 74500), route.length


def test_route_bundle_width() -> None:
    top = gf.Component()
    wg1 = top << gf.components.straight()
    wg2 = top << gf.components.straight()
    wg2.movey(50)
    wg3 = top << gf.components.straight()
    wg3.move((50, 20))
    wg4 = top << gf.components.straight()
    wg4.move((50, 70))

    route = gf.routing.route_bundle(
        top,
        [wg1["o2"], wg2["o2"]],
        [wg3["o1"], wg4["o1"]],
        layer=(1, 0),
        route_width=0.5,
    )
    assert route[0].length == 20000, route[0].length


if __name__ == "__main__":
    test_route_bundle_width()
    # test_route_bundle_small()
    # test_route_bundle_udirect(None, check=False)
    # test_route_bundle(None)
