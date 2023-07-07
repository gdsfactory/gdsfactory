from __future__ import annotations

import numpy as np
import pytest

import gdsfactory as gf
from gdsfactory.routing.manhattan import RouteWarning, route_manhattan

TOLERANCE = 0.001
DEG2RAD = np.pi / 180
RAD2DEG = 1 / DEG2RAD

O2D = {0: "East", 180: "West", 90: "North", 270: "South"}


def test_manhattan_sbend_pass() -> None:
    c = gf.Component()
    length = 10
    c1 = c << gf.components.straight(length=length)
    c2 = c << gf.components.straight(length=length)

    dy = 4.0
    c2.y = dy
    c2.movex(length + 20)

    route = route_manhattan(
        input_port=c1.ports["o2"],
        output_port=c2.ports["o1"],
        radius=5.0,
        with_sbend=True,
    )

    c.add(route.references)
    if route.labels:
        c.add(route.labels)


def test_manhattan_sbend_fail() -> None:
    with pytest.warns(RouteWarning):
        c = gf.Component()
        length = 10
        c1 = c << gf.components.straight(length=length)
        c2 = c << gf.components.straight(length=length)

        dy = 4.0
        c2.y = dy
        c2.movex(length + 20)

        route = route_manhattan(
            input_port=c1.ports["o2"],
            output_port=c2.ports["o1"],
            radius=5.0,
            with_sbend=False,
        )

        c.add(route.references)
        if route.labels:
            c.add(route.labels)


if __name__ == "__main__":
    test_manhattan_sbend_pass()
    # test_manhattan_sbend_fail()

    # c = gf.Component("demo_sbend")
    # length = 10
    # c1 = c << gf.components.straight(length=length)
    # c2 = c << gf.components.straight(length=length)

    # dy = 20.0
    # dy = 4.0
    # c2.y = dy
    # c2.movex(length + 20)

    # route = route_manhattan(
    #     input_port=c1.ports["o2"],
    #     output_port=c2.ports["o1"],
    #     radius=5.0,
    #     with_sbend=False,
    # )

    # c.add(route.references)
    # c.show()
