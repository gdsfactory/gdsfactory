from __future__ import annotations

import pytest

import gdsfactory as gf
from gdsfactory.routing.manhattan import RouteWarning


def test_route_error() -> None:
    """Ensures that an impossible route raises RouteWarning."""
    c = gf.Component("test_route_error")
    w = gf.components.straight()
    left = c << w
    right = c << w
    right.move((100, 80))

    p1 = left.ports["o2"]
    p2 = right.ports["o2"]

    with pytest.warns(RouteWarning):
        route = gf.routing.get_route_from_steps(
            port1=p2,
            port2=p1,
            steps=[
                {"x": 20, "y": 0},
                {"x": 20, "y": 20},
                {"x": 120, "y": 20},
                {"x": 120, "y": 80},
            ],
        )
        c.add(route.references)
        c.add(route.labels)


def test_route_error2() -> None:
    """Impossible route."""
    c = gf.Component("pads_route_from_steps")
    pt = c << gf.components.pad_array(orientation=270, columns=3)
    pb = c << gf.components.pad_array(orientation=90, columns=3)
    pt.move((100, 200))
    route = gf.routing.get_route_from_steps(
        pt.ports["e11"],
        pb.ports["e11"],
        steps=[
            {"y": 100},
        ],
        cross_section="metal_routing",
        bend=gf.components.wire_corner,
    )
    c.add(route.references)
    c.add(route.labels)


if __name__ == "__main__":
    test_route_error()
