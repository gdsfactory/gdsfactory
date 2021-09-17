import pytest

import gdsfactory as gf
from gdsfactory.routing.manhattan import RouteWarning


def test_route_error():
    """Ensures that an impossible route raises value Error"""
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
    return c


if __name__ == "__main__":
    c = test_route_error()
    c.show()
