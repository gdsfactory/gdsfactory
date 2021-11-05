import pytest

import gdsfactory as gf
from gdsfactory.routing.manhattan import RouteWarning


def test_route_error_bundle():
    """Ensures that an impossible route raises value Error"""
    c = gf.Component("get_route_from_steps_sample")

    w = gf.components.array(
        gf.partial(gf.c.straight, layer=(2, 0)),
        rows=3,
        columns=1,
        spacing=(0, 50),
    )

    left = c << w
    right = c << w
    right.move((200, 100))
    p1 = left.get_ports_list(orientation=0)
    p2 = right.get_ports_list(orientation=180)

    with pytest.warns(RouteWarning):
        routes = gf.routing.get_bundle_from_steps(
            p1,
            p2,
            steps=[{"x": 300}, {"x": 301}],
        )

    for route in routes:
        c.add(route.references)
    return c


if __name__ == "__main__":
    c = test_route_error_bundle()
    c.show()
