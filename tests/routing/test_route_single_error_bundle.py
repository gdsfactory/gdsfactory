from __future__ import annotations

from functools import partial

import pytest

import gdsfactory as gf
from gdsfactory.routing.utils import RouteWarning


def test_route_error_bundle() -> None:
    """Ensures that an impossible route raises a RouteWarning."""
    c = gf.Component()

    w = gf.components.array(
        partial(gf.components.straight, layer=(2, 0)),
        rows=3,
        columns=1,
        spacing=(0, 50),
    )

    left = c << w
    right = c << w
    right.move((200, 100))
    p1 = gf.port.get_ports_list(left.ports, orientation=0)
    p2 = gf.port.get_ports_list(right.ports, orientation=180)

    with pytest.warns(RouteWarning):
        gf.routing.route_bundle(
            p1,
            p2,
            steps=[{"x": 300}, {"x": 301}],
        )


if __name__ == "__main__":
    test_route_error_bundle()
