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
    p1 = left.ports.filter(orientation=0)
    p2 = right.ports.filter(orientation=180)

    with pytest.warns(RouteWarning):
        gf.routing.route_bundle(
            c,
            p1,
            p2,
        )


if __name__ == "__main__":
    # test_route_error_bundle()
    c = gf.Component()

    w = gf.components.array(
        partial(gf.components.straight, layer=(2, 0)),
        rows=3,
        columns=1,
        spacing=(0, 50),
    )

    left = c << w
    right = c << w
    right.d.move((200, 100))
    p1 = left.ports.filter(orientation=0)
    p2 = right.ports.filter(orientation=180)

    gf.routing.route_bundle(
        c,
        p1,
        p2,
    )
    c.show()
