from __future__ import annotations

import pytest

import gdsfactory as gf


def test_route_error() -> None:
    """Ensures that an impossible route raises RouteWarning."""
    c = gf.Component()
    w = gf.components.straight()
    left = c << w
    right = c << w
    right.dmove((100, 80))

    p1 = left.ports["o2"]
    p2 = right.ports["o2"]

    with pytest.raises(gf.kf.routing.generic.PlacerError):
        gf.routing.route_single(
            c,
            cross_section="strip",
            port1=p2,
            port2=p1,
            steps=[
                {"x": 20, "y": 0},
                {"x": 20, "y": 20},
                {"x": 120, "y": 20},
                {"x": 120, "y": 80},
            ],
        )


if __name__ == "__main__":
    c = gf.Component()
    w = gf.components.straight()
    left = c << w
    right = c << w
    right.dmove((100, 80))

    p1 = left.ports["o2"]
    p2 = right.ports["o2"]

    route = gf.routing.route_single(
        c,
        port1=p2,
        port2=p1,
        cross_section="strip",
        steps=[
            {"x": 20, "y": 0},
            {"x": 20, "y": 20},
            {"x": 120, "y": 20},
            {"x": 120, "y": 80},
        ],
    )
    c.show()
