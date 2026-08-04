from __future__ import annotations

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.routing.route_bundle import route_bundle


def test_route_bundle_waypoints(data_regression: DataRegressionFixture) -> None:
    """route_bundle with explicit waypoints routes without error and produces consistent route lengths."""
    c = gf.Component()
    w1 = c << gf.components.straight()
    w2 = c << gf.components.straight()
    w2.dmove((200, 100))

    p1 = w1.ports["o2"]
    p2 = w2.ports["o1"]
    p1x, p1y = p1.center
    p2x, p2y = p2.center
    mid_y = (p1y + p2y) / 2

    routes = route_bundle(
        c,
        [p1],
        [p2],
        cross_section="strip",
        waypoints=[
            (p1x + 50, p1y),
            (p1x + 50, mid_y),
            (p2x - 50, mid_y),
            (p2x - 50, p2y),
        ],
    )

    lengths = {i: route.length for i, route in enumerate(routes)}
    data_regression.check(lengths)


def test_route_bundle_waypoints_collinear_collapsed(
    data_regression: DataRegressionFixture,
) -> None:
    """Collinear intermediate waypoints on the same axis must not produce degenerate routes.

    Before the fix, redundant collinear points caused kfactory to treat each
    point as a new bundle front, producing zero-length or broken segments.
    """
    c = gf.Component()
    w1 = c << gf.components.straight()
    w2 = c << gf.components.straight()
    w2.dmove((300, 0))

    p1 = w1.ports["o2"]
    p2 = w2.ports["o1"]
    p1x, p1y = p1.center
    p2x, _ = p2.center

    # Three collinear points at the same y — middle one is redundant
    routes = route_bundle(
        c,
        [p1],
        [p2],
        cross_section="strip",
        waypoints=[
            (p1x + 50, p1y),
            (p1x + 150, p1y),
            (p2x - 50, p1y),
        ],
    )

    lengths = {i: route.length for i, route in enumerate(routes)}
    data_regression.check(lengths)
