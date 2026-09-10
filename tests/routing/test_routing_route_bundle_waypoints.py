from __future__ import annotations

from collections.abc import Iterator

import kfactory as kf
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.routing.route_bundle import _ensure_manhattan_waypoints, route_bundle


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


@pytest.mark.parametrize("base_y", [99.999, 1234.0])
def test_ensure_manhattan_waypoints_one_dbu_offset_is_position_independent(
    base_y: float,
) -> None:
    """A one-dbu offset must classify the same wherever it sits in the layout.

    `abs(1234.001 - 1234.0)` evaluates to slightly less than one dbu while
    `abs(100.0 - 99.999)` evaluates to slightly more, so a tolerance sitting on
    the exact dbu boundary classified identical geometry differently depending
    on the absolute coordinates.
    """
    dbu = gf.kcl.dbu
    result = _ensure_manhattan_waypoints(
        [kf.kdb.DPoint(0, base_y), kf.kdb.DPoint(10, base_y + dbu)]
    )

    # Within tolerance: treated as horizontal, no corner inserted.
    assert [(p.x, p.y) for p in result] == [(0.0, base_y), (10.0, base_y + dbu)]


@pytest.fixture
def fine_grid() -> Iterator[None]:
    """Temporarily halve the layout database unit."""
    from gdsfactory.gpdk import PDK

    original_dbu = gf.kcl.dbu
    gf.kcl.clear_kcells()
    try:
        gf.kcl.dbu = 0.0005
        yield
    finally:
        gf.kcl.clear_kcells()
        gf.kcl.dbu = original_dbu
        PDK.activate(force=True)


def test_ensure_manhattan_waypoints_tolerance_tracks_dbu(fine_grid: None) -> None:
    """The tolerance follows the active grid, not a hardcoded 1 nm."""
    # 0.001 um is one dbu on the default grid (collapsed, see the test above)
    # but two dbu here, so it is a real diagonal and gets a corner.
    result = _ensure_manhattan_waypoints(
        [kf.kdb.DPoint(0, 0), kf.kdb.DPoint(10, 0.001)]
    )

    assert [(p.x, p.y) for p in result] == [(0.0, 0.0), (10.0, 0.0), (10.0, 0.001)]


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
