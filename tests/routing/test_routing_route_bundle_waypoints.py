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
    gf.clear_cache()
    try:
        gf.kcl.dbu = 0.0005
        yield
    finally:
        gf.clear_cache()
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


def test_route_bundle_waypoints_bend_sequence() -> None:
    """Ordered bend specs with different radii must be preserved through waypoints."""
    c = gf.Component()
    w1 = c << gf.components.straight()
    w2 = c << gf.components.straight()
    w2.dmove((900, 700))

    p1 = w1.ports["o2"]
    p2 = w2.ports["o1"]
    p2x, p2y = p2.center

    route = route_bundle(
        c,
        [p1],
        [p2],
        cross_section="strip",
        bend=[
            gf.components.bend_circular(radius=5),
            gf.components.bend_euler(radius=10),
            gf.components.bend_topic(radius=15),
            gf.components.bend_euler(radius=20),
            gf.components.bend_euler(radius=25),
        ],
        waypoints=[
            (80, 0),
            (80, 140),
            (260, 140),
            (260, 340),
            (520, 340),
            (520, 620),
            (p2x - 80, 620),
            (p2x - 80, p2y),
        ],
        auto_taper=False,
        raise_on_error=True,
    )[0]

    c.show()

    bend_names = [
        instance.cell.name
        for instance in route.instances
        if instance.cell.name.startswith("bend_")
    ]

    assert len(bend_names) >= 5
    assert bend_names[0].startswith("bend_circular")
    assert "R10" in bend_names[1]
    assert bend_names[2].startswith("bend_topic")
    assert "R20" in bend_names[3]
    assert "R25" in bend_names[4]
    assert bend_names[1].startswith("bend_euler")
    assert bend_names[3].startswith("bend_euler")
    assert bend_names[4].startswith("bend_euler")


def test_route_bundle_steps_bend_sequence() -> None:
    """Ordered bend specs with different radii must be preserved through steps."""
    c = gf.Component()
    w1 = c << gf.components.straight()
    w2 = c << gf.components.straight()
    w2.dmove((900, 700))

    p1 = w1.ports["o2"]
    p2 = w2.ports["o1"]
    p2x, p2y = p2.center

    route = route_bundle(
        c,
        [p1],
        [p2],
        cross_section="strip",
        bend=[
            gf.components.bend_circular(radius=5),
            gf.components.bend_euler(radius=10),
            gf.components.bend_topic(radius=15),
            gf.components.bend_euler(radius=20),
            gf.components.bend_euler(radius=25),
        ],
        steps=[
            {"x": 80},
            {"y": 140},
            {"x": 260},
            {"y": 340},
            {"x": 520},
            {"y": 620},
            {"x": p2x - 80},
            {"y": p2y},
        ],
        auto_taper=False,
        raise_on_error=True,
    )[0]

    bend_names = [
        instance.cell.name
        for instance in route.instances
        if instance.cell.name.startswith("bend_")
    ]

    assert len(bend_names) >= 5
    assert bend_names[0].startswith("bend_circular")
    assert "R10" in bend_names[1]
    assert bend_names[2].startswith("bend_topic")
    assert "R20" in bend_names[3]
    assert "R25" in bend_names[4]
    assert bend_names[1].startswith("bend_euler")
    assert bend_names[3].startswith("bend_euler")
    assert bend_names[4].startswith("bend_euler")
