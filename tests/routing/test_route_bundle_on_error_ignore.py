from __future__ import annotations

import warnings

import gdsfactory as gf
from gdsfactory.typings import Step

# Corners separated by only 1um, far tighter than the default bend radius (~10um).
# With 3 corners the bends are placed but overlap each other (routing collision).
COLLIDING_STEPS: list[Step] = [{"dx": 1}, {"dy": 1}, {"dx": 1}]

# With 5 corners the placer gives up before placing anything (placer error).
UNPLACEABLE_STEPS: list[Step] = [
    {"dx": 1},
    {"dy": 1},
    {"dx": 1},
    {"dy": 1},
    {"dx": 1},
]


def _make_route_args() -> tuple[gf.Component, list[gf.Port], list[gf.Port]]:
    """Return (component, ports1, ports2) to route with the steps above."""
    c = gf.Component()
    s1 = c << gf.components.straight(length=5, cross_section="strip")
    s2 = c << gf.components.straight(length=5, cross_section="strip")
    s2.dmove((100, 80))
    return c, [s1.ports["o2"]], [s2.ports["o1"]]


def _routing_warnings(records: list[warnings.WarningMessage]) -> list[str]:
    return [str(r.message) for r in records if "Routing failed" in str(r.message)]


def _error_marker_area(c: gf.Component) -> float:
    """Area of the error path markers placed when a route fails."""
    layer = gf.get_layer(gf.CONF.layer_error_path)
    return gf.kdb.Region(c.begin_shapes_rec(layer)).area()


def test_route_bundle_on_collision_ignore() -> None:
    """on_collision='ignore' places the route without checking for collisions."""
    c, ports1, ports2 = _make_route_args()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        routes = gf.routing.route_bundle(
            c,
            ports1,
            ports2,
            cross_section="strip",
            steps=COLLIDING_STEPS,
            on_collision="ignore",
            raise_on_error=True,
        )
    assert len(routes) == 1
    assert _routing_warnings(w) == []
    assert _error_marker_area(c) == 0


def test_route_bundle_on_collision_warning() -> None:
    """on_collision='warning' warns about the same route and falls back to markers."""
    c, ports1, ports2 = _make_route_args()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gf.routing.route_bundle(
            c,
            ports1,
            ports2,
            cross_section="strip",
            steps=COLLIDING_STEPS,
            on_collision="warning",
        )
    assert _routing_warnings(w)
    assert _error_marker_area(c) > 0


def test_route_bundle_on_placer_error_ignore() -> None:
    """on_placer_error='ignore' skips the route instead of erroring on it."""
    c, ports1, ports2 = _make_route_args()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gf.routing.route_bundle(
            c,
            ports1,
            ports2,
            cross_section="strip",
            steps=UNPLACEABLE_STEPS,
            # ignored as well, so that only the placer error is under test
            on_collision="ignore",
            on_placer_error="ignore",
            raise_on_error=True,
        )
    assert _routing_warnings(w) == []
    assert _error_marker_area(c) == 0


def test_route_bundle_on_placer_error_warning() -> None:
    """on_placer_error='warning' warns about the same route and falls back to markers."""
    c, ports1, ports2 = _make_route_args()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gf.routing.route_bundle(
            c,
            ports1,
            ports2,
            cross_section="strip",
            steps=UNPLACEABLE_STEPS,
            on_collision="ignore",
            on_placer_error="warning",
        )
    assert _routing_warnings(w)
    assert _error_marker_area(c) > 0
