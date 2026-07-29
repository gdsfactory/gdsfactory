from __future__ import annotations

import warnings

import pytest
from kfactory.routing.generic import PlacerError

import gdsfactory as gf


def _make_failing_route_args() -> tuple[gf.Component, gf.Port, gf.Port]:
    """Return (component, port1, port2) that will fail to route.

    Two straights placed with the same orientation but only 2um apart
    vertically. Both o2 ports face right so the router needs U-turns
    that require far more space than 2um (default radius ~10um).
    """
    c = gf.Component()
    s1 = c << gf.components.straight(length=5, cross_section="strip")
    s2 = c << gf.components.straight(length=5, cross_section="strip")
    s2.dmove((0, 2))
    return c, s1.ports["o2"], s2.ports["o2"]


def test_route_single_on_placer_error_none() -> None:
    """on_placer_error=None silently falls back to error markers."""
    c, p1, p2 = _make_failing_route_args()
    route = gf.routing.route_single(
        c,
        p1,
        p2,
        cross_section="strip",
        on_placer_error=None,
    )
    assert route is not None


def test_route_single_on_placer_error_error() -> None:
    """on_placer_error='error' raises an exception."""
    c, p1, p2 = _make_failing_route_args()
    with pytest.raises((PlacerError, ValueError)):
        gf.routing.route_single(
            c,
            p1,
            p2,
            cross_section="strip",
            on_placer_error="error",
        )


def test_route_single_on_placer_error_warning() -> None:
    """on_placer_error='warning' emits a warning and falls back to error markers."""
    c, p1, p2 = _make_failing_route_args()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        route = gf.routing.route_single(
            c,
            p1,
            p2,
            cross_section="strip",
            on_placer_error="warning",
        )
        routing_warnings = [x for x in w if "Routing failed" in str(x.message)]
        assert len(routing_warnings) >= 1
    assert route is not None


def test_route_single_on_placer_error_show_error() -> None:
    """on_placer_error='show_error' raises and sends error to klayout marker database."""
    c, p1, p2 = _make_failing_route_args()
    with pytest.raises((PlacerError, ValueError)):
        gf.routing.route_single(
            c,
            p1,
            p2,
            cross_section="strip",
            on_placer_error="show_error",
        )
