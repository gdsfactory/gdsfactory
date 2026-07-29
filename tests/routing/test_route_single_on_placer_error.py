from __future__ import annotations

import warnings

import pytest
from kfactory.routing.generic import PlacerError

import gdsfactory as gf


def _make_failing_route_args() -> tuple[gf.Component, gf.Port, gf.Port]:
    """Return (component, port1, port2) that will fail to route.

    Both ports face the same direction with a tiny offset, making it
    impossible to fit the default bend radius.
    """
    c = gf.Component()
    p1 = gf.Port(name="o1", center=(0, 0), width=0.5, orientation=0, layer=(1, 0))
    p2 = gf.Port(name="o2", center=(3, 3), width=0.5, orientation=0, layer=(1, 0))
    return c, p1, p2


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
