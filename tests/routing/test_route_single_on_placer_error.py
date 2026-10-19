from __future__ import annotations

import warnings

import pytest
from kfactory.routing.generic import PlacerError

import gdsfactory as gf
from gdsfactory.config import CONF


def _make_failing_route_args() -> tuple[gf.Component, gf.Port, gf.Port, list[dict]]:
    """Return (component, port1, port2, steps) that will fail to route.

    Uses steps with corners separated by only 1um, which is far too tight
    for the default bend radius (~10um). The placer will fail trying to
    fit bend cells into these corners.
    """
    c = gf.Component()
    s1 = c << gf.components.straight(length=5, cross_section="strip")
    s2 = c << gf.components.straight(length=5, cross_section="strip")
    s2.dmove((100, 80))
    steps = [
        {"dx": 1},
        {"dy": 1},
        {"dx": 1},
    ]
    return c, s1.ports["o2"], s2.ports["o1"], steps


def test_route_single_on_placer_error_none() -> None:
    """on_placer_error=None defers to CONF.on_placer_error ("warning" by default)."""
    c, p1, p2, steps = _make_failing_route_args()
    with pytest.warns(UserWarning, match="Routing failed"):
        route = gf.routing.route_single(
            c,
            p1,
            p2,
            cross_section="strip",
            steps=steps,
            on_placer_error=None,
        )
    assert route is not None


def test_route_single_on_placer_error_ignore() -> None:
    """on_placer_error='ignore' silently falls back to error markers."""
    c, p1, p2, steps = _make_failing_route_args()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        route = gf.routing.route_single(
            c,
            p1,
            p2,
            cross_section="strip",
            steps=steps,
            on_placer_error="ignore",
        )
    assert route is not None


def test_route_single_on_placer_error_error() -> None:
    """on_placer_error='error' raises an exception."""
    c, p1, p2, steps = _make_failing_route_args()
    with pytest.raises((PlacerError, ValueError)):
        gf.routing.route_single(
            c,
            p1,
            p2,
            cross_section="strip",
            steps=steps,
            on_placer_error="error",
        )


def test_route_single_on_placer_error_warning() -> None:
    """on_placer_error='warning' emits a warning and falls back to error markers."""
    c, p1, p2, steps = _make_failing_route_args()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        route = gf.routing.route_single(
            c,
            p1,
            p2,
            cross_section="strip",
            steps=steps,
            on_placer_error="warning",
        )
        routing_warnings = [x for x in w if "Routing failed" in str(x.message)]
        assert len(routing_warnings) >= 1
    assert route is not None


def test_route_single_on_placer_error_show_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """on_placer_error='show_error' raises and sends error to klayout marker database."""
    c, p1, p2, steps = _make_failing_route_args()
    shown = False

    def show_error(*args: object, **kwargs: object) -> None:
        nonlocal shown
        shown = True

    monkeypatch.setattr(c, "show", show_error)
    with pytest.raises((PlacerError, ValueError)):
        gf.routing.route_single(
            c,
            p1,
            p2,
            cross_section="strip",
            steps=steps,
            on_placer_error="show_error",
        )
    assert shown


def test_route_single_raise_on_error() -> None:
    """raise_on_error=True raises even when on_placer_error is 'warning'."""
    c, p1, p2, steps = _make_failing_route_args()
    with pytest.raises(PlacerError):
        gf.routing.route_single(
            c,
            p1,
            p2,
            cross_section="strip",
            steps=steps,
            on_placer_error="warning",
            raise_on_error=True,
        )


def test_route_single_raise_on_error_ignore() -> None:
    """raise_on_error=True does not override on_placer_error='ignore'."""
    c, p1, p2, steps = _make_failing_route_args()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        route = gf.routing.route_single(
            c,
            p1,
            p2,
            cross_section="strip",
            steps=steps,
            on_placer_error="ignore",
            raise_on_error=True,
        )
    assert route is not None


def _make_colliding_route_args() -> tuple[gf.Component, gf.Port, gf.Port]:
    """Return (component, port1, port2) whose auto-route (no steps) collides."""
    c = gf.Component()
    s1 = c << gf.components.straight(length=5, cross_section="strip")
    s2 = c << gf.components.straight(length=5, cross_section="strip")
    s2.dmove((6, 1))
    return c, s1.ports["o2"], s2.ports["o1"]


def test_route_single_raise_on_error_without_steps() -> None:
    """raise_on_error=True raises when auto-routing fails with 'warning'."""
    c, p1, p2 = _make_colliding_route_args()
    with pytest.raises(RuntimeError, match="Routing collision"):
        gf.routing.route_single(
            c,
            p1,
            p2,
            cross_section="strip",
            on_collision="warning",
            on_placer_error="warning",
            raise_on_error=True,
        )


def test_route_single_raise_on_error_defaults_to_conf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """raise_on_error=None defers to CONF.raise_on_error."""
    monkeypatch.setattr(CONF, "raise_on_error", True)
    c, p1, p2 = _make_colliding_route_args()
    with pytest.raises(RuntimeError, match="Routing collision"):
        gf.routing.route_single(c, p1, p2, cross_section="strip")
