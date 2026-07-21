from __future__ import annotations

import importlib
from typing import Any

import gdsfactory as gf


def test_route_single_delegates_to_route_bundle(monkeypatch: Any) -> None:
    route_single_module = importlib.import_module("gdsfactory.routing.route_single")
    captured: dict[str, Any] = {}
    expected_route = object()

    def fake_route_bundle(**kwargs: Any) -> list[object]:
        captured.update(kwargs)
        return [expected_route]

    monkeypatch.setattr(route_single_module, "route_bundle", fake_route_bundle)
    component = gf.Component()
    port1 = gf.Port(
        name="o1",
        center=(0, 0),
        width=0.5,
        orientation=0,
        layer=gf.get_layer("WG"),
    )
    port2 = gf.Port(
        name="o2",
        center=(20, 10),
        width=0.5,
        orientation=180,
        layer=gf.get_layer("WG"),
    )

    route = route_single_module.route_single(
        component,
        port1,
        port2,
        cross_section="strip",
        steps=[{"dx": 5}],
        on_error="error",
    )

    assert route is expected_route
    assert captured["component"] is component
    assert captured["ports1"] == [port1]
    assert captured["ports2"] == [port2]
    assert captured["steps"] == [{"dx": 5}]
    assert captured["raise_on_error"] is True
