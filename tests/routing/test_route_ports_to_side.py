from __future__ import annotations

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf


def test_route_ports_to_side(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    c = gf.Component()
    cross_section = "strip"
    dummy = gf.c.nxn(north=2, south=2, west=2, east=2, cross_section=cross_section)
    dummy_ref = c << dummy
    routes, _ = gf.routing.route_ports_to_side(
        c,
        ports=dummy_ref.ports,
        side="south",
        cross_section=cross_section,
        y=-91,
        x=-100,
    )

        lengths: dict[int, int] = {i: route.length for i, route in enumerate(routes)}
    if check:
        data_regression.check(lengths)  # type: ignore


def test_route_ports_to_x(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    c = gf.Component()
    cross_section = "strip"
    dummy = gf.c.nxn(north=2, south=2, west=2, east=2, cross_section=cross_section)
    dummy_ref = c << dummy
    routes, _ = gf.routing.route_ports_to_side(
        c,
        ports=dummy_ref.ports,
        cross_section=cross_section,
        x=50,
        side="east",
    )
        lengths: dict[int, int] = {i: route.length for i, route in enumerate(routes)}
    if check:
        data_regression.check(lengths)  # type: ignore


def test_route_ports_to_y(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    c = gf.Component()
    cross_section = "strip"
    dummy = gf.c.nxn(north=2, south=2, west=2, east=2, cross_section=cross_section)
    dummy_ref = c << dummy
    routes, _ = gf.routing.route_ports_to_side(
        c,
        ports=dummy_ref.ports,
        cross_section=cross_section,
        y=50,
        side="north",
    )
        lengths: dict[int, int] = {i: route.length for i, route in enumerate(routes)}
    if check:
        data_regression.check(lengths)  # type: ignore
