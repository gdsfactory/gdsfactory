from __future__ import annotations

import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf


def test_route_south(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    c = gf.Component()
    cr = c << gf.components.mmi2x2()
    routes = gf.routing.route_south(c, cr)

    lengths: dict[int, int] = {}
    for i, route in enumerate(routes):
        lengths[i] = route.length
    if check:
        data_regression.check(lengths)


def test_route_south_nxn() -> None:
    c = gf.Component()
    cr = c << gf.components.nxn(north=4, south=2, west=2, east=2)
    routes = gf.routing.route_south(c, cr)
    assert len(routes) == 6  # 4 north + 2 east ports


def test_route_south_mzi_with_bend() -> None:
    c = gf.Component()

    @gf.cell
    def mzi_with_bend() -> gf.Component:
        c = gf.Component()
        bend = c.add_ref(gf.components.bend_euler(radius=10))
        mzi = c.add_ref(gf.components.mzi())
        bend.connect("o1", mzi.ports["o2"])
        c.add_port(name="o1", port=mzi.ports["o1"])
        c.add_port(name="o2", port=bend.ports["o2"])
        return c

    cr = c << mzi_with_bend()
    routes = gf.routing.route_south(c, cr)
    assert len(routes) == 1


def test_route_south_with_auto_taper() -> None:
    c = gf.Component()
    cr = c << gf.components.straight(length=10, width=2)
    routes = gf.routing.route_south(c, cr, auto_taper=True)
    assert len(routes) == 1


def test_route_south_invalid_type() -> None:
    c = gf.Component()
    cr = c << gf.components.mmi2x2()
    with pytest.raises(ValueError, match="not in supported"):
        gf.routing.route_south(c, cr, optical_routing_type=3)


def test_route_south_empty_ports() -> None:
    c = gf.Component()
    cr = c << gf.components.mmi2x2()
    routes = gf.routing.route_south(c, cr, excluded_ports=("o1", "o2", "o3", "o4"))
    assert len(routes) == 0


def test_route_south_with_port_names() -> None:
    c = gf.Component()
    cr = c << gf.components.nxn(north=4, south=2, west=2, east=2)
    routes = gf.routing.route_south(c, cr, port_names=["o7", "o8"])
    assert len(routes) == 2


def test_route_south_with_io_gratings() -> None:
    c = gf.Component()
    cr = c << gf.components.nxn(north=4, south=2, west=2, east=2)

    gc_line = [c << gf.components.grating_coupler_elliptical() for _ in range(6)]
    for i, gc in enumerate(gc_line[1:], 1):
        gc.movex(gc_line[0].x + i * 127)

    routes = gf.routing.route_south(
        component=c, component_to_route=cr, io_gratings_lines=[gc_line]
    )
    assert len(routes) == 6  # 4 north + 2 east ports
