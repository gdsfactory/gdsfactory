from __future__ import annotations

from functools import partial

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.port import port_array


def test_route_bundle_west_to_north(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    lengths = {}

    c = gf.Component("test_route_bundle_west_to_north")
    pad = partial(gf.components.pad, size=(10, 10))
    c = gf.Component()
    pad_south = gf.components.pad_array(
        port_orientation=270, spacing=(15.0, 0.0), pad=pad
    )
    pad_north = gf.components.pad_array(
        port_orientation=90, spacing=(15.0, 0.0), pad=pad
    )
    pl = c << pad_south
    pb = c << pad_north
    pl.rotate(90)
    pb.move((100, -100))

    pbports = pb.ports
    ptports = pl.ports

    c.add_ports(pbports, prefix="N")
    c.add_ports(ptports, prefix="S")

    routes = gf.routing.route_bundle(
        c,
        pbports,
        ptports,
        bend=gf.components.wire_corner,
        enforce_port_ordering=False,
        place_port_type="electrical",
    )
    for i, route in enumerate(routes):
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)


def test_route_bundle_west_to_north2(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    layer = (1, 0)

    lengths = {}
    c = gf.Component("test_route_bundle_west_to_north2")
    pbottom_facing_north = port_array(
        center=(0, 0), orientation=90, pitch=(30, 0), layer=layer
    )
    ptop_facing_west = port_array(
        center=(100, 100), orientation=180, pitch=(0, -30), layer=layer
    )

    routes = gf.routing.route_bundle(
        c,
        pbottom_facing_north,
        ptop_facing_west,
        bend=gf.components.wire_corner,
    )

    for i, route in enumerate(routes):
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)


if __name__ == "__main__":
    test_route_bundle_west_to_north(None, check=False)
    # test_route_bundle_west_to_north2(None, check=False)
