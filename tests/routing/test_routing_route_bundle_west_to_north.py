from __future__ import annotations

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf


def test_route_bundle_west_to_north(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    lengths = {}

    c = gf.Component()
    pad = gf.components.pad
    c = gf.Component()
    pad_south = gf.components.pad_array(
        port_orientation=270, pad=pad, size=(10, 10), column_pitch=15
    )
    pad_north = gf.components.pad_array(
        port_orientation=90, pad=pad, size=(10, 10), column_pitch=15
    )
    pl = c << pad_south
    pb = c << pad_north
    pl.rotate(90)
    pb.dmove((100, -100))

    pbports = pb.ports
    ptports = pl.ports

    c.add_ports(pbports, prefix="N")
    c.add_ports(ptports, prefix="S")

    routes = gf.routing.route_bundle(
        c,
        pbports,
        ptports,
        bend=gf.components.wire_corner,
        port_type="electrical",
        cross_section="metal3",
    )
    for i, route in enumerate(routes):
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)
