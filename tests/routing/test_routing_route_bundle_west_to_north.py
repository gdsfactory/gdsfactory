from __future__ import annotations

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.port import port_array


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
    pl.drotate(90)
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


def test_route_bundle_west_to_north2(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    layer = "M3"
    width = 10

    lengths = {}
    c = gf.Component()
    pbottom_facing_north = port_array(
        center=(0, 0),
        orientation=90,
        pitch=(30, 0),
        layer=gf.get_layer(layer),
        width=width,
    )
    ptop_facing_west = port_array(
        center=(100, 100),
        orientation=180,
        pitch=(0, -30),
        layer=gf.get_layer(layer),
        width=width,
    )

    routes = gf.routing.route_bundle(
        c,
        pbottom_facing_north,
        ptop_facing_west,
        bend=gf.components.wire_corner,
        port_type="electrical",
        cross_section="metal_routing",
        allow_width_mismatch=True,
    )

    for i, route in enumerate(routes):
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)


if __name__ == "__main__":
    # test_route_bundle_west_to_north(None, check=False)
    test_route_bundle_west_to_north2(None, check=False)
