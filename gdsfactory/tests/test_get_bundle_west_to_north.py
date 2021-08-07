from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.component import Component


def test_get_bundle_west_to_north(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:

    lengths = {}

    c = gf.Component("test_get_bundle_west_to_north")
    w = h = 10
    c = gf.Component()
    pad_south = gf.components.pad_array(
        port_list=["S"], pitch=15.0, pad_settings=dict(width=w, height=h), n=3
    )
    pad_north = gf.components.pad_array(
        port_list=["N"], pitch=15.0, pad_settings=dict(width=w, height=h), n=3
    )
    pl = c << pad_south
    pb = c << pad_north
    pl.rotate(90)
    pb.move((100, -100))

    pbports = pb.get_ports_list()
    ptports = pl.get_ports_list()

    routes = gf.routing.get_bundle(
        pbports,
        ptports,
        bend_factory=gf.components.wire_corner,
        waveguide="metal_routing",
    )
    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)
    return c


def test_get_bundle_west_to_north2(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:

    lengths = {}

    c = gf.Component("test_get_bundle_west_to_north2")
    pbottom_facing_north = gf.port_array(midpoint=(0, 0), orientation=90, pitch=(30, 0))
    ptop_facing_west = gf.port_array(
        midpoint=(100, 100), orientation=180, pitch=(0, -30)
    )

    routes = gf.routing.get_bundle(
        pbottom_facing_north,
        ptop_facing_west,
        bend_factory=gf.components.wire_corner,
        waveguide="metal_routing",
    )

    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)
    return c


if __name__ == "__main__":
    c = test_get_bundle_west_to_north(None, check=False)
    c.show()
