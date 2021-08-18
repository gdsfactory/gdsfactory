import pytest

import gdsfactory as gf
from gdsfactory.samples.pdk.fab_c import (
    LAYER,
    WIDTH_NITRIDE_CBAND,
    bend_euler_c,
    fabc_nitride_cband,
    straight_c,
)


@pytest.mark.parametrize("optical_routing_type", [0, 1])
def test_add_pins_with_routes(optical_routing_type) -> None:
    """
    Add grating couplers to a straight
    ensure that all the routes have pins

    """
    c = straight_c(length=11.0)
    gc = gf.components.grating_coupler_elliptical_te(
        wg_width=WIDTH_NITRIDE_CBAND, layer=LAYER.WGN
    )
    cc = gf.routing.add_fiber_single(
        component=c,
        grating_coupler=gc,
        cross_section=fabc_nitride_cband,
        straight_factory=straight_c,
        bend_factory=bend_euler_c,
        optical_routing_type=optical_routing_type,
    )
    pins_component = cc.extract(layers=(LAYER.PIN,))
    pins_component.name = "test_add_pins_with_routes_component"
    cc.show()
    assert len(pins_component.polygons) == 8, len(pins_component.polygons)


def test_add_pins() -> None:
    """ensure that all the waveguide has 2 pins"""
    c = straight_c(length=11.0)
    pins_component = c.extract(layers=(LAYER.PIN,))
    pins_component.name = "test_add_pins_component"
    assert len(pins_component.polygons) == 2, len(pins_component.polygons)


if __name__ == "__main__":
    # test_add_pins()
    test_add_pins_with_routes(0)
    # test_add_pins_with_routes(1)

    # c = straight_c()
    # gc = gf.components.grating_coupler_elliptical_te(wg_width=WIDTH_NITRIDE_CBAND)
    # cc = gf.routing.add_fiber_single(
    #     component=c,
    #     grating_coupler=gc,
    #     straight_factory=straight_c,
    #     optical_routing_type=1,
    # )
    # cc.show()
    # pins_component = cc.extract(layers=(LAYER.PIN,))
    # print(len(pins_component.polygons))
