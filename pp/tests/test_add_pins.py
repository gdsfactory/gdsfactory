import pytest

import pp
from pp.samples.pdk.fab_c import LAYER, LIBRARY, WIDTH_NITRIDE_CBAND, straight_cband


@pytest.mark.parametrize("optical_routing_type", [0, 1])
def test_add_pins_with_routes(optical_routing_type) -> None:
    """
    Add grating couplers to a waveguide
    ensure that all the waveguide routes have pins

    FIXME. add pins in add_fiber_single
    """
    c = LIBRARY.get_component("straight_cband")
    # c = straight_cband(length=11.0)
    # c = straight_cband()
    gc = pp.c.grating_coupler_elliptical_te(
        wg_width=WIDTH_NITRIDE_CBAND, layer=LAYER.WGN
    )
    cc = pp.routing.add_fiber_single(
        component=c,
        grating_coupler=gc,
        waveguide="fabc_nitride_cband",
        straight_factory=straight_cband,
        optical_routing_type=optical_routing_type,
        library=LIBRARY,
    )
    pins_component = cc.extract(layers=(LAYER.PIN,))
    pins_component.name = "test_add_pins_with_routes_component"
    # print(len(pins_component.polygons))
    # cc.show()
    assert len(pins_component.polygons) == 2, print(len(pins_component.polygons))


def test_add_pins() -> None:
    """ensure that all the waveguide has 2 pins"""
    c = LIBRARY.get_component("straight_cband")
    pins_component = c.extract(layers=(LAYER.PIN,))
    pins_component.name = "test_add_pins_component"
    assert len(pins_component.polygons) == 2, print(len(pins_component.polygons))


if __name__ == "__main__":
    test_add_pins()
    test_add_pins_with_routes(0)
    test_add_pins_with_routes(1)

    # c = mzi_nitride_cband()
    # c = straight_cband()
    # gc = pp.c.grating_coupler_elliptical_te(wg_width=WIDTH_NITRIDE_CBAND)
    # cc = pp.routing.add_fiber_single(
    #     component=c,
    #     grating_coupler=gc,
    #     waveguide="nitride_cband",
    #     straight_factory=straight_cband,
    #     optical_routing_type=1,
    # )
    # cc.show()
    # pins_component = cc.extract(layers=(LAYER.PIN,))
    # print(len(pins_component.polygons))
