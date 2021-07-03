import pytest

import pp
from pp.samples.pdk.fab_c import FACTORY, LAYER, WIDTH_NITRIDE_CBAND, straight_cband


@pytest.mark.parametrize("optical_routing_type", [0, 1])
def test_add_pins(optical_routing_type) -> None:
    """
    Add grating couplers to a waveguide
    ensure that all the waveguide routes have pins

    FIXME. it does not work if no `length` defined
    """
    c = straight_cband(length=11.0)
    gc = pp.c.grating_coupler_elliptical_te(
        wg_width=WIDTH_NITRIDE_CBAND, layer=LAYER.WGN
    )
    cc = pp.routing.add_fiber_single(
        component=c,
        grating_coupler=gc,
        waveguide="nitride_cband",
        straight_factory=straight_cband,
        optical_routing_type=optical_routing_type,
        factory=FACTORY,
    )
    pins_component = cc.extract(layers=(LAYER.PIN,))
    # print(len(pins_component.polygons))
    # cc.show()
    assert len(pins_component.polygons) == 8


if __name__ == "__main__":
    test_add_pins(0)
    test_add_pins(1)
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
