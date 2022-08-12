import pytest

import gdsfactory as gf
from gdsfactory.tech import LAYER


@pytest.mark.parametrize("optical_routing_type", [0, 1])
def test_add_pins_with_routes(optical_routing_type) -> gf.Component:
    """Add grating couplers to a straight ensure that all the routes have
    pins."""
    c = gf.components.straight(length=11.0)
    gc = gf.components.grating_coupler_elliptical_te()
    cc = gf.routing.add_fiber_single(
        component=c,
        grating_coupler=[gc, gf.components.grating_coupler_tm],
        optical_routing_type=optical_routing_type,
    )
    pins_component = cc.extract(layers=(LAYER.PORT,))
    pins_component.name = "test_add_pins_with_routes_component"
    assert len(pins_component.polygons) == 20, len(pins_component.polygons)
    return cc


def test_add_pins() -> None:
    """ensure that all the waveguide has 2 pins."""
    c = gf.components.straight(length=11.0)
    pins_component = c.extract(layers=(LAYER.PORT,))
    pins_component.name = "test_add_pins_component"
    assert len(pins_component.polygons) == 4, len(pins_component.polygons)


if __name__ == "__main__":
    test_add_pins()
    # c = test_add_pins_with_routes(0)
    # c.show(show_ports=True)

    # test_add_pins_with_routes(1)

    # c = straight_nc()
    # gc = gf.components.grating_coupler_elliptical_te(wg_width=WIDTH_NITRIDE_CBAND)
    # cc = gf.routing.add_fiber_single(
    #     component=c,
    #     grating_coupler=gc,
    #     straight=straight_nc,
    #     optical_routing_type=1,
    # )
    # cc.show(show_ports=True)
    # pins_component = cc.extract(layers=(LAYER.PIN,))
    # print(len(pins_component.polygons))
