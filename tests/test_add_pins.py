from __future__ import annotations

from functools import partial

import pytest

import gdsfactory as gf
from gdsfactory.add_pins import add_bbox_siepic, add_pins_siepic
from gdsfactory.generic_tech import LAYER

cladding_layers_optical_siepic = ("DEVREC",)  # for SiEPIC verification
cladding_offsets_optical_siepic = (0,)  # for SiEPIC verification

add_pins_siepic_100nm = partial(add_pins_siepic, pin_length=0.1)

strip_siepic100nm = partial(
    gf.cross_section.cross_section,
    add_pins=add_pins_siepic_100nm,
    add_bbox=add_bbox_siepic,
    cladding_layers=cladding_layers_optical_siepic,
    cladding_offsets=cladding_offsets_optical_siepic,
)


@pytest.mark.parametrize("optical_routing_type", [0, 1])
def test_add_pins_with_routes(optical_routing_type) -> None:
    """Add pins to a straight ensure that all the routes have pins."""
    cross_section = "xs_sc"
    c = gf.components.straight(length=1.0, cross_section=cross_section)
    gc = gf.components.grating_coupler_elliptical_te(cross_section=cross_section)
    cc = gf.routing.add_fiber_single(
        component=c,
        grating_coupler=[gc, gf.components.grating_coupler_tm],
        optical_routing_type=optical_routing_type,
        cross_section=cross_section,
    )
    pins_component = cc.extract(layers=(LAYER.PORT,))
    assert len(pins_component.polygons) == 12, len(pins_component.polygons)


def test_add_pins() -> None:
    """Ensure that all the waveguide has 2 pins."""
    cross_section = "xs_sc"
    c = gf.components.straight(length=1.0, cross_section=cross_section)
    pins_component = c.extract(layers=(LAYER.PORT,))
    assert len(pins_component.polygons) == 2, len(pins_component.polygons)


if __name__ == "__main__":
    # test_add_pins()
    test_add_pins_with_routes(0)
