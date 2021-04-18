from typing import Optional

from phidl.device_layout import CrossSection

from pp.add_padding import add_padding
from pp.cell import cell
from pp.component import Component
from pp.cross_section import strip
from pp.path import component, straight
from pp.types import CrossSectionFactory


@cell
def coupler_straight(
    length: float = 10.0,
    gap: float = 0.27,
    cross_section_factory: Optional[CrossSectionFactory] = None,
    snap_to_grid_nm: int = 1,
    **cross_section_settings
) -> Component:
    """Coupler_straight with two parallel waveguides.

    Args:
        length: of straight
        gap: between waveguides


    """
    cross_section_factory = cross_section_factory or strip

    cross_section_single = cross_section_factory(**cross_section_settings)
    cladding_offset = cross_section_single.info["cladding_offset"]
    layers_cladding = cross_section_single.info["layers_cladding"]
    width = cross_section_single.info["width"]
    layer = cross_section_single.info["layer"]

    cross_section = CrossSection()
    cross_section.add(width=width, offset=0, layer=layer, ports=["W0", "E0"])
    cross_section.add(
        width=width,
        offset=gap + width,
        layer=layer,
        ports=["W1", "E1"],
    )

    p = straight(length=length, npoints=2)
    c = component(p, cross_section, snap_to_grid_nm=snap_to_grid_nm)
    add_padding(
        c,
        default=cladding_offset,
        right=0,
        left=0,
        top=cladding_offset,
        bottom=cladding_offset,
        layers=layers_cladding,
    )
    return c


if __name__ == "__main__":
    c = coupler_straight(width=1)
    c.show()
