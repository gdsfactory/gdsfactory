"""Straight waveguide."""
from typing import Iterable, Optional

from pp.add_padding import get_padding_points
from pp.cell import cell
from pp.component import Component
from pp.cross_section import strip
from pp.path import component
from pp.path import straight as straight_path
from pp.snap import snap_to_grid
from pp.tech import TECH
from pp.types import CrossSectionFactory, Layer


@cell
def straight(
    length: float = 10.0,
    npoints: int = 2,
    layers_cladding: Optional[Iterable[Layer]] = TECH.waveguide.strip.layers_cladding,
    cladding_offset: float = TECH.waveguide.strip.cladding_offset,
    cross_section_factory: Optional[CrossSectionFactory] = None,
    **cross_section_settings
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: of straight
        npoints: number of points
        cross_section: cross_section or function that returns a cross_section
        **cross_section_settings

    """
    cross_section_factory = cross_section_factory or strip

    p = straight_path(length=length, npoints=npoints)
    cross_section = cross_section_factory(**cross_section_settings)
    c = component(p, cross_section)
    c.length = snap_to_grid(length)
    c.width = cross_section.info["width"]
    points = get_padding_points(
        component=c,
        default=0,
        bottom=cladding_offset,
        top=cladding_offset,
    )
    for layer in layers_cladding or []:
        c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    # c = straight(length=10.0)
    # c.pprint()

    c = straight(
        length=10.001,
        width=0.5,
        cross_section={"clad": dict(width=3, offset=0, layer=(111, 0))},
    )
    c = straight()
    # c = straight(cross_section_factory=TECH.waveguide.rib_slab90)
    # print(c.name)
    # print(c.length)
    # print(c.ports)
    c.show(show_ports=True)
    # c.plot()
