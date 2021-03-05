"""Straight waveguide."""
from typing import Optional

from pp.cell import cell
from pp.component import Component
from pp.cross_section import CrossSectionFactory, strip
from pp.path import component, straight
from pp.snap import snap_to_grid
from pp.tech import TECH_SILICON_C, Tech
from pp.types import Layer


@cell
def waveguide(
    length: float = 10.0,
    npoints: int = 2,
    width: float = TECH_SILICON_C.wg_width,
    layer: Layer = TECH_SILICON_C.layer_wg,
    cross_section_factory: Optional[CrossSectionFactory] = None,
    tech: Optional[Tech] = None,
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: of straight
        npoints: number of points
        width: waveguide width
        layer: layer for
        layers_cladding: for cladding
        cladding_offset: offset from waveguide to cladding edge
        cross_section_factory: function that returns a cross_section
        tech: Technology with default

    .. plot::
      :include-source:

      import pp

      c = pp.c.waveguide(length=10)
      c.plot()

    """
    tech = tech or TECH_SILICON_C
    cross_section_factory = cross_section_factory or strip

    p = straight(length=length, npoints=npoints)
    cross_section = cross_section_factory(
        width=width,
        layer=layer,
        layers_cladding=tech.layers_cladding,
        cladding_offset=tech.cladding_offset,
    )
    c = component(p, cross_section, snap_to_grid_nm=tech.snap_to_grid_nm)
    c.width = width
    c.length = snap_to_grid(length)
    return c


if __name__ == "__main__":
    from pp.tech import TECH_METAL1

    # c = waveguide(length=10.0)
    # c.pprint()

    c = waveguide(length=10.001, width=10, tech=TECH_METAL1)
    print(c.length)
    print(c.ports)
    c.show()
