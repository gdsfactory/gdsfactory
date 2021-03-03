"""Straight waveguides"""
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
    width: Optional[float] = None,
    layer: Optional[Layer] = None,
    cross_section_factory: CrossSectionFactory = strip,
    tech: Tech = TECH_SILICON_C,
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: of straight
        npoints: number of points
        width: waveguide width (defaults to tech.wg_width)
        layer: layer for bend (defaults to tech.layer_wg)
        cross_section_factory: function that returns a cross_section
        tech: Technology

    .. plot::
      :include-source:

      import pp

      c = pp.c.waveguide(length=10)
      c.plot()

    """
    p = straight(length=length, npoints=npoints)
    width = width or tech.wg_width
    cross_section = cross_section_factory(width=width, layer=layer)
    c = component(p, cross_section)
    c.width = width
    c.length = snap_to_grid(length)
    return c


if __name__ == "__main__":
    c = waveguide(length=10, width=10)
    # c = waveguide(length=10)
    c.pprint()
    # print(c.ports)
    c.show()
