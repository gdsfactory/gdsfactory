"""Straight waveguide."""
from typing import Optional

from pp.cell import cell
from pp.component import Component
from pp.cross_section import strip
from pp.path import component
from pp.path import straight as straight_path
from pp.snap import snap_to_grid
from pp.types import CrossSectionFactory


@cell
def straight(
    length: float = 10.0,
    npoints: int = 2,
    snap_to_grid_nm: int = 1,
    cross_section_factory: Optional[CrossSectionFactory] = None,
    **cross_section_settings
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: of straight
        npoints: number of points
        snap_to_grid_nm: snaps points a nm grid
        cross_section: cross_section or function that returns a cross_section
        **cross_section_settings

    """
    cross_section_factory = cross_section_factory or strip

    p = straight_path(length=length, npoints=npoints)
    cross_section = cross_section_factory(**cross_section_settings)
    c = component(p, cross_section, snap_to_grid_nm=snap_to_grid_nm)
    c.length = snap_to_grid(length)
    c.width = cross_section.info["width"]
    return c


if __name__ == "__main__":
    # c = straight(length=10.0)
    # c.pprint()

    c = straight(length=10.001, width=3)
    # print(c.name)
    # print(c.length)
    # print(c.ports)
    c.show()
