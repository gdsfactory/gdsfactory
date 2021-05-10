"""Straight waveguide."""
from pp.cell import cell
from pp.component import Component
from pp.cross_section import cross_section, get_cross_section_settings
from pp.path import extrude
from pp.path import straight as straight_path
from pp.snap import snap_to_grid


@cell
def straight(
    length: float = 10.0, npoints: int = 2, cross_section_name: str = "strip", **kwargs
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: of straight
        npoints: number of points
        cross_section_name: from TECH.waveguide
        kwargs: cross_section_settings

    """
    p = straight_path(length=length, npoints=npoints)
    cross_section_settings = get_cross_section_settings(cross_section_name, **kwargs)
    x = cross_section(**cross_section_settings)
    c = extrude(p, x)
    c.length = snap_to_grid(length)
    c.width = x.info["width"]
    return c


if __name__ == "__main__":

    c = straight(width=2.0)
    c = straight()
    print(c.name)
    c.pprint()
    # print(c.get_settings()['settings']['cross_section_settings']['layers_cladding'])

    # print(c.name)
    # print(c.length)
    # print(c.ports)
    c.show(show_ports=True)
    # c.plot()
