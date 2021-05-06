"""Straight waveguide."""
from pp.cell import cell
from pp.component import Component
from pp.cross_section import cross_section
from pp.path import extrude
from pp.path import straight as straight_path
from pp.snap import snap_to_grid
from pp.tech import TECH


@cell
def straight(
    length: float = 10.0,
    npoints: int = 2,
    cross_section_settings=TECH.waveguide.strip,
    **kwargs
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: of straight
        npoints: number of points
        cross_section_settings: settings for cross_section
        kwargs: overwrites cross_section_settings

    """
    p = straight_path(length=length, npoints=npoints)
    settings = dict(cross_section_settings)
    settings.update(**kwargs)
    x = cross_section(**settings)
    c = extrude(p, x)
    c.length = snap_to_grid(length)
    c.width = x.info["width"]
    return c


if __name__ == "__main__":
    # c = straight(length=10.0)
    # c.pprint()

    # c = straight(
    #     length=10.001,
    #     width=0.5,
    #     cross_section={"clad": dict(width=3, offset=0, layer=(111, 0))},
    # )

    # c = straight(settings=TECH.waveguide.metal_routing)

    # settings.update(width=2)
    # c = straight(**settings)

    # settings = TECH.waveguide.rib_slab90
    # c = straight(cross_section_settings=TECH.waveguide.rib_slab90)

    c = straight()
    c.pprint()
    # print(c.get_settings()['settings']['cross_section_settings']['layers_cladding'])

    # print(c.name)
    # print(c.length)
    # print(c.ports)
    c.show(show_ports=True)
    # c.plot()
