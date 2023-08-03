from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.cross_section import rib_conformal, strip_rib_tip
from gdsfactory.route_info import route_info
from gdsfactory.typings import CrossSectionSpec


@cell
def taper_cross_section(
    cross_section1: CrossSectionSpec = strip_rib_tip,
    cross_section2: CrossSectionSpec = rib_conformal,
    length: float = 10,
    npoints: int = 100,
    linear: bool = False,
    width_type: str = "sine",
    **kwargs,
) -> Component:
    r"""Returns taper transition between cross_section1 and cross_section2.

    Args:
        cross_section1: start cross_section factory.
        cross_section2: end cross_section factory.
        length: transition length.
        npoints: number of points.
        linear: shape of the transition, sine when False.
        width_type: shape of the transition ONLY IF linear is False
        kwargs: cross_section settings for section2.


    .. code::

                           _____________________
                          /
                  _______/______________________
                        /
       cross_section1  |        cross_section2
                  ______\_______________________
                         \
                          \_____________________


    """
    x1 = gf.get_cross_section(cross_section1)
    x2 = gf.get_cross_section(cross_section2, **kwargs)
    transition = gf.path.transition(
        cross_section1=x1,
        cross_section2=x2,
        width_type="linear" if linear else width_type,
    )
    taper_path = gf.path.straight(length=length, npoints=npoints)

    c = gf.Component()
    ref = c << gf.path.extrude(taper_path, cross_section=transition)
    c.add_ports(ref.ports)
    c.absorb(ref)
    if "type" in x1.info and x1.info["type"] == x2.info.get("type"):
        c.info["route_info"] = route_info(x1.info["type"], length=length, taper=True)
    return c


taper_cross_section_linear = partial(taper_cross_section, linear=True, npoints=2)
taper_cross_section_sine = partial(taper_cross_section, linear=False, npoints=101)
taper_cross_section_parabolic = partial(
    taper_cross_section, linear=False, width_type="parabolic", npoints=101
)


if __name__ == "__main__":
    # x1 = partial(strip, width=0.5)
    # x2 = partial(strip, width=2.5)
    # c = taper_cross_section_linear(x1, x2)

    # x1 = partial(strip, width=0.5)
    # x2 = partial(rib, width=2.5)
    # c = taper_cross_section_linear(x1, x2)

    # c = taper_cross_section(gf.cross_section.strip, gf.cross_section.rib)
    # c = taper_cross_section_sine()
    c = taper_cross_section_parabolic()
    print([i.name for i in c.get_dependencies()])
    c.show(show_ports=True)
