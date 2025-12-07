from __future__ import annotations

__all__ = [
    "taper_cross_section",
    "taper_cross_section_linear",
    "taper_cross_section_parabolic",
    "taper_cross_section_sine",
]

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec


@gf.cell_with_module_name
def taper_cross_section(
    cross_section1: CrossSectionSpec = "strip_rib_tip",
    cross_section2: CrossSectionSpec = "rib2",
    length: float = 10,
    npoints: int = 100,
    linear: bool = False,
    width_type: str = "sine",
) -> Component:
    r"""Returns taper transition between cross_section1 and cross_section2.

    Args:
        cross_section1: start cross_section factory.
        cross_section2: end cross_section factory.
        length: transition length.
        npoints: number of points.
        linear: shape of the transition, sine when False.
        width_type: shape of the transition ONLY IF linear is False


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
    x2 = gf.get_cross_section(cross_section2)
    transition = gf.path.transition(
        cross_section1=x1,
        cross_section2=x2,
        width_type="linear" if linear else width_type,  # type: ignore
        offset_type="linear" if linear else width_type,  # type: ignore
    )
    taper_path = gf.path.straight(length=length, npoints=npoints)

    c = gf.Component()
    ref = c << gf.path.extrude_transition(taper_path, transition=transition)
    c.add_ports(ref.ports)
    c.add_route_info(cross_section=x1, length=length, taper=True)
    c.flatten()
    return c


taper_cross_section_linear = partial(taper_cross_section, linear=True, npoints=2)
taper_cross_section_sine = partial(taper_cross_section, linear=False, npoints=101)
taper_cross_section_parabolic = partial(
    taper_cross_section, linear=False, width_type="parabolic", npoints=101
)
