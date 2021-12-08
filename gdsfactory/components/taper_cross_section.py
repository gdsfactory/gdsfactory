import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.cross_section import rib, strip, strip_rib_tip
from gdsfactory.types import CrossSectionOrFactory


@cell
def taper_cross_section(
    cross_section1: CrossSectionOrFactory = strip_rib_tip,
    cross_section2: CrossSectionOrFactory = rib,
    length: float = 10,
    npoints: int = 100,
    linear: bool = False,
    **kwargs
) -> Component:
    r"""Returns taper transition between cross_section1 and cross_section2

    Args:
        cross_section1: start cross_section factory
        cross_section2: end cross_section factory
        length: transition length
        npoints: number of points
        linear: shape of the transition, sine when False
        kwargs: cross_section settings for section2



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
    transition = gf.path.transition(
        cross_section1=cross_section1() if callable(cross_section1) else cross_section1,
        cross_section2=cross_section2(**kwargs)
        if callable(cross_section2)
        else cross_section2,
        width_type="linear" if linear else "sine",
    )
    taper_path = gf.path.straight(length=length, npoints=npoints)

    c = gf.Component()
    ref = c << gf.path.extrude(taper_path, transition)
    c.add_ports(ref.ports)
    return c


taper_cross_section_linear = gf.partial(taper_cross_section, linear=True, npoints=2)
taper_cross_section_sine = gf.partial(taper_cross_section, linear=False, npoints=101)


if __name__ == "__main__":
    # x1 = gf.partial(strip, width=0.5)
    # x2 = gf.partial(strip, width=2.5)
    # c = taper_cross_section_linear(x1, x2)

    x1 = gf.partial(strip, width=0.5)
    x2 = gf.partial(rib, width=2.5)
    c = taper_cross_section_linear(x1, x2)
    c.show()
