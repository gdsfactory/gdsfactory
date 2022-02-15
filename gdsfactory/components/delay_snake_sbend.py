import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.bend_s import bend_s
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.types import ComponentFactory, Float2


@gf.cell
def delay_snake_sbend(
    length: float = 1600.0,
    waveguide_spacing: float = 5.0,
    bend: ComponentFactory = bend_euler,
    sbend: ComponentFactory = bend_s,
    sbend_size: Float2 = (100, 50),
    straight: ComponentFactory = straight_function,
    debug: bool = False,
    **kwargs
) -> Component:
    r"""Return compact Snake with sbend in the middle.
    Input port faces west and output port faces east.

    Args:
        length: total length
        waveguide_spacing: waveguide pitch
        bend:
        sbend:
        sbend_size:
        straight:
        debug: prints sbend min_bend_radius
        kwargs: cross_section settings

    .. code::

              >-------------
      waveguide_spacing    |
           |  \            |
           |    \          | bend1
           |      \sbend   |
      bend2|        \      |
           |            \_ |
           |
           ---------------->
    """
    c = Component()
    bend = bend(radius=(sbend_size[1] + waveguide_spacing) / 2, angle=180, **kwargs)
    sbend = sbend(size=sbend_size, **kwargs)
    if debug:
        print(sbend.info["min_bend_radius"])

    b1 = c << bend
    b2 = c << bend
    bs = c << sbend
    bs.mirror()

    straight_length = length - (2 * bend.info["length"] - bs.info["length"])
    straight = straight(length=straight_length / 2, **kwargs)
    s1 = c << straight
    s2 = c << straight

    b1.connect("o2", s1.ports["o2"])
    bs.connect("o2", b1.ports["o1"])
    b2.connect("o1", bs.ports["o1"])
    s2.connect("o1", b2.ports["o2"])

    return c


if __name__ == "__main__":
    cc = delay_snake_sbend(debug=True)
    cc.show()
