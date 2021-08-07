from typing import Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler import coupler as coupler_function
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper
from gdsfactory.types import ComponentFactory


@gf.cell
def mzit(
    w0: float = 0.5,
    w1: float = 0.45,
    w2: float = 0.55,
    dy: float = 2.0,
    delta_length: float = 10.0,
    Ls: float = 1.0,
    coupler_length1: float = 5.0,
    coupler_length2: float = 10.0,
    coupler_gap1: float = 0.2,
    coupler_gap2: float = 0.3,
    bend_radius: float = 10.0,
    taper_factory: ComponentFactory = taper,
    taper_length: float = 5.0,
    bend90: ComponentFactory = bend_euler,
    straight_factory: ComponentFactory = straight_function,
    coupler1: Optional[ComponentFactory] = coupler_function,
    coupler2: ComponentFactory = coupler_function,
    **kwargs,
) -> Component:
    r"""Mzi tolerant to fab variations
    based on Yufei Xing thesis http://photonics.intec.ugent.be/publications/PhD.asp?ID=250

    Args:
        w1: narrow wg_width
        w2: wide wg_width
        dy: port to port vertical spacing
        delta_length: length difference between arms
        Ls: shared length for w1 and w2
        coupler_length1: length of coupler1
        coupler_length2: length of coupler2
        coupler_gap1: coupler1
        coupler_gap2: coupler2
        bend_radius: 10.0
        taper_factory: taper library
        taper_length:
        bend90: bend_circular or library
        straight_factory: library
        coupler1: coupler1 or library, can be None
        coupler2: coupler2 or library
        kwargs: overwrites waveguide_settings

    .. code::

                           cp1
           W3  W1 __                  __ E1___w0_t2   _w2___
                    \                /                      \
                     \    length1   /                        |
                      ============== gap1                    |
                     /              \                        |
                  __/                \_____w0___t1   _w1     |
           W2  W0                       E0               \   |
                                                         |   |
           W1  W1                                        |   |
                  __                  __w0____t1____w1___/   |
                    \                /                       |
                     \    length2   /                        |
                      ============== gap2                    |
                     /               \                       |
                  __/                 \_                     |
           W0  W0                      \ E0_w0__t2 __w1_____/
                           cp2


    """
    c = gf.Component()
    cp2 = (
        c
        << coupler2(
            length=coupler_length2,
            gap=coupler_gap2,
            dy=dy,
            **kwargs,
        )
        if callable(coupler2)
        else coupler2
    )

    # inner arm (w1)
    t1 = c << taper_factory(
        width1=w0,
        width2=w1,
        length=taper_length,
        **kwargs,
    )
    t1.connect("1", cp2.ports["E1"])
    b1t = c << bend90(
        width=w1,
        radius=bend_radius,
        **kwargs,
    )
    b1b = c << bend90(
        width=w1,
        radius=bend_radius,
        **kwargs,
    )

    b1b.connect("W0", t1.ports["2"])
    b1t.connect("W0", b1b.ports["N0"])

    t3b = c << taper_factory(
        width1=w1,
        width2=w2,
        length=taper_length,
        **kwargs,
    )
    t3b.connect("1", b1t.ports["N0"])
    wgs2 = c << straight_factory(width=w2, length=Ls, **kwargs)
    wgs2.connect("W0", t3b.ports["2"])
    t20i = c << taper_factory(
        width1=w2,
        width2=w0,
        length=taper_length,
        **kwargs,
    )
    t20i.connect("1", wgs2.ports["E0"])

    # outer_arm (w2)
    t2 = c << taper_factory(
        width1=w0,
        width2=w2,
        length=taper_length,
        **kwargs,
    )
    t2.connect("1", cp2.ports["E0"])

    dx = (delta_length - 2 * dy) / 2
    assert (
        delta_length >= 4 * dy
    ), f"`delta_length`={delta_length} needs to be at least {4*dy}"

    wg2b = c << straight_factory(width=w2, length=dx, **kwargs)
    wg2b.connect("W0", t2.ports["2"])

    b2t = c << bend90(
        width=w2,
        radius=bend_radius,
        **kwargs,
    )
    b2b = c << bend90(
        width=w2,
        radius=bend_radius,
        **kwargs,
    )

    b2b.connect("W0", wg2b.ports["E0"])
    # vertical straight
    wg2y = c << straight_factory(width=w2, length=2 * dy, **kwargs)
    wg2y.connect("W0", b2b.ports["N0"])
    b2t.connect("W0", wg2y.ports["E0"])

    wg2t = c << straight_factory(width=w2, length=dx, **kwargs)
    wg2t.connect("W0", b2t.ports["N0"])

    t3t = c << taper_factory(
        width1=w2,
        width2=w1,
        length=taper_length,
    )
    t3t.connect("1", wg2t.ports["E0"])
    wgs1 = c << straight_factory(width=w1, length=Ls, **kwargs)
    wgs1.connect("W0", t3t.ports["2"])
    t20o = c << taper_factory(
        width1=w1,
        width2=w0,
        length=taper_length,
        **kwargs,
    )
    t20o.connect("1", wgs1.ports["E0"])

    if coupler1 is not None:
        cp1 = (
            c
            << coupler1(
                length=coupler_length1,
                gap=coupler_gap1,
                dy=dy,
                **kwargs,
            )
            if callable(coupler1)
            else coupler1
        )
        cp1.connect("E1", t20o.ports["2"])
        cp1.connect("E0", t20i.ports["2"])
        c.add_port("W3", port=cp1.ports["W1"])
        c.add_port("W2", port=cp1.ports["W0"])
    else:
        c.add_port("W3", port=t20o.ports["2"])
        c.add_port("W2", port=t20i.ports["2"])

    c.add_port("W1", port=cp2.ports["W1"])
    c.add_port("W0", port=cp2.ports["W0"])
    return c


if __name__ == "__main__":
    # c = mzit(coupler1=None)
    # c = mzit(delta_length=20, layer=(2, 0))
    # c = mzit(delta_length=20)
    c = mzit(delta_length=20, coupler_gap1=0.1, coupler_gap2=0.5)
    c = mzit(delta_length=20, coupler_gap1=0.5, coupler_gap2=0.1)
    c.show()
