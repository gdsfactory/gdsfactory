from typing import Callable, Optional

import pp
from pp.component import Component
from pp.components.bend_circular import bend_circular
from pp.components.coupler import coupler as coupler_function
from pp.components.taper import taper
from pp.components.waveguide import waveguide as waveguide_function


@pp.cell
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
    taper_factory: Callable = taper,
    taper_length: float = 5.0,
    bend90: Callable = bend_circular,
    waveguide_factory: Callable = waveguide_function,
    coupler1: Optional[Callable] = coupler_function,
    coupler2: Callable = coupler_function,
    pins: bool = True,
    **kwargs,
) -> Component:
    r"""Mzi tolerant to fab variations
    based on Yufei Xing thesis http://photonics.intec.ugent.be/publications/PhD.asp?ID=250

    Args:
        w0: coupler wg_width
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
        taper_factory: taper factory
        taper_length:
        bend90: bend_circular or factory
        waveguide_factory: factory
        coupler1: coupler1 or factory, can be None
        coupler2: coupler2 or factory
        kwargs: shared kwargs for all factories

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


    .. plot::
      :include-source:

      import pp

      c = pp.c.mzit()
      pp.plotgds(c)

    """
    c = pp.Component()
    cp2 = (
        c
        << coupler2(
            wg_width=w0, length=coupler_length2, gap=coupler_gap2, dy=dy, **kwargs
        )
        if callable(coupler2)
        else coupler2
    )

    # inner arm (w1)
    t1 = c << taper_factory(width1=w0, width2=w1, length=taper_length, **kwargs)
    t1.connect("1", cp2.ports["E1"])
    b1t = c << bend90(width=w1, radius=bend_radius, **kwargs)
    b1b = c << bend90(width=w1, radius=bend_radius, **kwargs)

    b1b.connect("W0", t1.ports["2"])
    b1t.connect("W0", b1b.ports["N0"])
    # wg1 = c << waveguide_factory(width=w1, length=coupler_gap2+coupler_gap1,**kwargs)
    # wg1.connect("W0", b1b.ports["N0"])
    # b1t.connect("W0", wg1.ports["E0"])

    t3b = c << taper_factory(width1=w1, width2=w2, length=taper_length, **kwargs)
    t3b.connect("1", b1t.ports["N0"])
    wgs2 = c << waveguide_factory(width=w2, length=Ls, **kwargs)
    wgs2.connect("W0", t3b.ports["2"])
    t20i = c << taper_factory(width1=w2, width2=w0, length=taper_length, **kwargs)
    t20i.connect("1", wgs2.ports["E0"])

    # outer_arm (w2)
    t2 = c << taper_factory(width1=w0, width2=w2, length=taper_length, **kwargs)
    t2.connect("1", cp2.ports["E0"])

    dx = (delta_length - 2 * dy) / 2
    assert (
        delta_length >= 4 * dy
    ), f"`delta_length`={delta_length} needs to be at least {4*dy}"

    wg2b = c << waveguide_factory(width=w2, length=dx, **kwargs)
    wg2b.connect("W0", t2.ports["2"])

    b2t = c << bend90(width=w2, radius=bend_radius, **kwargs)
    b2b = c << bend90(width=w2, radius=bend_radius, **kwargs)

    b2b.connect("W0", wg2b.ports["E0"])
    # vertical waveguide
    wg2y = c << waveguide_factory(width=w2, length=2 * dy, **kwargs)
    wg2y.connect("W0", b2b.ports["N0"])
    b2t.connect("W0", wg2y.ports["E0"])

    wg2t = c << waveguide_factory(width=w2, length=dx, **kwargs)
    wg2t.connect("W0", b2t.ports["N0"])

    t3t = c << taper_factory(width1=w2, width2=w1, length=taper_length, **kwargs)
    t3t.connect("1", wg2t.ports["E0"])
    wgs1 = c << waveguide_factory(width=w1, length=Ls, **kwargs)
    wgs1.connect("W0", t3t.ports["2"])
    t20o = c << taper_factory(width1=w1, width2=w0, length=taper_length, **kwargs)
    t20o.connect("1", wgs1.ports["E0"])

    if coupler1 is not None:
        cp1 = (
            c
            << coupler1(
                wg_width=w0, length=coupler_length1, gap=coupler_gap1, dy=dy, **kwargs
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

    if pins:
        pp.add_pins_to_references(c)
    return c


if __name__ == "__main__":
    # c = mzit(coupler1=None)
    # c = mzit(delta_length=20, layer=(2, 0))
    # c = mzit(delta_length=20, pins=True)
    c = mzit(delta_length=20, coupler_gap1=0.1, coupler_gap2=0.5)
    c = mzit(delta_length=20, coupler_gap1=0.5, coupler_gap2=0.1)
    pp.show(c)
