from __future__ import annotations

from typing import Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler import coupler as coupler_function
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.typings import ComponentSpec


@gf.cell
def mzit(
    w0: float = 0.5,
    w1: float = 0.45,
    w2: float = 0.55,
    dy: float = 2.0,
    delta_length: float = 10.0,
    length: float = 1.0,
    coupler_length1: float = 5.0,
    coupler_length2: float = 10.0,
    coupler_gap1: float = 0.2,
    coupler_gap2: float = 0.3,
    taper: ComponentSpec = taper_function,
    taper_length: float = 5.0,
    bend90: ComponentSpec = bend_euler,
    straight: ComponentSpec = straight_function,
    coupler1: Optional[ComponentSpec] = coupler_function,
    coupler2: ComponentSpec = coupler_function,
    **kwargs,
) -> Component:
    r"""Mzi tolerant to fabrication variations.

    based on Yufei Xing thesis
    http://photonics.intec.ugent.be/publications/PhD.asp?ID=250

    Args:
        w1: narrow waveguide width (um).
        w2: wide waveguide width (um).
        dy: port to port vertical spacing.
        delta_length: length difference between arms (um).
        length: shared length for w1 and w2.
        coupler_length1: length of coupler1.
        coupler_length2: length of coupler2.
        coupler_gap1: coupler1.
        coupler_gap2: coupler2.
        taper: taper spec.
        taper_length: from w0 to w1.
        bend90: bend spec.
        straight: spec.
        coupler1: coupler1 spec (optional).
        coupler2: coupler2 spec.
        kwargs: cross_section settings.

    .. code::

                           cp1
            4   2 __                  __  3___w0_t2   _w2___
                    \                /                      \
                     \    length1   /                        |
                      ============== gap1                    |
                     /              \                        |
                  __/                \_____w0___t1   _w1     |
            3   1                        4               \   |
                                                         |   |
            2   2                                        |   |
                  __                  __w0____t1____w1___/   |
                    \                /                       |
                     \    length2   /                        |
                      ============== gap2                    |
                     /               \                       |                       |
                  __/                 \ E0_w0__t2 __w1______/
            1   1
                           cp2


    """
    c = gf.Component()
    cp2 = c << coupler2(
        length=coupler_length2,
        gap=coupler_gap2,
        dy=dy,
        **kwargs,
    )

    # inner arm (w1)
    t1 = c << gf.get_component(
        taper,
        width1=w0,
        width2=w1,
        length=taper_length,
        **kwargs,
    )
    t1.connect("o1", cp2.ports["o3"])
    b1t = c << gf.get_component(
        bend90,
        width=w1,
        **kwargs,
    )
    b1b = c << gf.get_component(
        bend90,
        width=w1,
        **kwargs,
    )

    b1b.connect("o1", t1.ports["o2"])
    b1t.connect("o1", b1b.ports["o2"])

    t3b = c << gf.get_component(
        taper,
        width1=w1,
        width2=w2,
        length=taper_length,
        **kwargs,
    )
    t3b.connect("o1", b1t.ports["o2"])
    wgs2 = c << gf.get_component(straight, width=w2, length=length, **kwargs)
    wgs2.connect("o1", t3b.ports["o2"])
    t20i = c << gf.get_component(
        taper,
        width1=w2,
        width2=w0,
        length=taper_length,
        **kwargs,
    )
    t20i.connect("o1", wgs2.ports["o2"])

    # outer_arm (w2)
    t2 = c << gf.get_component(
        taper,
        width1=w0,
        width2=w2,
        length=taper_length,
        **kwargs,
    )
    t2.connect("o1", cp2.ports["o4"])

    dx = (delta_length - 2 * dy) / 2
    assert (
        delta_length >= 4 * dy
    ), f"`delta_length`={delta_length} needs to be at least {4*dy}"

    wg2b = c << gf.get_component(straight, width=w2, length=dx, **kwargs)
    wg2b.connect("o1", t2.ports["o2"])

    b2t = c << gf.get_component(
        bend90,
        width=w2,
        **kwargs,
    )
    b2b = c << gf.get_component(
        bend90,
        width=w2,
        **kwargs,
    )

    b2b.connect("o1", wg2b.ports["o2"])
    # vertical straight
    wg2y = c << gf.get_component(straight, width=w2, length=2 * dy, **kwargs)
    wg2y.connect("o1", b2b.ports["o2"])
    b2t.connect("o1", wg2y.ports["o2"])

    wg2t = c << gf.get_component(straight, width=w2, length=dx, **kwargs)
    wg2t.connect("o1", b2t.ports["o2"])

    t3t = c << gf.get_component(
        taper,
        width1=w2,
        width2=w1,
        length=taper_length,
        **kwargs,
    )
    t3t.connect("o1", wg2t.ports["o2"])
    wgs1 = c << gf.get_component(straight, width=w1, length=length, **kwargs)
    wgs1.connect("o1", t3t.ports["o2"])
    t20o = c << gf.get_component(
        taper,
        width1=w1,
        width2=w0,
        length=taper_length,
        **kwargs,
    )
    t20o.connect("o1", wgs1.ports["o2"])

    if coupler1 is not None:
        cp1 = c << coupler1(
            length=coupler_length1,
            gap=coupler_gap1,
            dy=dy,
            **kwargs,
        )

        cp1.connect("o3", t20o.ports["o2"])
        cp1.connect("o4", t20i.ports["o2"])
        c.add_port("W3", port=cp1.ports["o2"])
        c.add_port("W2", port=cp1.ports["o1"])
    else:
        c.add_port("W3", port=t20o.ports["o2"])
        c.add_port("W2", port=t20i.ports["o2"])

    c.add_port("o2", port=cp2.ports["o2"])
    c.add_port("o1", port=cp2.ports["o1"])
    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    # c = mzit(coupler1=None)
    # c = mzit(delta_length=20, layer=(2, 0))
    c = mzit(delta_length=20)
    # c = mzit(delta_length=20, coupler_gap1=0.1, coupler_gap2=0.5)
    # c = mzit(delta_length=20, coupler_gap1=0.5, coupler_gap2=0.1)
    # c = mzit(coupler_length1=200)
    c.show(show_ports=True)
