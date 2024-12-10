from __future__ import annotations

from collections.abc import Sequence

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, Delta


@gf.cell
def mzit(
    w0: float = 0.5,
    w1: float = 0.45,
    w2: float = 0.55,
    dy: Delta = 2.0,
    delta_length: float = 10.0,
    length: float = 1.0,
    coupler_length1: float = 5.0,
    coupler_length2: float = 10.0,
    coupler_gap1: float = 0.2,
    coupler_gap2: float = 0.3,
    taper: ComponentSpec = "taper",
    taper_length: float = 5.0,
    bend90: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    coupler1: ComponentSpec | None = "coupler",
    coupler2: ComponentSpec = "coupler",
    cross_section: str = "strip",
) -> Component:
    r"""Mzi tolerant to fabrication variations.

    based on Yufei Xing thesis
    http://photonics.intec.ugent.be/publications/PhD.asp?ID=250

    Args:
        w0: input waveguide width (um).
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
        cross_section: cross_section spec.

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

    cp2 = c << gf.get_component(
        coupler2,
        length=coupler_length2,
        gap=coupler_gap2,
        dy=dy,
        cross_section=cross_section,
    )

    # inner arm (w1)
    t1 = c << gf.get_component(
        taper,
        width1=w0,
        width2=w1,
        length=taper_length,
        cross_section=cross_section,
    )
    t1.connect("o1", cp2.ports["o3"])

    b1 = gf.get_component(bend90, cross_section=cross_section, width=w1)
    b1t = c << b1
    b1b = c << b1

    b1b.connect("o1", t1.ports["o2"])
    b1t.connect("o1", b1b.ports["o2"])

    t3b = c << gf.get_component(
        taper,
        width1=w1,
        width2=w2,
        length=taper_length,
        cross_section=cross_section,
    )
    t3b.connect("o1", b1t.ports["o2"])
    wgs2 = c << gf.get_component(
        straight, length=length, cross_section=cross_section, width=w2
    )
    wgs2.connect("o1", t3b.ports["o2"])
    t20i = c << gf.get_component(
        taper,
        width1=w2,
        width2=w0,
        length=taper_length,
        cross_section=cross_section,
    )
    t20i.connect("o1", wgs2.ports["o2"])

    # outer_arm (w2)
    t2 = c << gf.get_component(
        taper,
        width1=w0,
        width2=w2,
        length=taper_length,
        cross_section=cross_section,
    )
    t2.connect("o1", cp2.ports["o4"])

    dx = (delta_length - 2 * dy) / 2
    assert (
        delta_length >= 4 * dy
    ), f"`delta_length`={delta_length} needs to be at least {4 * dy}"

    wg2b = c << gf.get_component(
        straight, length=dx, cross_section=cross_section, width=w2
    )
    wg2b.connect("o1", t2.ports["o2"])

    b2 = gf.get_component(bend90, cross_section=cross_section, width=w2)
    b2t = c << b2
    b2b = c << b2
    wy = gf.get_component(
        straight, length=2 * dy, cross_section=cross_section, width=w2
    )
    wx = gf.get_component(straight, length=dx, cross_section=cross_section, width=w2)

    b2b.connect("o1", wg2b.ports["o2"])

    # vertical straight
    wg2y = c << wy
    wg2y.connect("o1", b2b.ports["o2"])
    b2t.connect("o1", wg2y.ports["o2"])

    wg2t = c << wx
    wg2t.connect("o1", b2t.ports["o2"])

    t3t = c << gf.get_component(
        taper,
        width1=w2,
        width2=w1,
        length=taper_length,
        cross_section=cross_section,
    )
    t3t.connect("o1", wg2t.ports["o2"])
    wgs1 = c << gf.get_component(
        straight, length=length, cross_section=cross_section, width=w1
    )
    wgs1.connect("o1", t3t.ports["o2"])
    t20o = c << gf.get_component(
        taper,
        width1=w1,
        width2=w0,
        length=taper_length,
        cross_section=cross_section,
    )
    t20o.connect("o1", wgs1.ports["o2"])

    if coupler1 is not None:
        cp1 = c << gf.get_component(
            coupler1,
            length=coupler_length1,
            gap=coupler_gap1,
            dy=dy,
            cross_section=cross_section,
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


@gf.cell
def mzit_lattice(
    coupler_lengths: Sequence[float] = (10.0, 20.0),
    coupler_gaps: Sequence[float] = (0.2, 0.3),
    delta_lengths: Sequence[float] = (10.0,),
    mzi: ComponentSpec = mzit,
) -> Component:
    r"""Mzi fab tolerant lattice filter.

    .. code::

                           cp1
           o4  o2 __                  __ o3___w0_t2   _w2___
                    \                /                      \
                     \    length1   /                        |
                      ============== gap1                    |
                     /              \                        |
                  __/                \_____w0___t1   _w1     |
           o3  o1                       o4               \   | .
                            ...                          |   | .
           o2  o2                    o3                  |   | .
                  __                  _____w0___t1___w1__/   |
                    \                /                       |
                     \    lengthN   /                        |
                      ============== gapN                    |
                     /               \                       |
                  __/                 \_                     |
           o1  o1                      \___w0___t2___w1_____/
                           cpN       o4


    """
    if len(coupler_lengths) != len(coupler_gaps):
        raise ValueError(
            f"Got {len(coupler_lengths)} coupler_lengths and "
            f"{len(coupler_gaps)} coupler_gaps"
        )
    if len(coupler_lengths) != len(delta_lengths) + 1:
        raise ValueError(
            f"Got {len(coupler_lengths)} coupler_lengths and "
            f"{len(delta_lengths)} delta_lengths. "
            "You need one more coupler_length than delta_lengths "
        )

    assert len(coupler_lengths) >= 2

    c = Component()

    cp1 = coupler0 = c << gf.get_component(
        mzi,
        coupler_gap1=coupler_gaps[0],
        coupler_gap2=coupler_gaps[1],
        coupler_length1=coupler_lengths[0],
        coupler_length2=coupler_lengths[1],
        delta_length=delta_lengths[0],
    )

    couplers = [
        c
        << gf.get_component(
            mzi,
            coupler_gap2=coupler_gap,
            coupler_length2=coupler_length,
            coupler1=None,
            delta_length=delta_length,
        )
        for coupler_length, coupler_gap, delta_length in zip(
            coupler_lengths[2:], coupler_gaps[2:], delta_lengths[1:]
        )
    ]

    for i, coupler in enumerate(couplers):
        if i % 2 == 0:
            coupler.dmirror()
        coupler.connect("o3", coupler0.ports["o1"])
        coupler.connect("o4", coupler0.ports["o2"])
        coupler0 = coupler

    c.add_port("o1", port=coupler0.ports["o1"])
    c.add_port("o2", port=coupler0.ports["o2"])
    c.add_port("o3", port=cp1.ports["o3"])
    c.add_port("o4", port=cp1.ports["o4"])
    return c


if __name__ == "__main__":
    c = mzit(cross_section="rib")
    # c = mzit(coupler1=None)
    # c = mzit(delta_length=20, layer=(2, 0))
    # c = mzit(delta_length=20, cross_section="rib_bbox")
    # c = mzit(delta_length=20, coupler_gap1=0.1, coupler_gap2=0.5)
    # c = mzit(delta_length=20, coupler_gap1=0.5, coupler_gap2=0.1)
    # c = mzit(coupler_length1=200)
    c.show()
