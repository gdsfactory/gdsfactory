from typing import Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.mzit import mzit
from gdsfactory.types import ComponentFactory


@gf.cell
def mzit_lattice(
    coupler_lengths: Tuple[float, ...] = (10.0, 20.0),
    coupler_gaps: Tuple[float, ...] = (0.2, 0.3),
    delta_lengths: Tuple[float, ...] = (10.0,),
    mzi_factory: ComponentFactory = mzit,
) -> Component:
    r"""Mzi fab tolerant lattice filter.

    .. code::

                           cp1
           W3  W1 __                  __ E1___w0_t2   _w2___
                    \                /                      \
                     \    length1   /                        |
                      ============== gap1                    |
                     /              \                        |
                  __/                \_____w0___t1   _w1     |
           W2  W0                       E0               \   | .
                            ...                          |   | .
           W1  W1                                        |   | .
                  __                  __w0____t1____w1___/   |
                    \                /                       |
                     \    lengthN   /                        |
                      ============== gapN                    |
                     /               \                       |
                  __/                 \_                     |
           W0  W0                      \ E0_w0__t2 __w1_____/
                           cpN


    """
    assert len(coupler_lengths) == len(coupler_gaps)
    assert len(coupler_lengths) == len(delta_lengths) + 1
    assert len(coupler_lengths) >= 2

    c = Component()

    cp1 = coupler0 = c << mzi_factory(
        coupler_gap1=coupler_gaps[0],
        coupler_gap2=coupler_gaps[1],
        coupler_length1=coupler_lengths[0],
        coupler_length2=coupler_lengths[1],
        delta_length=delta_lengths[0],
    )

    couplers = [
        c
        << mzi_factory(
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
            coupler.mirror()
        coupler.connect("o3", coupler0.ports["o1"])
        coupler.connect("o4", coupler0.ports["o2"])
        coupler0 = coupler

    c.add_port("o1", port=coupler0.ports["o1"])
    c.add_port("o2", port=coupler0.ports["o2"])
    c.add_port("o3", port=cp1.ports["o3"])
    c.add_port("o4", port=cp1.ports["o4"])
    return c


if __name__ == "__main__":
    # cpl = [10, 20, 30]
    # cpg = [0.2, 0.3, 0.5]
    # dl0 = [10, 20]

    cpl = [10, 20, 30, 40]
    cpg = [0.2, 0.3, 0.5, 0.5]
    dl0 = [10, 20, 30]

    c = mzit_lattice(coupler_lengths=cpl, coupler_gaps=cpg, delta_lengths=dl0)
    # c = mzit_lattice()
    c.show()
