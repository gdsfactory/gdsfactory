from typing import Callable, List
from pp.components.waveguide import waveguide
from pp.components.coupler import coupler
from pp.components.mzi import mzi
from pp.component import Component
import pp


@pp.cell
def mzi_lattice(
    coupler_lengths: List[float] = [10, 20],
    coupler_gaps: List[float] = [0.2, 0.3],
    delta_lengths: List[float] = [10],
    mzi_factory: Callable = mzi,
    coupler_factory: Callable = coupler,
    waveguide_factory: Callable = waveguide,
) -> Component:
    r"""Mzi lattice filter.

    .. code::

               ______             ______
              |      |           |      |
              |      |           |      |
         cp1==|      |===cp2=====|      |===cp3===
              |      |           |      |
              |      |           |      |
             DL1     |          DL2     |
              |      |           |      |
              |______|           |      |
                                 |______|

    """
    assert len(coupler_lengths) == len(coupler_gaps)
    assert len(coupler_lengths) == len(delta_lengths) + 1

    c = Component()

    cp1 = coupler_factory(gap=coupler_gaps[0], length=coupler_lengths[0])

    cp2 = coupler_factory(gap=coupler_gaps[1], length=coupler_lengths[1])

    sprevious = c << mzi_factory(
        coupler=cp1, combiner=cp2, DL=delta_lengths[0], waveguide=waveguide_factory
    )

    stages = [
        c
        << mzi_factory(
            coupler=cp1,
            combiner=coupler_factory(length=coupler_length, gap=couler_gap),
            DL=delta_length,
            with_coupler=False,
            waveguide=waveguide_factory,
        )
        for coupler_length, couler_gap, delta_length in zip(
            coupler_lengths[2:], coupler_gaps[2:], delta_lengths[1:]
        )
    ]

    for stage in stages:
        stage.connect("W0", sprevious.ports["E0"])
        stage.connect("W1", sprevious.ports["E1"])
        sprevious = stage

    for port in cp1.get_ports_list(prefix="W"):
        c.add_port(port.name, port=port)

    for port in sprevious.get_ports_list(prefix="E"):
        c.add_port(port.name, port=port)

    return c


if __name__ == "__main__":
    cpl = [10, 20, 30]
    cpg = [0.2, 0.3, 0.5]
    dl0 = [10, 20]

    # cpl = [10, 20, 30, 40]
    # cpg = [0.2, 0.3, 0.5, 0.5]
    # dl0 = [0, 50, 100]

    c = mzi_lattice(coupler_lengths=cpl, coupler_gaps=cpg, delta_lengths=dl0)
    pp.show(c)
