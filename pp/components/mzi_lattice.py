from typing import Callable, List

import pp
from pp.component import Component
from pp.components.coupler import coupler as coupler_function
from pp.components.mzi import mzi as mzi_function
from pp.components.waveguide import waveguide as waveguide_function


@pp.cell
def mzi_lattice(
    coupler_lengths: List[float] = (10, 20),
    coupler_gaps: List[float] = (0.2, 0.3),
    delta_lengths: List[float] = (10,),
    mzi_factory: Callable = mzi_function,
    splitter: Callable = coupler_function,
    waveguide: Callable = waveguide_function,
    pins: bool = False,
    **kwargs
) -> Component:
    r"""Mzi lattice filter.

    .. code::

               ______             ______
              |      |           |      |
              |      |           |      |
         cp1==|      |===cp2=====|      |=== .... ===cp_last===
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

    splitter_settings = dict(gap=coupler_gaps[0], length=coupler_lengths[0])

    combiner_settings = dict(gap=coupler_gaps[1], length=coupler_lengths[1])

    cp1 = splitter(**splitter_settings)

    sprevious = c << mzi_factory(
        splitter=splitter,
        combiner=splitter,
        with_splitter=True,
        delta_length=delta_lengths[0],
        waveguide=waveguide,
        combiner_settings=combiner_settings,
        splitter_settings=splitter_settings,
        pins=pins,
        **kwargs
    )

    stages = []

    for length, gap, delta_length in zip(
        coupler_lengths[2:], coupler_gaps[2:], delta_lengths[1:]
    ):

        splitter_settings = dict(gap=coupler_gaps[1], length=coupler_lengths[1])
        combiner_settings = dict(length=length, gap=gap)

        stage = c << mzi_factory(
            splitter=splitter,
            combiner=splitter,
            with_splitter=False,
            delta_length=delta_length,
            waveguide=waveguide,
            pins=pins,
            splitter_settings=splitter_settings,
            combiner_settings=combiner_settings,
            **kwargs
        )
        splitter_settings = combiner_settings

        stages.append(stage)

    for stage in stages:
        stage.connect("W0", sprevious.ports["E0"])
        stage.connect("W1", sprevious.ports["E1"])
        sprevious = stage

    for port in cp1.get_ports_list(prefix="W"):
        c.add_port(port.name, port=port)

    for port in sprevious.get_ports_list(prefix="E"):
        c.add_port(port.name, port=port)

    if pins:
        pp.add_pins_to_references(c)

    return c


if __name__ == "__main__":
    cpl = [10, 20, 30]
    cpg = [0.1, 0.2, 0.3]
    dl0 = [100, 200]

    # cpl = [10, 20, 30, 40]
    # cpg = [0.2, 0.3, 0.5, 0.5]
    # dl0 = [0, 50, 100]

    c = mzi_lattice(
        coupler_lengths=cpl, coupler_gaps=cpg, delta_lengths=dl0, length_x=10, pins=True
    )
    pp.show(c)
