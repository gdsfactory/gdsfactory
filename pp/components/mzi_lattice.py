from typing import Dict, Tuple, Union

from pp.cell import cell
from pp.component import Component
from pp.tech import LIBRARY, Library
from pp.types import Number


@cell
def mzi_lattice(
    coupler_lengths: Tuple[Number, ...] = (10.0, 20.0),
    coupler_gaps: Tuple[Number, ...] = (0.2, 0.3),
    delta_lengths: Tuple[Number, ...] = (10.0,),
    mzi: Union[str, Dict] = "mzi",
    splitter: str = "coupler",
    straight: Union[str, Dict] = "straight",
    library: Library = LIBRARY,
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
    get = library.get_component

    splitter_settings = dict(
        component=splitter, gap=coupler_gaps[0], length=coupler_lengths[0]
    )
    combiner_settings = dict(
        component=splitter, gap=coupler_gaps[1], length=coupler_lengths[1]
    )

    cp1 = get(**splitter_settings)

    sprevious = c << get(
        component=mzi,
        with_splitter=True,
        delta_length=delta_lengths[0],
        straight=straight,
        combiner=combiner_settings,
        splitter=splitter_settings,
        **kwargs
    )

    stages = []

    for length, gap, delta_length in zip(
        coupler_lengths[2:], coupler_gaps[2:], delta_lengths[1:]
    ):

        splitter_settings = dict(
            component=splitter, gap=coupler_gaps[1], length=coupler_lengths[1]
        )
        combiner_settings = dict(component=splitter, length=length, gap=gap)

        stage = c << get(
            component=mzi,
            with_splitter=False,
            delta_length=delta_length,
            straight=straight,
            splitter=splitter_settings,
            combiner=combiner_settings,
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

    return c


if __name__ == "__main__":
    cpl = [10, 20, 30]
    cpg = [0.1, 0.2, 0.3]
    dl0 = [100, 200]

    # cpl = [10, 20, 30, 40]
    # cpg = [0.2, 0.3, 0.5, 0.5]
    # dl0 = [0, 50, 100]

    c = mzi_lattice(
        coupler_lengths=cpl, coupler_gaps=cpg, delta_lengths=dl0, length_x=10
    )
    c.show()
