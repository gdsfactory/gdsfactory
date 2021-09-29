from functools import partial
from typing import Tuple

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.coupler import coupler as coupler_function
from gdsfactory.components.mzi import mzi as mzi_function
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.types import ComponentFactory


@cell
def mzi_lattice(
    coupler_lengths: Tuple[float, ...] = (10.0, 20.0),
    coupler_gaps: Tuple[float, ...] = (0.2, 0.3),
    delta_lengths: Tuple[float, ...] = (10.0,),
    mzi_factory: ComponentFactory = mzi_function,
    splitter: ComponentFactory = coupler_function,
    straight: ComponentFactory = straight_function,
    **kwargs,
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

    splitter1 = partial(splitter, **splitter_settings)
    combiner1 = partial(splitter, **combiner_settings)

    cp1 = splitter1()

    sprevious = c << mzi_factory(
        splitter=splitter1,
        combiner=combiner1,
        with_splitter=True,
        delta_length=delta_lengths[0],
        straight=straight,
        **kwargs,
    )

    stages = []

    for length, gap, delta_length in zip(
        coupler_lengths[2:], coupler_gaps[2:], delta_lengths[1:]
    ):

        splitter_settings = dict(gap=coupler_gaps[1], length=coupler_lengths[1])
        combiner_settings = dict(length=length, gap=gap)
        splitter1 = partial(splitter, **splitter_settings)
        combiner1 = partial(splitter, **combiner_settings)

        stage = c << mzi_factory(
            splitter=splitter1,
            combiner=combiner1,
            with_splitter=False,
            delta_length=delta_length,
            straight=straight,
            **kwargs,
        )
        splitter_settings = combiner_settings

        stages.append(stage)

    for stage in stages:
        stage.connect("o1", sprevious.ports["o4"])
        # stage.connect('o2', sprevious.ports['o1'])
        sprevious = stage

    for port in cp1.get_ports_list(orientation=180):
        c.add_port(port.name, port=port)

    for port in sprevious.get_ports_list(orientation=0):
        c.add_port(f"o_{port.name}", port=port)

    c.auto_rename_ports()
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
