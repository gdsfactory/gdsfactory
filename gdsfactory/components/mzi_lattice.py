from functools import partial
from typing import Tuple

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.coupler import coupler as coupler_function
from gdsfactory.components.mzi import mzi_coupler
from gdsfactory.types import ComponentFactory


@cell
def mzi_lattice(
    coupler_lengths: Tuple[float, ...] = (10.0, 20.0),
    coupler_gaps: Tuple[float, ...] = (0.2, 0.3),
    delta_lengths: Tuple[float, ...] = (10.0,),
    mzi: ComponentFactory = mzi_coupler,
    splitter: ComponentFactory = coupler_function,
    **kwargs,
) -> Component:
    r"""Mzi lattice filter.

    Args:
        coupler_lengths: list of length for each coupler
        coupler_gaps: list of coupler gaps
        delta_lengths: list of length differences
        mzi: function for the mzi
        splitter: splitter function

    keyword Args:
        length_y: vertical length for both and top arms
        length_x: horizontal length
        bend: 90 degrees bend library
        straight: straight function
        straight_y: straight for length_y and delta_length
        straight_x_top: top straight for length_x
        straight_x_bot: bottom straight for length_x
        cross_section: for routing (sxtop/sxbot to combiner)

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

    sprevious = c << mzi(
        splitter=splitter1,
        combiner=combiner1,
        with_splitter=True,
        delta_length=delta_lengths[0],
        **kwargs,
    )
    c.add_ports(sprevious.get_ports_list(port_type="electrical"))

    stages = []

    for length, gap, delta_length in zip(
        coupler_lengths[2:], coupler_gaps[2:], delta_lengths[1:]
    ):

        splitter_settings = dict(gap=coupler_gaps[1], length=coupler_lengths[1])
        combiner_settings = dict(length=length, gap=gap)
        splitter1 = partial(splitter, **splitter_settings)
        combiner1 = partial(splitter, **combiner_settings)

        stage = c << mzi(
            splitter=splitter1,
            combiner=combiner1,
            with_splitter=False,
            delta_length=delta_length,
            **kwargs,
        )
        splitter_settings = combiner_settings

        stages.append(stage)
        c.add_ports(stage.get_ports_list(port_type="electrical"))

    for stage in stages:
        stage.connect("o1", sprevious.ports["o4"])
        # stage.connect('o2', sprevious.ports['o1'])
        sprevious = stage

    for port in cp1.get_ports_list(orientation=180, port_type="optical"):
        c.add_port(port.name, port=port)

    for port in sprevious.get_ports_list(orientation=0, port_type="optical"):
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
        coupler_lengths=cpl, coupler_gaps=cpg, delta_lengths=dl0, length_x=1
    )
    c = mzi_lattice(delta_lengths=(20,), length_x=0)
    c.show()
