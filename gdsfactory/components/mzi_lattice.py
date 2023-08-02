from __future__ import annotations

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.coupler import coupler as coupler_function
from gdsfactory.components.mmi2x2 import mmi2x2 as mmi_splitter_function
from gdsfactory.components.mzi import mzi2x2_2x2 as mmi_coupler_function
from gdsfactory.components.mzi import mzi_coupler
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.typings import ComponentSpec


@cell
def mzi_lattice(
    coupler_lengths: tuple[float, ...] = (10.0, 20.0),
    coupler_gaps: tuple[float, ...] = (0.2, 0.3),
    delta_lengths: tuple[float, ...] = (10.0,),
    mzi: ComponentSpec = mzi_coupler,
    splitter: ComponentSpec = coupler_function,
    **kwargs,
) -> Component:
    r"""Mzi lattice filter.

    Args:
        coupler_lengths: list of length for each coupler.
        coupler_gaps: list of coupler gaps.
        delta_lengths: list of length differences.
        mzi: function for the mzi.
        splitter: splitter function.

    keyword Args:
        length_y: vertical length for both and top arms.
        length_x: horizontal length.
        bend: 90 degrees bend library.
        straight: straight function.
        straight_y: straight for length_y and delta_length.
        straight_x_top: top straight for length_x.
        straight_x_bot: bottom straight for length_x.
        cross_section: for routing (sxtop/sxbot to combiner).

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

    c = Component()

    splitter_settings = dict(gap=coupler_gaps[0], length=coupler_lengths[0])
    combiner_settings = dict(gap=coupler_gaps[1], length=coupler_lengths[1])

    cp1 = splitter1 = gf.get_component(splitter, **splitter_settings)
    combiner1 = gf.get_component(splitter, **combiner_settings)

    sprevious = c << gf.get_component(
        mzi,
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
        splitter1 = gf.get_component(splitter, **splitter_settings)
        combiner1 = gf.get_component(splitter, **combiner_settings)

        stage = c << gf.get_component(
            mzi,
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


@cell
def mzi_lattice_mmi(
    coupler_widths=(None, None),
    coupler_widths_tapers: tuple[float, ...] = (
        1.0,
        1.0,
    ),
    coupler_lengths_tapers: tuple[float, ...] = (
        10.0,
        10.0,
    ),
    coupler_lengths_mmis: tuple[float, ...] = (
        5.5,
        5.5,
    ),
    coupler_widths_mmis: tuple[float, ...] = (
        2.5,
        2.5,
    ),
    coupler_gaps_mmis: tuple[float, ...] = (
        0.25,
        0.25,
    ),
    taper_functions_mmis=(
        taper_function,
        taper_function,
    ),
    straight_functions_mmis=(straight_function, straight_function),
    cross_sections_mmis=("strip", "strip"),
    delta_lengths: tuple[float, ...] = (10.0,),
    mzi=mmi_coupler_function,
    splitter=mmi_splitter_function,
    **kwargs,
) -> Component:
    r"""Mzi lattice filter, with MMI couplers.

    Args:
        coupler_widths: (for each MMI coupler, list of) input and output straight width.
        coupler_widths_tapers: (for each MMI coupler, list of) interface between input straights and mmi region.
        coupler_lengths_tapers: (for each MMI coupler, list of) into the mmi region.
        coupler_lengths_mmis: (for each MMI coupler, list of) in x direction.
        coupler_widths_mmis: (for each MMI coupler, list of) in y direction.
        coupler_gaps_mmis: (for each MMI coupler, list of) (width_taper + gap between tapered wg)/2.
        taper_functions_mmis: (for each MMI coupler, list of) taper function.
        straight_functions_mmis: (for each MMI coupler, list of) straight function.
        cross_sections_mmis: (for each MMI coupler, list of) spec.
        delta_lengths: list of length differences.
        mzi: function for the mzi.
        splitter: splitter function.

    keyword Args:
        length_y: vertical length for both and top arms.
        length_x: horizontal length.
        bend: 90 degrees bend library.
        straight: straight function.
        straight_y: straight for length_y and delta_length.
        straight_x_top: top straight for length_x.
        straight_x_bot: bottom straight for length_x.
        cross_section: for routing (sxtop/sxbot to combiner).

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
    length = len(coupler_widths)
    if all(
        len(lst) != length
        for lst in [
            coupler_widths_tapers,
            coupler_lengths_tapers,
            coupler_lengths_mmis,
            coupler_widths_mmis,
            coupler_gaps_mmis,
            taper_functions_mmis,
            straight_functions_mmis,
            cross_sections_mmis,
        ]
    ):
        raise ValueError("All MMI-related argument lists must be the same length.")
    if len(coupler_widths) != len(delta_lengths) + 1:
        raise ValueError(
            f"Got {len(coupler_widths)} coupler_widths and "
            f"{len(delta_lengths)} delta_lengths. "
            "You need one more coupler_width than delta_lengths "
        )

    c = Component()

    splitter_settings = dict(
        width=coupler_widths[0],
        width_taper=coupler_widths_tapers[0],
        length_taper=coupler_lengths_tapers[0],
        length_mmi=coupler_lengths_mmis[0],
        width_mmi=coupler_widths_mmis[0],
        gap_mmi=coupler_gaps_mmis[0],
        taper=taper_functions_mmis[0],
        straight=straight_functions_mmis[0],
        cross_section=cross_sections_mmis[0],
    )
    combiner_settings = dict(
        width=coupler_widths[1],
        width_taper=coupler_widths_tapers[1],
        length_taper=coupler_lengths_tapers[1],
        length_mmi=coupler_lengths_mmis[1],
        width_mmi=coupler_widths_mmis[1],
        gap_mmi=coupler_gaps_mmis[1],
        taper=taper_functions_mmis[1],
        straight=straight_functions_mmis[1],
        cross_section=cross_sections_mmis[1],
    )

    cp1 = splitter1 = gf.get_component(splitter, **splitter_settings)
    combiner1 = gf.get_component(splitter, **combiner_settings)

    sprevious = c << gf.get_component(
        mzi,
        splitter=splitter1,
        combiner=combiner1,
        with_splitter=True,
        delta_length=delta_lengths[0],
        **kwargs,
    )
    c.add_ports(sprevious.get_ports_list(port_type="electrical"))

    stages = []

    for (
        coupler_width,
        coupler_width_taper,
        coupler_length_taper,
        coupler_length_mmi,
        coupler_width_mmi,
        coupler_gap_mmi,
        taper,
        straight,
        cross_section,
        delta_length,
    ) in zip(
        coupler_widths[2:],
        coupler_widths_tapers[2:],
        coupler_lengths_tapers[2:],
        coupler_lengths_mmis[2:],
        coupler_widths_mmis[2:],
        coupler_gaps_mmis[2:],
        taper_functions_mmis[2:],
        straight_functions_mmis[2:],
        cross_sections_mmis[2:],
        delta_lengths[1:],
    ):
        splitter_settings = dict(
            width=coupler_widths[1],
            width_taper=coupler_widths_tapers[1],
            length_taper=coupler_lengths_tapers[1],
            length_mmi=coupler_lengths_mmis[1],
            width_mmi=coupler_widths_mmis[1],
            gap_mmi=coupler_gaps_mmis[1],
            taper=taper_functions_mmis[1],
            straight=straight_functions_mmis[1],
            cross_section=cross_sections_mmis[1],
        )
        combiner_settings = dict(
            width=coupler_width,
            width_taper=coupler_width_taper,
            length_taper=coupler_length_taper,
            length_mmi=coupler_length_mmi,
            width_mmi=coupler_width_mmi,
            gap_mmi=coupler_gap_mmi,
            taper=taper,
            straight=straight,
            cross_section=cross_section,
        )
        splitter1 = gf.get_component(splitter, **splitter_settings)
        combiner1 = gf.get_component(splitter, **combiner_settings)

        stage = c << gf.get_component(
            mzi,
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

    # c = mzi_lattice(
    #     coupler_lengths=cpl, coupler_gaps=cpg, delta_lengths=dl0, length_x=1
    # )
    # c = mzi_lattice(delta_lengths=(20,))
    # c.show(show_ports=True)

    c = mzi_lattice_mmi(
        coupler_widths=(None,) * 5,
        coupler_widths_tapers=(1.0,) * 5,
        coupler_lengths_tapers=(10.0,) * 5,
        coupler_lengths_mmis=(5.5,) * 5,
        coupler_widths_mmis=(2.5,) * 5,
        coupler_gaps_mmis=(0.25,) * 5,
        taper_functions_mmis=(taper_function,) * 5,
        straight_functions_mmis=(straight_function,) * 5,
        cross_sections_mmis=("strip",) * 5,
        delta_lengths=(10.0,) * 4,
    )
    c.show(show_ports=True)
