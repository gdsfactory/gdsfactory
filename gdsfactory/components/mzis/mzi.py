from __future__ import annotations

__all__ = [
    "mzi",
    "mzi1x2",
    "mzi1x2_2x2",
    "mzi2x2_2x2",
    "mzi2x2_2x2_phase_shifter",
    "mzi_coupler",
    "mzi_phase_shifter",
    "mzi_phase_shifter_top_heater_metal",
    "mzi_pin",
    "mzm",
]

from functools import partial

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import mzi_2x2_schematic


@gf.cell_with_module_name(schematic_function=mzi_2x2_schematic, tags=["mzis"])
def mzi(
    delta_length: float = 10.0,
    length_y: float = 2.0,
    length_x: float | None = 0.1,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    straight_y: ComponentSpec | None = None,
    straight_x_top: ComponentSpec | None = None,
    straight_x_bot: ComponentSpec | None = None,
    splitter: ComponentSpec = "mmi1x2",
    combiner: ComponentSpec | None = None,
    with_splitter: bool = True,
    port_e1_splitter: str = "o2",
    port_e0_splitter: str = "o3",
    port_e1_combiner: str = "o2",
    port_e0_combiner: str = "o3",
    port1: str = "o1",
    port2: str = "o2",
    nbends: int = 2,
    cross_section: CrossSectionSpec = "strip",
    cross_section_x_top: CrossSectionSpec | None = None,
    cross_section_x_bot: CrossSectionSpec | None = None,
    mirror_bot: bool = False,
    add_optical_ports_arms: bool = False,
    min_length: float = 10e-3,
    auto_rename_ports: bool = True,
    auto_detect_port_names: bool = False,
) -> Component:
    r"""Mzi.

    Args:
        delta_length: bottom arm vertical extra length.
        length_y: vertical length for both and top arms.
        length_x: horizontal length. None uses to the straight_x_bot/top defaults.
        bend: 90 degrees bend library.
        straight: straight function.
        straight_y: straight for length_y and delta_length.
        straight_x_top: top straight for length_x.
        straight_x_bot: bottom straight for length_x.
        splitter: splitter function.
        combiner: combiner function.
        with_splitter: if False removes splitter.
        port_e1_splitter: east top splitter port.
        port_e0_splitter: east bot splitter port.
        port_e1_combiner: east top combiner port.
        port_e0_combiner: east bot combiner port.
        port1: input port name.
        port2: output port name.
        nbends: from straight top/bot to combiner (at least 2).
        cross_section: for routing (sxtop/sxbot to combiner).
        cross_section_x_top: optional top cross_section (defaults to cross_section).
        cross_section_x_bot: optional bottom cross_section (defaults to cross_section).
        mirror_bot: if true, mirrors the bottom arm.
        add_optical_ports_arms: add all other optical ports in the arms
            with top\\_ and bot\\_ prefix.
        min_length: minimum length for the straight.
        auto_rename_ports: if True, renames ports.
        auto_detect_port_names: whether to auto detect ports names. Ignores port_e* arguments if True.

    ```text
                       b2______b3
                      |  sxtop  |
              straight_y        |
                      |         |
                      b1        b4
            splitter==|         |==combiner
                      b5        b8
                      |         |
              straight_y        |
                      |         |
        delta_length/2          |
                      |         |
                     b6__sxbot__b7
                          Lx
    ```
    """
    return cf.mzi(
        delta_length=delta_length,
        length_y=length_y,
        length_x=length_x,
        bend=bend,
        straight=straight,
        straight_y=straight_y,
        straight_x_top=straight_x_top,
        straight_x_bot=straight_x_bot,
        splitter=splitter,
        combiner=combiner,
        with_splitter=with_splitter,
        port_e1_splitter=port_e1_splitter,
        port_e0_splitter=port_e0_splitter,
        port_e1_combiner=port_e1_combiner,
        port_e0_combiner=port_e0_combiner,
        port1=port1,
        port2=port2,
        nbends=nbends,
        cross_section=cross_section,
        cross_section_x_top=cross_section_x_top,
        cross_section_x_bot=cross_section_x_bot,
        mirror_bot=mirror_bot,
        add_optical_ports_arms=add_optical_ports_arms,
        min_length=min_length,
        auto_rename_ports=auto_rename_ports,
        auto_detect_port_names=auto_detect_port_names,
    )


mzi1x2 = partial(mzi, splitter="mmi1x2", combiner="mmi1x2")
mzi2x2_2x2 = partial(
    mzi,
    splitter="mmi2x2",
    combiner="mmi2x2",
    port_e1_splitter="o3",
    port_e0_splitter="o4",
    port_e1_combiner="o3",
    port_e0_combiner="o4",
    length_x=None,
)

mzi1x2_2x2 = partial(
    mzi,
    combiner="mmi2x2",
    port_e1_combiner="o3",
    port_e0_combiner="o4",
)

mzi_coupler = partial(
    mzi2x2_2x2,
    splitter="coupler",
    combiner="coupler",
)

mzi_pin = partial(
    mzi,
    straight_x_top="straight_pin",
    cross_section_x_top="pin",
    delta_length=0.0,
    length_x=100,
)

mzi_phase_shifter = partial(mzi, straight_x_top="straight_heater_metal", length_x=200)

mzi2x2_2x2_phase_shifter = partial(
    mzi2x2_2x2, straight_x_top="straight_heater_metal", length_x=200
)

mzi_phase_shifter_top_heater_metal = partial(
    mzi_phase_shifter, straight_x_top="straight_heater_metal"
)

mzm = partial(
    mzi_phase_shifter, straight_x_top="straight_pin", straight_x_bot="straight_pin"
)
