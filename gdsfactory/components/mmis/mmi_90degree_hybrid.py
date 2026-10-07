from __future__ import annotations

__all__ = ["mmi_90degree_hybrid"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import ckt_schematic


@gf.cell_with_module_name(schematic_function=ckt_schematic, tags=["mmis"])
def mmi_90degree_hybrid(
    width: float = 0.5,
    width_taper: float = 1.7,
    length_taper: float = 40.0,
    length_mmi: float = 175.0,
    width_mmi: float = 10.0,
    gap_mmi: float = 0.8,
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""90 degree hybrid based on a 4x4 MMI.

    Default values from Watanabe et al.,
    "Coherent few mode demultiplexer realized as a
    2D grating coupler array in silicon", Optics Express 28(24), 2020

    It could be interesting to consider the design in Guan et al.,
    "Compact and low loss 90° optical hybrid on a silicon-on-insulator
    platform", Optics Express 25(23), 2017

    Args:
        width: input and output straight width.
        width_taper: interface between input straights and mmi region.
        length_taper: into the mmi region.
        length_mmi: in x direction.
        width_mmi: in y direction.
        gap_mmi: (width_taper + gap between tapered wg)/2.
        straight: straight function.
        cross_section: spec.

    ```text
                   length_mmi
                    <------>
                    ________
                   |        |
                __/          \__
     signal_in  __            __  I_out1
                  \          /_ _ _ _
                  |         | _ _ _ _| gap_mmi
                  |          \__
                  |           __  Q_out1
                  |          /
                  |        |
                  |
                __/          \__
        LO_in   __            __  Q_out2
                  \          /_ _ _ _
                  |         | _ _ _ _| gap_mmi
                  |          \__
                  |           __  I_out2
                  |          /
                  | ________|
    ```


                 <->
            length_taper
    """
    return cf.mmi_90degree_hybrid(
        width=width,
        width_taper=width_taper,
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        straight=straight,
        cross_section=cross_section,
    )
