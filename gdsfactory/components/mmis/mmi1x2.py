from __future__ import annotations

__all__ = ["mmi1x2"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import mmi_1x2_schematic


@gf.cell_with_module_name(schematic_function=mmi_1x2_schematic, tags=["mmis"])
def mmi1x2(
    width: float | None = None,
    width_taper: float = 1.0,
    length_taper: float = 10.0,
    length_mmi: float = 5.5,
    width_mmi: float = 2.5,
    gap_mmi: float = 0.25,
    taper: ComponentSpec = "taper",
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""1x2 MultiMode Interferometer (MMI).

    Args:
        width: input and output straight width. Defaults to cross_section width.
        width_taper: interface between input straights and mmi region.
        length_taper: into the mmi region.
        length_mmi: in x direction.
        width_mmi: in y direction.
        gap_mmi:  gap between tapered wg.
        taper: taper function.
        straight: straight function.
        cross_section: specification (CrossSection, string or dict).

    ```text
               length_mmi
                <------>
                ________
               |        |
               |         \__
               |          __  o2
            __/          /_ _ _ _
         o1 __          | _ _ _ _| gap_mmi
              \          \__
               |          __  o3
               |         /
               |________|
    ```

             <->
        length_taper

    """
    return cf.mmi1x2(
        width=width,
        width_taper=width_taper,
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        taper=taper,
        straight=straight,
        cross_section=cross_section,
    )
