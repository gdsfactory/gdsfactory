from __future__ import annotations

__all__ = ["mmi"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import ckt_schematic


@gf.cell_with_module_name(schematic_function=ckt_schematic, tags=["mmis"])
def mmi(
    inputs: int = 1,
    outputs: int = 4,
    width: float | None = None,
    width_taper: float = 1.0,
    length_taper: float = 10.0,
    length_mmi: float = 5.5,
    width_mmi: float = 5,
    gap_input_tapers: float = 0.25,
    gap_output_tapers: float = 0.25,
    taper: ComponentSpec = "taper",
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
    input_positions: list[float] | None = None,
    output_positions: list[float] | None = None,
) -> Component:
    r"""Mxn MultiMode Interferometer (MMI).

    Args:
        inputs: number of inputs.
        outputs: number of outputs.
        width: input and output straight width. Defaults to cross_section.
        width_taper: interface between input straights and mmi region.
        length_taper: into the mmi region.
        length_mmi: in x direction.
        width_mmi: in y direction.
        gap_input_tapers: gap between input tapers from edge to edge.
        gap_output_tapers: gap between output tapers from edge to edge.
        taper: taper function.
        straight: straight function.
        cross_section: specification (CrossSection, string or dict).
        input_positions: optional positions of the inputs.
        output_positions: optional positions of the outputs.

    ```text
                   length_mmi
                    <------>
                    ________
                   |        |
                __/          \__
            o2  __            __  o3
                  \          /_ _ _ _
                  |         | _ _ _ _| gap_output_tapers
                __/          \__
            o1  __            __  o4
                  \          /
                   |________|
                 | |
                 <->
            length_taper
    ```
    """
    return cf.mmi(
        inputs=inputs,
        outputs=outputs,
        width=width,
        width_taper=width_taper,
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_input_tapers=gap_input_tapers,
        gap_output_tapers=gap_output_tapers,
        taper=taper,
        straight=straight,
        cross_section=cross_section,
        input_positions=input_positions,
        output_positions=output_positions,
    )
