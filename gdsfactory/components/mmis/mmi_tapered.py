from __future__ import annotations

__all__ = ["mmi_tapered"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import mmi_1x2_schematic


@gf.cell_with_module_name(schematic_function=mmi_1x2_schematic, tags=["mmis"])
def mmi_tapered(
    inputs: int = 1,
    outputs: int = 2,
    width: float | None = None,
    width_taper_in: float = 2.0,
    length_taper_in: float = 1.0,
    width_taper_out: float | None = None,
    length_taper_out: float | None = None,
    width_taper: float = 1.0,
    length_taper: float = 10.0,
    length_taper_start: float | None = None,
    length_taper_end: float | None = None,
    length_mmi: float = 5.5,
    width_mmi: float = 5,
    width_mmi_inner: float | None = None,
    gap_input_tapers: float = 0.25,
    gap_output_tapers: float = 0.25,
    taper: ComponentSpec = "taper",
    cross_section: CrossSectionSpec = "strip",
    input_positions: list[float] | None = None,
    output_positions: list[float] | None = None,
) -> Component:
    r"""Mxn MultiMode Interferometer (MMI).

    This is jut a more general version of the mmi component.
    Make sure you simulate and optimize the component before using it.

    Args:
        inputs: number of inputs.
        outputs: number of outputs.
        width: input and output straight width. Defaults to cross_section.
        width_taper_in: interface between input straights and mmi region.
        length_taper_in: into the mmi region.
        width_taper_out: interface between mmi region and output straights.
        length_taper_out: into the mmi region.
        width_taper: interface between mmi region and output straights.
        length_taper: into the mmi region.
        length_taper_start: length of the taper at the start. Defaults to length_taper.
        length_taper_end: length of the taper at the end. Defaults to length_taper.
        length_mmi: in x direction.
        width_mmi: in y direction.
        width_mmi_inner: allows adding a different width for the inner mmi region.
        gap_input_tapers: gap between input tapers from edge to edge.
        gap_output_tapers: gap between output tapers from edge to edge.
        taper: taper function.
        cross_section: specification (CrossSection, string or dict).
        input_positions: optional positions of the inputs.
        output_positions: optional positions of the outputs.

    ```text
                                       ┌───────────┐
                                       │           ├───────────────┐
                                       │           │               ├────────────┐
               width_taper             │           │               │            │
                    ▲ ┌────────────────┤           │               ├────────────┘
                    │ │                │           ├───────────────┘
        ┌───────────┼─┤                │           │
        │           │ │                │           │
        ◄───────────┼─►                │           ├───────────────┐
        └───────────┼─┐                │           │               ├─────────────┐
                    ▼ └────────────────┤           │               │             │
                      ◄───────────────►│           │               ├─────────────┘
        length_taper    length_taper_in│           ├───────────────┘ length_taper
        ◄────────────►                 └───────────┘◄────────────►  ◄────────────►
            start                                  length_taper_out      end
                                       ◄───────────►
                                        length_mmi
    ```
    """
    return cf.mmi_tapered(
        inputs=inputs,
        outputs=outputs,
        width=width,
        width_taper_in=width_taper_in,
        length_taper_in=length_taper_in,
        width_taper_out=width_taper_out,
        length_taper_out=length_taper_out,
        width_taper=width_taper,
        length_taper=length_taper,
        length_taper_start=length_taper_start,
        length_taper_end=length_taper_end,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        width_mmi_inner=width_mmi_inner,
        gap_input_tapers=gap_input_tapers,
        gap_output_tapers=gap_output_tapers,
        taper=taper,
        cross_section=cross_section,
        input_positions=input_positions,
        output_positions=output_positions,
    )
