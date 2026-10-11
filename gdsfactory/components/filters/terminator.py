from __future__ import annotations

__all__ = ["terminator"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.cross_section import strip
from gdsfactory.typings import CrossSectionSpec, LayerSpecs

from .._schematic import terminator_schematic


@gf.cell_with_module_name(schematic_function=terminator_schematic, tags=["filters"])
def terminator(
    length: float | None = 50,
    cross_section_input: CrossSectionSpec = strip,
    cross_section_tip: CrossSectionSpec | None = None,
    tapered_width: float = 0.2,
    doping_layers: LayerSpecs = ("NPP",),
    doping_offset: float = 1.0,
) -> gf.Component:
    """Returns doped taper to terminate waveguides.

    Args:
        length: distance between input and narrow tapered end.
        cross_section_input: input cross-section.
        cross_section_tip: cross-section at the end of the termination.
        tapered_width: width of the default cross-section at the end of the termination.
            Only used if cross_section_tip is not None.
        doping_layers: doping layers to superimpose on the taper. Default N++.
        doping_offset: offset of the doping layer beyond the bbox
    """
    return cf.terminator(
        length=length,
        cross_section_input=cross_section_input,
        cross_section_tip=cross_section_tip,
        tapered_width=tapered_width,
        doping_layers=doping_layers,
        doping_offset=doping_offset,
    )
