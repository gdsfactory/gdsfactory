__all__ = ["mmi2x2_with_sbend"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import mmi_2x2_schematic


@gf.cell_with_module_name(schematic_function=mmi_2x2_schematic, tags=["mmis"])
def mmi2x2_with_sbend(
    with_sbend: bool = True,
    s_bend: ComponentSpec = "bend_s",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns mmi2x2 for Cband.

    C_band 2x2MMI in 220nm thick silicon
    <https://opg.optica.org/oe/fulltext.cfm?uri=oe-25-23-28957&id=376719>

    Args:
        with_sbend: add sbend.
        s_bend: S-bend function.
        cross_section: spec.
    """
    return cf.mmi2x2_with_sbend(
        with_sbend=with_sbend,
        s_bend=s_bend,
        cross_section=cross_section,
    )
