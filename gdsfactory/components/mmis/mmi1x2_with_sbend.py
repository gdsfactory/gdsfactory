__all__ = ["mmi1x2_with_sbend", "mmi_widths"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions.mmis.mmi1x2_with_sbend import mmi_widths
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import mmi_1x2_schematic


@gf.cell_with_module_name(schematic_function=mmi_1x2_schematic, tags=["mmis"])
def mmi1x2_with_sbend(
    with_sbend: bool = True,
    s_bend: ComponentSpec = "bend_s",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns 1x2 splitter for Cband.

    <https://opg.optica.org/oe/fulltext.cfm?uri=oe-21-1-1310&id=248418>

    Args:
        with_sbend: add sbend.
        s_bend: S-bend spec.
        cross_section: spec.
    """
    return cf.mmi1x2_with_sbend(
        with_sbend=with_sbend,
        s_bend=s_bend,
        cross_section=cross_section,
    )
