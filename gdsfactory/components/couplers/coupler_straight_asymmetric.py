from __future__ import annotations

__all__ = ["coupler_straight_asymmetric"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec

from .._schematic import coupler_schematic


@gf.cell_with_module_name(schematic_function=coupler_schematic, tags=["couplers"])
def coupler_straight_asymmetric(
    length: float = 10.0,
    gap: float = 0.27,
    width_top: float = 0.5,
    width_bot: float = 1,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Coupler with two parallel straights of different widths.

    Args:
        length: of straight.
        gap: between straights.
        width_top: of top straight.
        width_bot: of bottom straight.
        cross_section: cross_section spec.
    """
    return cf.coupler_straight_asymmetric(
        length=length,
        gap=gap,
        width_top=width_top,
        width_bot=width_bot,
        cross_section=cross_section,
    )
