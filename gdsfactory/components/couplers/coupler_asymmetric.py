from __future__ import annotations

__all__ = ["coupler_asymmetric"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, Delta

from .._schematic import mmi_1x2_schematic


@gf.cell_with_module_name(schematic_function=mmi_1x2_schematic, tags=["couplers"])
def coupler_asymmetric(
    gap: float = 0.234,
    dy: Delta = 2.5,
    dx: Delta = 10.0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Bend coupled to straight waveguide.

    Args:
        gap: um.
        dy: port to port vertical spacing.
        dx: bend length in x direction.
        cross_section: spec.

    ```text
                        dx
                     |-----|
                      _____ o2
                     /         |
               _____/          |
         gap o1____________    |  dy
                            o3
    ```
    """
    return cf.coupler_asymmetric(
        gap=gap,
        dy=dy,
        dx=dx,
        cross_section=cross_section,
    )
