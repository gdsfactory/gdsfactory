from __future__ import annotations

__all__ = ["ring_crow_couplers"]

from collections.abc import Sequence

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import ring_double_schematic


@gf.cell_with_module_name(schematic_function=ring_double_schematic, tags=["rings"])
def ring_crow_couplers(
    radius: Sequence[float] = (10.0,) * 3,
    bends: Sequence[ComponentSpec] = ("bend_circular",) * 3,
    ring_cross_sections: Sequence[CrossSectionSpec] = ("strip",) * 3,
    couplers: Sequence[ComponentSpec] = ("coupler",) * 4,
) -> Component:
    """Coupled ring resonators with coupler components between gaps.

    Args:
        radius: for the bend and coupler.
        bends: bend specs.
        ring_cross_sections: cross_section for the ring.
        couplers: coupling component between rings and bus.

    ```text
         --==ct==-- gap[N-1]   <------- couplers[N-1]
          |      |
          sl     sr ring[N-1]
          |      |
         --==cb==-- gap[N-2]   <------- couplers[N-2]
    ```

             .
             .
             .

    ```text
         --==ct==--
          |      |
          sl     sr lengths_y[1], ring[1]
          |      |
         --==cb==-- gap[1]
                                <------- couplers[1]
         --==ct==--
          |      |
          sl     sr lengths_y[0], ring[0]
          |      |
         --==cb==-- gap[0]      <------- couplers[0]
    ```

          length_x
    """
    return cf.ring_crow_couplers(
        radius=radius,
        bends=bends,
        ring_cross_sections=ring_cross_sections,
        couplers=couplers,
    )
