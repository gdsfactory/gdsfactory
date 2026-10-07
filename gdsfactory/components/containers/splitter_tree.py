"""Returns a switch_tree.

```text
          __
        _|  |_
  __   | |  |_   _
 |  |__| |__|    |
_|  |__          |dy
 |__|  |  __     |
       |_|  |_   |
         |  |_   -
         |__|
```

   |<-dx->|

"""

from __future__ import annotations

__all__ = ["mzi1x2_2x2_heater", "splitter_tree", "switch_tree"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Spacing

from ..mzis import mzi1x2_2x2


@gf.cell_with_module_name(tags=["containers"])
def splitter_tree(
    coupler: ComponentSpec = "mmi1x2",
    noutputs: int = 4,
    spacing: Spacing = (90.0, 50.0),
    bend_s: ComponentSpec | None = "bend_s",
    bend_s_xsize: float | None = None,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """Tree of power splitters.

    Args:
        coupler: coupler factory.
        noutputs: number of outputs.
        spacing: x, y spacing between couplers.
        bend_s: Sbend function for termination.
        bend_s_xsize: xsize for the sbend.
        cross_section: cross_section.

             __|
          __|  |__
        _|  |__
         |__        dy

          dx
    """
    return cf.splitter_tree(
        coupler=coupler,
        noutputs=noutputs,
        spacing=spacing,
        bend_s=bend_s,
        bend_s_xsize=bend_s_xsize,
        cross_section=cross_section,
    )


mzi1x2_2x2_heater = CellAlias(
    mzi1x2_2x2,
    combiner="mmi2x2",
    delta_length=0,
    straight_x_top="straight_heater_metal",
    length_x=None,
)

switch_tree = CellAlias(
    splitter_tree,
    coupler="mzi1x2_2x2_heater",
    spacing=(500, 100),
)
