from __future__ import annotations

__all__ = ["delay_snake2"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import spiral_schematic


@gf.cell_with_module_name(schematic_function=spiral_schematic, tags=["spirals"])
def delay_snake2(
    length: float = 1600.0,
    length0: float = 0.0,
    length2: float = 0.0,
    n: int = 2,
    bend180: ComponentSpec = "bend_euler180",
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
) -> Component:
    """Returns Snake with a starting straight and 180 bends.

    Input faces west output faces east.

    Args:
        length: total length.
        length0: start length.
        length2: end length.
        n: number of loops.
        bend180: ubend spec.
        cross_section: cross_section spec.
        width: width of the waveguide. If None, it will use the width of the cross_section.

       | length0 | length1 |

    ```text
                 >---------|
                           | bend180.length
       |-------------------|
       |
       |------------------->------- |
                            length2
       |   delta_length    |        |
    ```
    """
    return cf.delay_snake2(
        length=length,
        length0=length0,
        length2=length2,
        n=n,
        bend180=bend180,
        cross_section=cross_section,
        width=width,
    )
