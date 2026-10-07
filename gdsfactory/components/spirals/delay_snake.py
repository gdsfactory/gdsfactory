from __future__ import annotations

__all__ = ["delay_snake"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import spiral_schematic


@gf.cell_with_module_name(schematic_function=spiral_schematic, tags=["spirals"])
def delay_snake(
    length: float = 1600.0,
    length0: float = 0.0,
    length2: float = 0.0,
    n: int = 2,
    bend180: ComponentSpec = "bend_euler180",
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
) -> Component:
    r"""Returns Snake with a starting bend and 180 bends.

    Args:
        length: total length.
        length0: start length.
        length2: end length.
        n: number of loops.
        bend180: ubend spec.
        cross_section: cross_section spec.
        width: width of the waveguide. If None, it will use the width of the cross_section.

                 | length0   |

    ```text
                 >---------\
                            \bend180.info['length']
                            /
       |-------------------/
       |
       |------------------->------->|
                            length2
       |   delta_length    |        |
    ```


    """
    return cf.delay_snake(
        length=length,
        length0=length0,
        length2=length2,
        n=n,
        bend180=bend180,
        cross_section=cross_section,
        width=width,
    )
