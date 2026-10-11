from __future__ import annotations

__all__ = ["delay_snake_sbend"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import spiral_schematic


@gf.cell_with_module_name(schematic_function=spiral_schematic, tags=["spirals"])
def delay_snake_sbend(
    length: float = 100.0,
    length1: float = 0.0,
    length4: float = 0.0,
    radius: float = 5.0,
    waveguide_spacing: float = 5.0,
    bend: ComponentSpec = "bend_euler",
    sbend: ComponentSpec = "bend_s",
    sbend_xsize: float = 100.0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""Returns compact Snake with sbend in the middle.

    Input port faces west and output port faces east.

    Args:
        length: total length.
        length1: first straight section length in um.
        length4: fourth straight section length in um.
        radius: u bend radius in um.
        waveguide_spacing: waveguide pitch in um.
        bend: bend spec.
        sbend: sbend spec.
        sbend_xsize: sbend size.
        cross_section: cross_section spec.

    ```text
                         length1
         <----------------------------
               length2    spacing    |
                _______              |
               |        \            |
               |          \          | bend1 radius
               |            \sbend   |
          bend2|              \      |
               |                \    |
               |                  \__|
               |
               ---------------------->----------->
                   length3              length4
    ```

        We adjust length2 and length3
    """
    return cf.delay_snake_sbend(
        length=length,
        length1=length1,
        length4=length4,
        radius=radius,
        waveguide_spacing=waveguide_spacing,
        bend=bend,
        sbend=sbend,
        sbend_xsize=sbend_xsize,
        cross_section=cross_section,
    )


if __name__ == "__main__":
    import math

    c = delay_snake_sbend()
    print(c.info["length"])

    area = c.area(layer=(1, 0))
    # area = width * length
    length = area / 0.5
    print(length)

    assert math.isclose(c.info["length"], length, rel_tol=1e-3), (
        f"{c.info['length']} != {length}"
    )
    c.show()
