from __future__ import annotations

__all__ = ["straight_heater_doped_rib", "straight_heater_doped_strip"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Size

from .._schematic import straight_schematic


@gf.cell_with_module_name(schematic_function=straight_schematic, tags=["waveguides"])
def straight_heater_doped_rib(
    length: float = 320.0,
    nsections: int = 3,
    cross_section: CrossSectionSpec = "strip_rib_tip",
    cross_section_heater: CrossSectionSpec = "rib_heater_doped",
    via_stack: ComponentSpec | None = "via_stack_slab_npp_m3",
    via_stack_metal: ComponentSpec | None = "via_stack_m1_mtop",
    via_stack_metal_size: Size = (10.0, 10.0),
    via_stack_size: Size = (10.0, 10.0),
    taper: ComponentSpec | None = "taper_cross_section",
    heater_width: float = 2.0,
    heater_gap: float = 0.8,
    via_stack_gap: float = 0.0,
    width: float = 0.5,
    xoffset_tip1: float = 0.2,
    xoffset_tip2: float = 0.4,
) -> Component:
    r"""Returns a doped thermal phase shifter.

    dimensions from <https://doi.org/10.1364/OE.27.010456>

    Args:
        length: of the waveguide in um.
        nsections: between via_stacks.
        cross_section: for the input/output ports.
        cross_section_heater: for the heater.
        via_stack: optional function to connect the heater strip.
        via_stack_metal: function to connect the metal area.
        via_stack_metal_size: x, y size in um.
        via_stack_size: x, y size in um.
        taper: optional taper spec.
        heater_width: in um.
        heater_gap: in um.
        via_stack_gap: from edge of via_stack to waveguide.
        width: waveguide width on the ridge.
        xoffset_tip1: distance in um from input taper to via_stack.
        xoffset_tip2: distance in um from output taper to via_stack.


    ```text
                              length
        |<--------------------------------------------->|
        |              length_section                   |
        |    <--------------------------->              |
        |  length_via_stack                             |
        |    <------->                             taper|
        |    __________                   _________     |
        |   |          |                  |        |    |
        |   | via_stack|__________________|        |    |
        |   |  size    |  heater width    |        |    |
        |  /|__________|__________________|________|\   |
        | / |             heater_gap               | \  |
        |/  |______________________________________|  \ |
         \  |_______________width__________________|  /
          \ |                                      | /
           \|_____________heater_gap______________ |/
            |        |                    |        |
            |        |____heater_width____|        |
            |        |                    |        |
            |________|                    |________|
    ```

        taper         cross_section_heater



    ```text
                                   |<------width------>|
                                    ____________________ heater_gap             slab_gap
             top_via_stack         |                   |<---------->| bot_via_stack   <-->
         ___ ______________________|                   |___________________________|___
        |   |            |               undoped Si                 |              |   |
        |   |layer_heater|               intrinsic region           |layer_heater  |   |
        |___|____________|__________________________________________|______________|___|
                                                                     <------------>
                                                                      heater_width
        <------------------------------------------------------------------------------>
                                       slab_width
    ```
    """
    return cf.straight_heater_doped_rib(
        length=length,
        nsections=nsections,
        cross_section=cross_section,
        cross_section_heater=cross_section_heater,
        via_stack=via_stack,
        via_stack_metal=via_stack_metal,
        via_stack_metal_size=via_stack_metal_size,
        via_stack_size=via_stack_size,
        taper=taper,
        heater_width=heater_width,
        heater_gap=heater_gap,
        via_stack_gap=via_stack_gap,
        width=width,
        xoffset_tip1=xoffset_tip1,
        xoffset_tip2=xoffset_tip2,
    )


@gf.cell_with_module_name(schematic_function=straight_schematic, tags=["waveguides"])
def straight_heater_doped_strip(
    length: float = 320.0,
    nsections: int = 3,
    cross_section: CrossSectionSpec = "strip_heater_doped",
    cross_section_heater: CrossSectionSpec = "rib_heater_doped",
    via_stack: ComponentSpec | None = "via_stack_npp_m1",
    via_stack_metal: ComponentSpec | None = "via_stack_m1_mtop",
    via_stack_metal_size: Size = (10.0, 10.0),
    via_stack_size: Size = (10.0, 10.0),
    taper: ComponentSpec | None = "taper_cross_section",
    heater_width: float = 2.0,
    heater_gap: float = 0.8,
    via_stack_gap: float = 0.0,
    width: float = 0.5,
    xoffset_tip1: float = 0.2,
    xoffset_tip2: float = 0.4,
) -> Component:
    r"""Top view.

    Args:
        length: of the waveguide in um.
        nsections: between via_stacks.
        cross_section: for the input/output ports.
        cross_section_heater: for the heater.
        via_stack: optional function to connect the heater strip.
        via_stack_metal: function to connect the metal area.
        via_stack_metal_size: x, y size in um.
        via_stack_size: x, y size in um.
        taper: optional taper spec.
        heater_width: in um.
        heater_gap: in um.
        via_stack_gap: from edge of via_stack to waveguide.
        width: waveguide width on the ridge.
        xoffset_tip1: distance in um from input taper to via_stack.
        xoffset_tip2: distance in um from output taper to via_stack.

    ```text
                              length
          <-|--------|--------------------------------->
            |        | length_section
            |<--------------------------->
           length_via_stack
            |<------>|
            |________|_______________________________
           /|        |____________________|          |
          / |viastack|                    |via_stack |
          \ | size   |____________________|          |
           \|________|____________________|__________|
                                          |          |
                      cross_section_heater|          |
                                          |          |
                                          |          |
                                          |__________|
    ```

    cross_section

    ```text
                                  |<------width------>|
          ____________             ___________________               ______________
         |            |           |     undoped Si    |             |              |
         |layer_heater|           |  intrinsic region |<----------->| layer_heater |
         |____________|           |___________________|             |______________|
                                                                     <------------>
                                                        heater_gap     heater_width
    ```
    """
    return cf.straight_heater_doped_strip(
        length=length,
        nsections=nsections,
        cross_section=cross_section,
        cross_section_heater=cross_section_heater,
        via_stack=via_stack,
        via_stack_metal=via_stack_metal,
        via_stack_metal_size=via_stack_metal_size,
        via_stack_size=via_stack_size,
        taper=taper,
        heater_width=heater_width,
        heater_gap=heater_gap,
        via_stack_gap=via_stack_gap,
        width=width,
        xoffset_tip1=xoffset_tip1,
        xoffset_tip2=xoffset_tip2,
    )
