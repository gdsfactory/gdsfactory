from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.components.via_stack import via_stack_m1_m3 as via_stack_metal_function
from gdsfactory.components.via_stack import via_stack_slab_npp_m3
from gdsfactory.cross_section import rib_heater_doped, strip_rib_tip
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def straight_heater_doped_rib(
    length: float = 320.0,
    nsections: int = 3,
    cross_section: CrossSectionSpec = strip_rib_tip,
    cross_section_heater: CrossSectionSpec = rib_heater_doped,
    via_stack: ComponentSpec | None = via_stack_slab_npp_m3,
    via_stack_metal: ComponentSpec | None = via_stack_metal_function,
    via_stack_metal_size: tuple[float, float] = (10.0, 10.0),
    via_stack_size: tuple[float, float] = (10.0, 10.0),
    taper: ComponentSpec | None = taper_cross_section,
    with_taper1: bool = True,
    with_taper2: bool = True,
    heater_width: float = 2.0,
    heater_gap: float = 0.8,
    via_stack_gap: float = 0.0,
    width: float = 0.5,
    xoffset_tip1: float = 0.2,
    xoffset_tip2: float = 0.4,
    **kwargs,
) -> Component:
    r"""Returns a doped thermal phase shifter.

    dimensions from https://doi.org/10.1364/OE.27.010456

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
        kwargs: cross_section settings.

    .. code::

                              length
        |<--------------------------------------------->|
        |              length_section                   |
        |    <--------------------------->              |
        |  length_via_stack                               |
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

        taper         cross_section_heater



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

    """
    c = Component()
    cross_section_heater = partial(
        cross_section_heater,
        heater_width=heater_width,
        heater_gap=heater_gap,
        width=width,
        **kwargs,
    )

    if taper:
        taper = (
            taper(cross_section1=cross_section, cross_section2=cross_section_heater)
            if callable(taper)
            else taper
        )
        length -= taper.get_ports_xsize() * 2

    wg = c << gf.c.straight(
        cross_section=cross_section_heater,
        length=snap_to_grid(length),
    )

    if taper:
        if with_taper1:
            taper1 = c << taper
            taper1.connect("o2", wg.ports["o1"])
            c.add_port("o1", port=taper1.ports["o1"])
        else:
            c.add_port("o1", port=wg.ports["o1"])

        if with_taper2:
            taper2 = c << taper
            taper2.mirror()
            taper2.connect("o2", wg.ports["o2"])
            c.add_port("o2", port=taper2.ports["o1"])

        else:
            c.add_port("o2", port=wg.ports["o2"])
    else:
        c.add_port("o2", port=wg.ports["o2"])
        c.add_port("o1", port=wg.ports["o1"])

    if via_stack_metal:
        via_stack_section = via_stack_metal(size=via_stack_metal_size)

    via_stacks = []
    length_via_stack = snap_to_grid(via_stack_size[1])
    length_section = snap_to_grid((length - length_via_stack) / nsections)
    x0 = via_stack_size[0] / 2 - xoffset_tip1

    for i in range(nsections + 1):
        xi = x0 + length_section * i

        if via_stack_metal and via_stack:
            via_stack_center = c.add_ref(via_stack_section)
            via_stack_center.x = xi
            via_stack_ref = c << via_stack_section
            via_stack_ref.x = xi
            via_stack_ref.y = (
                +via_stack_metal_size[1] if i % 2 == 0 else -via_stack_metal_size[1]
            )
            via_stacks.append(via_stack_ref)

        if via_stack:
            via_stack_top = c << via_stack(size=via_stack_size)
            via_stack_top.x = xi
            via_stack_top.ymin = +(heater_gap + width / 2 + via_stack_gap)

            via_stack_bot = c << via_stack(size=via_stack_size)
            via_stack_bot.x = xi
            via_stack_bot.ymax = -(heater_gap + width / 2 + via_stack_gap)

    if via_stack:
        via_stack.xmax = x0 + length_section * nsections
        via_stack_top.movex(xoffset_tip2)
    if via_stack:
        via_stack.xmax = x0 + length_section * nsections
        via_stack_bot.movex(xoffset_tip2)

    if via_stack_metal and via_stack:
        via_stack_length = length + via_stack_metal_size[0]
        via_stack_top = c << via_stack_metal(
            size=(via_stack_length, via_stack_metal_size[0]),
        )
        via_stack_bot = c << via_stack_metal(
            size=(via_stack_length, via_stack_metal_size[0]),
        )

        via_stack_bot.xmin = via_stacks[0].xmin
        via_stack_top.xmin = via_stacks[0].xmin

        via_stack_top.ymin = via_stacks[0].ymax
        via_stack_bot.ymax = via_stacks[1].ymin

        c.add_ports(via_stack_top.ports, prefix="top_")
        c.add_ports(via_stack_bot.ports, prefix="bot_")
    return c


def test_straight_heater_doped_rib_ports() -> None:
    c = straight_heater_doped_rib(length=100.0)
    assert c.get_ports_xsize(port_type="optical") == 100.0, c.get_ports_xsize(
        port_type="optical"
    )


if __name__ == "__main__":
    c = straight_heater_doped_rib(xoffset_tip1=10)
    # c = straight_heater_doped_rib(with_taper1=False)
    # c = straight_heater_doped_rib(length=500)
    c.show(show_ports=True)
