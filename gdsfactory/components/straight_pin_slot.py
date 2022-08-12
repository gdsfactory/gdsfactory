"""Straight Doped PIN waveguide."""
from typing import Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.taper import taper_strip_to_ridge
from gdsfactory.components.via_stack import via_stack_m1_m3
from gdsfactory.components.via_stack_slot import via_stack_slot_slab_m1
from gdsfactory.cross_section import pin, pn
from gdsfactory.types import ComponentSpec, CrossSectionSpec


@gf.cell
def straight_pin_slot(
    length: float = 500.0,
    cross_section: CrossSectionSpec = pin,
    via_stack: Optional[ComponentSpec] = via_stack_m1_m3,
    via_stack_width: float = 10.0,
    via_stack_slab: Optional[ComponentSpec] = via_stack_slot_slab_m1,
    via_stack_slab_top: Optional[ComponentSpec] = None,
    via_stack_slab_bot: Optional[ComponentSpec] = None,
    via_stack_slab_width: Optional[float] = None,
    via_stack_spacing: float = 3.0,
    via_stack_slab_spacing: float = 2.0,
    taper: Optional[ComponentSpec] = taper_strip_to_ridge,
    **kwargs,
) -> Component:
    """Returns a PIN straight waveguide with slotted via.

    https://doi.org/10.1364/OE.26.029983

    500um length for PI phase shift
    https://ieeexplore.ieee.org/document/8268112

    to go beyond 2PI, you will need at least 1mm
    https://ieeexplore.ieee.org/document/8853396/

    Args:
        length: of the waveguide.
        cross_section: for the waveguide.
        via_stack: for via_stacking the metal.
        via_stack_width: in um.
        via_stack_slab: function for the component via_stacking the slab.
        via_stack_slab_top: Optional, defaults to via_stack_slab.
        via_stack_slab_bot: Optional, defaults to via_stack_slab.
        via_stack_slab_width: defaults to via_stack_width.
        via_stack_spacing: spacing between via_stacks.
        taper: optional taper.
        kwargs: cross_section settings.
    """
    c = Component()
    if taper:
        taper = gf.get_component(taper)
        length -= 2 * taper.get_ports_xsize()

    wg = c << gf.components.straight(
        cross_section=cross_section,
        length=length,
        **kwargs,
    )

    via_stack_slab_width = via_stack_slab_width or via_stack_width
    via_stack_slab_spacing = via_stack_slab_spacing or via_stack_spacing

    if taper:
        t1 = c << taper
        t2 = c << taper
        t1.connect("o2", wg.ports["o1"])
        t2.connect("o2", wg.ports["o2"])
        c.add_port("o1", port=t1.ports["o1"])
        c.add_port("o2", port=t2.ports["o1"])

    else:
        c.add_ports(wg.get_ports_list())

    via_stack_length = length

    if via_stack:
        via_stack_top = c << via_stack(
            size=(via_stack_length, via_stack_width),
        )
        via_stack_bot = c << via_stack(
            size=(via_stack_length, via_stack_width),
        )

        via_stack_bot.x = wg.x
        via_stack_top.x = wg.x

        via_stack_top.ymin = +via_stack_spacing / 2
        via_stack_bot.ymax = -via_stack_spacing / 2
        c.add_ports(via_stack_bot.ports, prefix="bot_")
        c.add_ports(via_stack_top.ports, prefix="top_")

    via_stack_slab_top = via_stack_slab_top or via_stack_slab
    via_stack_slab_bot = via_stack_slab_bot or via_stack_slab

    if via_stack_slab_top:
        slot_top = c << via_stack_slab_top(
            size=(via_stack_length, via_stack_slab_width),
        )
        slot_top.x = wg.x
        slot_top.ymin = +via_stack_slab_spacing / 2

    if via_stack_slab_bot:
        slot_bot = c << via_stack_slab_bot(
            size=(via_stack_length, via_stack_slab_width),
        )
        slot_bot.x = wg.x
        slot_bot.ymax = -via_stack_slab_spacing / 2

    return c


straight_pn_slot = gf.partial(straight_pin_slot, cross_section=pn)

if __name__ == "__main__":
    c = straight_pin_slot(via_stack_width=4, via_stack_slab_width=3, length=50)
    c.show(show_ports=True)
