"""Straight Ge photodetector."""
from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.taper import taper as taper_func
from gdsfactory.components.via_stack import via_stack_slab_m2, via_stack_slab_m3
from gdsfactory.cross_section import pn_ge_detector_si_contacts
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

default_taper = partial(
    taper_func, length=20.0, width1=0.5, width2=0.8, cross_section="strip"
)


@gf.cell
def ge_detector_straight_si_contacts(
    length: float = 80.0,
    cross_section: CrossSectionSpec = pn_ge_detector_si_contacts,
    via_stack: ComponentSpec | tuple[ComponentSpec, ComponentSpec] = via_stack_slab_m3,
    via_stack_width: float = 10.0,
    via_stack_spacing: float = 5.0,
    via_stack_offset: float = 0.0,
    taper: ComponentSpec | None = default_taper,
    **kwargs,
) -> Component:
    """Returns a straight Ge on Si detector with silicon contacts.

    There are no contacts on the Ge. These detectors could have lower
    dark current and sensitivity compared to those with contacts in the
    Ge. See Chen et al., "High-Responsivity Low-Voltage 28-Gb/s Ge p-i-n
    Photodetector With Silicon Contacts", Journal of Lightwave Technology 33(4), 2015.

    https://doi.org/10.1109/JLT.2014.2367134

    Args:
        length: total length of the waveguide including the tapers.
        cross_section: for the waveguide.
        via_stack: for the via_stacks. First element
        via_stack_width: width of the via_stack.
        via_stack_spacing: spacing between via_stacks.
        via_stack_offset: with respect to the detector
        taper: optional taper to transition from the input waveguide
            into the absorption region.
        kwargs: cross_section settings.
    """
    c = Component()
    if taper:
        taper = gf.get_component(taper)
        length -= 2 * taper.get_ports_xsize()

    if type(via_stack) is tuple:
        via_stack_top = via_stack[0]
        via_stack_bot = via_stack[1]
    else:
        via_stack_top = via_stack
        via_stack_bot = via_stack

    wg = c << gf.components.straight(
        cross_section=cross_section,
        length=length,
        **kwargs,
    )

    if taper:
        t1 = c << taper
        t1.connect("o2", wg.ports["o1"])
        c.add_port("o1", port=t1.ports["o1"])

    else:
        c.add_ports(wg.get_ports_list())

    via_stack_length = length
    via_stack_top = c << via_stack_top(
        size=(via_stack_length, via_stack_width),
    )
    via_stack_bot = c << via_stack_bot(
        size=(via_stack_length, via_stack_width),
    )

    via_stack_bot.xmin = wg.xmin
    via_stack_top.xmin = wg.xmin

    via_stack_top.ymin = +via_stack_spacing / 2 + via_stack_offset
    via_stack_bot.ymax = -via_stack_spacing / 2 + via_stack_offset

    c.add_ports(via_stack_bot.ports, prefix="bot_")
    c.add_ports(via_stack_top.ports, prefix="top_")
    return c


if __name__ == "__main__":
    c = ge_detector_straight_si_contacts(
        via_stack=(via_stack_slab_m3, via_stack_slab_m2), via_stack_offset=0
    )
    # print(c.ports.keys())
    c.show(show_ports=True)
