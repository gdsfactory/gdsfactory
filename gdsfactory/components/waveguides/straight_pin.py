"""Straight Doped PIN waveguide."""

from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section import pin
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def straight_pin(
    length: float = 500.0,
    cross_section: CrossSectionSpec = pin,
    via_stack: ComponentSpec = "via_stack_slab_m3",
    via_stack_width: float = 10.0,
    via_stack_spacing: float = 2,
    taper: ComponentSpec | None = "taper_strip_to_ridge",
) -> Component:
    """Returns rib waveguide with doping and via_stacks used for PN and PIN modulators.

    For PIN:
    https://doi.org/10.1364/OE.26.029983

    500um length for PI phase shift
    https://ieeexplore.ieee.org/document/8268112

    to go beyond 2PI, you will need at least 1mm
    https://ieeexplore.ieee.org/document/8853396/

    For PN:
    Typical lengths in practice are 2-5mm depending on doping,engineering and application:
    https://opg.optica.org/oe/fulltext.cfm?uri=oe-21-25-30350&id=275107
    https://opg.optica.org/oe/fulltext.cfm?uri=oe-20-11-12014&id=233271

    Args:
        length: of the waveguide.
        cross_section: for the waveguide.
        via_stack: for the via_stacks.
        via_stack_width: width of the via_stack.
        via_stack_spacing: spacing between via_stacks.
        taper: optional taper.
    """
    c = Component()
    if taper:
        _taper = gf.get_component(taper)
        length -= 2 * _taper.dxsize

    wg = c << gf.components.straight(
        cross_section=cross_section,
        length=length,
    )

    if taper:
        t1 = c << _taper
        t2 = c << _taper
        t1.connect("o2", wg.ports["o1"])
        t2.connect("o2", wg.ports["o2"])
        c.add_port("o1", port=t1.ports["o1"])
        c.add_port("o2", port=t2.ports["o1"])

    else:
        c.add_ports(wg.ports)

    via_stack_length = length
    _via_stack = gf.get_component(via_stack, size=(via_stack_length, via_stack_width))
    via_stack_top = c << _via_stack
    via_stack_bot = c << _via_stack
    via_stack_bot.dxmin = wg.dxmin
    via_stack_top.dxmin = wg.dxmin

    via_stack_top.dymin = +via_stack_spacing / 2
    via_stack_bot.dymax = -via_stack_spacing / 2

    c.add_ports(via_stack_bot.ports, prefix="bot_")
    c.add_ports(via_stack_top.ports, prefix="top_")
    return c


straight_pn = partial(straight_pin, cross_section="pn", length=2000)

if __name__ == "__main__":
    c = straight_pn(length=40)
    c.show()
