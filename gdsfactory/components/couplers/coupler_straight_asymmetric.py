from __future__ import annotations

__all__ = ["coupler_straight_asymmetric"]

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec


@gf.cell_with_module_name
def coupler_straight_asymmetric(
    length: float = 10.0,
    gap: float = 0.27,
    width_top: float = 0.5,
    width_bot: float = 1,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Coupler with two parallel straights of different widths.

    Args:
        length: of straight.
        gap: between straights.
        width_top: of top straight.
        width_bot: of bottom straight.
        cross_section: cross_section spec.
    """
    c = Component()

    xs_top = gf.get_cross_section(cross_section, width=width_top)
    xs_bot = gf.get_cross_section(cross_section, width=width_bot)

    top = c << gf.c.straight(length=length, cross_section=xs_top)
    bot = c << gf.c.straight(length=length, cross_section=xs_bot)

    dy = 0.5 * (width_top + width_bot) + gap
    top.movey(dy)
    c.add_port("o1", port=bot.ports[0])
    c.add_port("o2", port=top.ports[0])
    c.add_port("o3", port=top.ports[1])
    c.add_port("o4", port=bot.ports[1])
    c.flatten()
    return c
