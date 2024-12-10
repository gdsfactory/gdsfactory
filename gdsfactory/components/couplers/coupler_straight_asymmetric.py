from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec


@gf.cell
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
    top.dmovey(dy)
    c.add_port("o1", port=bot.ports[0])
    c.add_port("o2", port=top.ports[0])
    c.add_port("o3", port=top.ports[1])
    c.add_port("o4", port=bot.ports[1])
    c.flatten()
    return c


if __name__ == "__main__":
    # d = {"length": 7.0, "gap": 0.15, "width_top": 0.405, "width_bot": 0.9}
    # d = dict(length=10.0, gap=0.1, width_top=0.5, width_bot=1)
    # d = dict(length=10.0, gap=0.1, width_top=1.0, width_bot=0.5)
    # c = coupler_straight_asymmetric(**d)
    c = coupler_straight_asymmetric()
    c.show()
