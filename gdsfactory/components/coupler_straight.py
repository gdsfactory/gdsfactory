from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.straight import straight
from gdsfactory.cross_section import CrossSectionSpec


@gf.cell
def coupler_straight(
    length: float = 10.0,
    gap: float = 0.27,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Coupler_straight with two parallel straights.

    Args:
        length: of straight.
        gap: between straights.
        cross_section: specification (CrossSection, string or dict).

    .. code::

        o2──────▲─────────o3
                │gap
        o1──────▼─────────o4
    """
    c = Component()
    _straight = straight(length=length, cross_section=cross_section)

    top = c << _straight
    bot = c << _straight

    w = _straight.ports["o1"].dwidth
    y = w + gap

    top.dmovey(+y)

    c.add_port("o1", port=bot.ports["o1"])
    c.add_port("o2", port=top.ports["o1"])
    c.add_port("o3", port=bot.ports["o2"])
    c.add_port("o4", port=top.ports["o2"])
    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = coupler_straight(length=2)
    c.show()
