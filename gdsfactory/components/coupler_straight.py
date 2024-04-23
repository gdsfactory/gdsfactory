from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.cross_section import CrossSectionSpec
from gdsfactory.typings import ComponentSpec


@gf.cell
def coupler_straight(
    length: float = 10.0,
    gap: float = 0.27,
    straight: ComponentSpec = straight_function,
    cross_section: CrossSectionSpec = "xs_sc",
    **kwargs,
) -> Component:
    """Coupler_straight with two parallel straights.

    Args:
        length: of straight.
        gap: between straights.
        straight: straight component (straight, bend_euler, bend_heater).
        cross_section: specification (CrossSection, string or dict).
        kwargs: cross_section settings.

    .. code::

        o2──────▲─────────o3
                │gap
        o1──────▼─────────o4
    """
    component = Component()

    straight_component = straight(length=length, cross_section=cross_section, **kwargs)

    top = component << straight_component
    bot = component << straight_component
    top.movey(straight_component.info["width"] + gap)

    component.add_port("o1", port=bot.ports["o1"])
    component.add_port("o2", port=top.ports["o1"])
    component.add_port("o3", port=bot.ports["o2"])
    component.add_port("o4", port=top.ports["o2"])
    component.auto_rename_ports()
    return component


if __name__ == "__main__":
    c = coupler_straight(length=2)
    c.show(show_ports=False)
