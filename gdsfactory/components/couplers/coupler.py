from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Delta


@gf.cell
def coupler_symmetric(
    bend: ComponentSpec = "bend_s",
    gap: float = 0.234,
    dy: Delta = 4.0,
    dx: Delta = 10.0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""Two coupled straights with bends.

    Args:
        bend: bend spec.
        gap: in um.
        dy: port to port vertical spacing.
        dx: bend length in x direction.
        cross_section: section.

    .. code::

                       dx
                    |-----|
                       ___ o3
                      /       |
             o2 _____/        |
                              |
             o1 _____         |  dy
                     \        |
                      \___    |
                           o4

    """
    c = Component()
    x = gf.get_cross_section(cross_section)
    width = x.width
    dy = (dy - gap - width) / 2

    bend_component = gf.get_component(
        bend,
        size=(dx, dy),
        cross_section=cross_section,
    )
    top_bend = c << bend_component
    bot_bend = c << bend_component
    bend_ports = top_bend.ports.filter(port_type="optical")
    bend_port1_name = bend_ports[0].name
    bend_port2_name = bend_ports[1].name

    w = bend_component[bend_port1_name].width
    y = w + gap
    y /= 2

    bot_bend.dmirror_y()
    top_bend.dmovey(+y)
    bot_bend.dmovey(-y)

    c.add_port("o1", port=bot_bend[bend_port1_name])
    c.add_port("o2", port=top_bend[bend_port1_name])
    c.add_port("o3", port=top_bend[bend_port2_name])
    c.add_port("o4", port=bot_bend[bend_port2_name])

    c.info["length"] = bend_component.info["length"]
    c.info["min_bend_radius"] = bend_component.info["min_bend_radius"]
    return c


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
    x = gf.get_cross_section(cross_section)
    _straight = gf.c.straight(length=length, cross_section=cross_section)

    top = c << _straight
    bot = c << _straight

    w = x.width
    y = w + gap

    top.dmovey(+y)

    if bot.ports and top.ports:
        c.add_port("o1", port=bot.ports[0])
        c.add_port("o2", port=top.ports[0])
        c.add_port("o3", port=bot.ports[1])
        c.add_port("o4", port=top.ports[1])
        c.auto_rename_ports()
    return c


@gf.cell
def coupler(
    gap: float = 0.236,
    length: float = 20.0,
    dy: Delta = 4.0,
    dx: Delta = 10.0,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    bend: ComponentSpec = "bend_s",
) -> Component:
    r"""Symmetric coupler.

    Args:
        gap: between straights in um.
        length: of coupling region in um.
        dy: port to port vertical spacing in um.
        dx: length of bend in x direction in um.
        cross_section: spec (CrossSection, string or dict).
        allow_min_radius_violation: if True does not check for min bend radius.
        bend: input and output sbend components.

    .. code::

               dx                                 dx
            |------|                           |------|
         o2 ________                           ______o3
                    \                         /           |
                     \        length         /            |
                      ======================= gap         | dy
                     /                       \            |
            ________/                         \_______    |
         o1                                          o4

                        coupler_straight  coupler_symmetric
    """
    c = Component()
    sbend = coupler_symmetric(
        gap=gap, dy=dy, dx=dx, cross_section=cross_section, bend=bend
    )

    sr = c << sbend
    sl = c << sbend
    cs = c << coupler_straight(length=length, gap=gap, cross_section=cross_section)
    sl.connect("o2", other=cs.ports["o1"])
    sr.connect("o1", other=cs.ports["o4"])

    c.add_port("o1", port=sl.ports["o3"])
    c.add_port("o2", port=sl.ports["o4"])
    c.add_port("o3", port=sr.ports["o3"])
    c.add_port("o4", port=sr.ports["o4"])

    c.info["length"] = sbend.info["length"]
    c.info["min_bend_radius"] = sbend.info["min_bend_radius"]
    c.auto_rename_ports()

    x = gf.get_cross_section(cross_section)
    x.add_bbox(c)
    c.flatten()
    assert x.radius is not None
    if not allow_min_radius_violation:
        x.validate_radius(x.radius)
    return c


if __name__ == "__main__":
    c = coupler(gap=0.2, dy=100)
    n = c.get_netlist()
    c.show()
