from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_s import bend_s
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def coupler_symmetric(
    bend: ComponentSpec = bend_s,
    gap: float = 0.234,
    dy: float = 4.0,
    dx: float = 10.0,
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

    w = bend_component[bend_port1_name].dwidth
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


if __name__ == "__main__":
    c = coupler_symmetric(gap=0.2)
    c.show()
    # c.pprint()

    # for dyi in [2, 3, 4, 5]:
    #     c = coupler_symmetric(gap=0.2, width=0.5, dy=dyi, dx=10.0, layer=(2, 0))
    #     print(f"dy={dyi}, min_bend_radius = {c.info['min_bend_radius']}")
