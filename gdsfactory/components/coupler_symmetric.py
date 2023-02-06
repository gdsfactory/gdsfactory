from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_s import bend_s
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def coupler_symmetric(
    bend: ComponentSpec = bend_s,
    gap: float = 0.234,
    dy: float = 5.0,
    dx: float = 10.0,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    r"""Two coupled straights with bends.

    Args:
        bend: bend spec.
        gap: in um.
        dy: port to port vertical spacing.
        dx: bend length in x direction.
        cross_section: section.
        **kwargs: cross_section settings.

    .. code::

                       dx
                    |-----|
                       ___ E1
                      /       |
                _____/        |
           gap  _____         |  dy
                     \        |
                      \___    |
                           E0

    """
    x = gf.get_cross_section(cross_section, **kwargs)
    width = x.width
    bend_component = gf.get_component(
        bend,
        size=(dx, (dy - gap - width) / 2),
        cross_section=cross_section,
        **kwargs,
    )

    w = bend_component.ports["o1"].width
    y = (w + gap) / 2

    c = Component()
    top_bend = bend_component.ref(position=(0, y), port_id="o1")
    bottom_bend = bend_component.ref(position=(0, -y), port_id="o1", v_mirror=True)

    c.add(top_bend)
    c.add(bottom_bend)

    c.absorb(top_bend)
    c.absorb(bottom_bend)

    c.add_port("o1", port=bottom_bend.ports["o1"])
    c.add_port("o2", port=top_bend.ports["o1"])

    c.add_port("o3", port=top_bend.ports["o2"])
    c.add_port("o4", port=bottom_bend.ports["o2"])
    c.info["length"] = bend_component.info["length"]
    c.info["min_bend_radius"] = bend_component.info["min_bend_radius"]
    return c


if __name__ == "__main__":
    c = coupler_symmetric(gap=0.2, width=0.9)
    c.show(show_ports=False)
    # c.pprint()

    # for dyi in [2, 3, 4, 5]:
    #     c = coupler_symmetric(gap=0.2, width=0.5, dy=dyi, dx=10.0, layer=(2, 0))
    #     print(f"dy={dyi}, min_bend_radius = {c.info['min_bend_radius']}")
