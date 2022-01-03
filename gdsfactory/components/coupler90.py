import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight
from gdsfactory.cross_section import strip
from gdsfactory.types import ComponentFactory, CrossSectionFactory, Optional


@gf.cell
def coupler90(
    gap: float = 0.2,
    radius: float = 10.0,
    bend: ComponentFactory = bend_euler,
    cross_section: CrossSectionFactory = strip,
    bend_cross_section: Optional[CrossSectionFactory] = None,
    **kwargs
) -> Component:
    r"""straight coupled to a bend.

    Args:
        gap: um
        radius: um
        straight: for straight
        bend: for bend
        cross_section:
        kwargs: cross_section settings

    .. code::

             3
             |
            /
           /
        2_/
        1____4

    """
    c = Component()
    x = cross_section(radius=radius, **kwargs)
    bend_cross_section = bend_cross_section or cross_section

    bend90 = (
        bend(cross_section=bend_cross_section, radius=radius, **kwargs)
        if callable(bend)
        else bend
    )
    bend_ref = c << bend90
    straight_component = (
        straight(
            cross_section=cross_section,
            length=bend90.ports["o2"].midpoint[0] - bend90.ports["o1"].midpoint[0],
            **kwargs
        )
        if callable(straight)
        else straight
    )

    wg_ref = c << straight_component
    width = x.info["width"]

    pbw = bend_ref.ports["o1"]
    bend_ref.movey(pbw.midpoint[1] + gap + width)

    c.absorb(wg_ref)
    c.absorb(bend_ref)

    c.add_port("o1", port=wg_ref.ports["o1"])
    c.add_port("o4", port=wg_ref.ports["o2"])
    c.add_port("o2", port=bend_ref.ports["o1"])
    c.add_port("o3", port=bend_ref.ports["o2"])
    return c


def coupler90circular(bend: ComponentFactory = bend_circular, **kwargs):
    return coupler90(bend=bend, **kwargs)


if __name__ == "__main__":
    # c = coupler90circular(gap=0.3)
    # c << coupler90(gap=0.3)
    c = coupler90(radius=3, layer=(2, 0))
    c.show()
    c.pprint()
    # print(c.ports)
