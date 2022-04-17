import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight
from gdsfactory.types import ComponentSpec, CrossSectionSpec, Optional


@gf.cell
def coupler90(
    gap: float = 0.2,
    radius: float = 10.0,
    bend: ComponentSpec = bend_euler,
    cross_section: CrossSectionSpec = "strip",
    bend_cross_section: Optional[CrossSectionSpec] = None,
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
    x = gf.get_cross_section(cross_section, radius=radius, **kwargs)
    bend_cross_section = bend_cross_section or cross_section

    bend90 = gf.get_component(
        bend, cross_section=bend_cross_section, radius=radius, **kwargs
    )
    bend_ref = c << bend90
    straight_component = gf.get_component(
        straight,
        cross_section=cross_section,
        length=bend90.ports["o2"].midpoint[0] - bend90.ports["o1"].midpoint[0],
        **kwargs
    )

    wg_ref = c << straight_component
    width = x.width

    pbw = bend_ref.ports["o1"]
    bend_ref.movey(pbw.midpoint[1] + gap + width)

    c.add_port("o1", port=wg_ref.ports["o1"])
    c.add_port("o4", port=wg_ref.ports["o2"])
    c.add_port("o2", port=bend_ref.ports["o1"])
    c.add_port("o3", port=bend_ref.ports["o2"])
    return c


coupler90circular = gf.partial(coupler90, bend=bend_circular)


if __name__ == "__main__":
    # c = coupler90circular(gap=0.3)
    # c << coupler90(gap=0.3)
    c = coupler90(radius=3, layer=(2, 0))
    c = coupler90(radius=10, cross_section="rib")
    c.show()
    c.pprint()
    # print(c.ports)
