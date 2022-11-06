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
    straight: ComponentSpec = straight,
    cross_section: CrossSectionSpec = "strip",
    bend_cross_section: Optional[CrossSectionSpec] = None,
    **kwargs
) -> Component:
    r"""Straight coupled to a bend.

    Args:
        gap: um.
        radius: um.
        straight: for straight.
        bend: bend spec.
        cross_section: cross_section spec.
        bend_cross_section: optional bend cross_section spec.
        kwargs: cross_section settings.

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
        length=bend90.ports["o2"].center[0] - bend90.ports["o1"].center[0],
        **kwargs
    )

    wg_ref = c << straight_component
    width = x.width

    pbw = bend_ref.ports["o1"]
    bend_ref.movey(pbw.center[1] + gap + width)

    c.add_ports(wg_ref.ports, prefix="wg")
    c.add_ports(bend_ref.ports, prefix="bend")
    c.auto_rename_ports()
    return c


coupler90circular = gf.partial(coupler90, bend=bend_circular)


if __name__ == "__main__":
    # c = coupler90circular(gap=0.3)
    # c << coupler90(gap=0.3)
    c = coupler90(radius=3, layer=(2, 0))
    c = coupler90(radius=10, cross_section="strip_heater_metal")
    c.show(show_ports=True)
    c.pprint()
    # print(c.ports)
