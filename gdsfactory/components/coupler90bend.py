from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.types import ComponentSpec, CrossSectionSpec


@gf.cell
def coupler90bend(
    radius: float = 10.0,
    gap: float = 0.2,
    angle_inner: float = 90.0,
    angle_outer: float = 90.0,
    bend: ComponentSpec = bend_euler,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
) -> Component:
    r"""Returns 2 coupled bends.

    Args:
        radius: um.
        gap: um.
        angle_inner: of the inner bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        angle_outer: of the outer bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        bend: for bend.
        cross_section_inner: spec inner bend.
        cross_section_outer: spec outer bend.


    .. code::

            r   3 4
            |   | |
            |  / /
            | / /
        2____/ /
        1_____/

    """
    c = Component()

    xi = gf.get_cross_section(cross_section_inner)
    xo = gf.get_cross_section(cross_section_outer)

    width = xo.width / 2 + xi.width / 2
    spacing = gap + width

    bend90_inner = gf.get_component(
        bend, radius=radius, cross_section=cross_section_inner, angle=angle_inner
    )
    bend90_outer = gf.get_component(
        bend,
        radius=radius + spacing,
        cross_section=cross_section_outer,
        angle=angle_outer,
    )
    bend_inner_ref = c << bend90_inner
    bend_outer_ref = c << bend90_outer

    pbw = bend_inner_ref.ports["o1"]
    bend_inner_ref.movey(pbw.center[1] + spacing)

    # This component is a leaf cell => using absorb
    c.absorb(bend_outer_ref)
    c.absorb(bend_inner_ref)

    c.add_port("o1", port=bend_outer_ref.ports["o1"])
    c.add_port("o2", port=bend_inner_ref.ports["o1"])
    c.add_port("o3", port=bend_inner_ref.ports["o2"])
    c.add_port("o4", port=bend_outer_ref.ports["o2"])
    return c


if __name__ == "__main__":

    from bend_circular import bend_circular

    c = coupler90bend(radius=3, bend=bend_circular, angle_inner=90, angle_outer=45)
    c.show(show_ports=True)
