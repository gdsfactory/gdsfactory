from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name
def coupler90bend(
    radius: float = 10.0,
    gap: float = 0.2,
    bend: ComponentSpec = "bend_euler",
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
) -> Component:
    r"""Returns 2 coupled bends.

    Args:
        radius: um.
        gap: um.
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
        bend, radius=radius, cross_section=cross_section_inner
    )
    bend90_outer = gf.get_component(
        bend, radius=radius + spacing, cross_section=cross_section_outer
    )
    bend_inner_ref = c << bend90_inner
    bend_outer_ref = c << bend90_outer

    pbw = bend_inner_ref["o1"]
    bend_inner_ref.movey(pbw.center[1] + spacing)

    c.add_port("o1", port=bend_outer_ref["o1"])
    c.add_port("o2", port=bend_inner_ref["o1"])
    c.add_port("o3", port=bend_inner_ref["o2"])
    c.add_port("o4", port=bend_outer_ref["o2"])
    c.flatten()
    return c


if __name__ == "__main__":
    c = coupler90bend()
    c.show()
