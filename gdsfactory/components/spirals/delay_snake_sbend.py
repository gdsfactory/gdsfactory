from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.waveguides.straight import straight
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

diagram = r"""

                         length1
         <----------------------------
               length2    spacing    |
                _______              |
               |        \            |
               |          \          | bend1 radius
               |            \sbend   |
          bend2|              \      |
               |                \    |
               |                  \__|
               |
               ---------------------->----------->
                   length3              length4
"""


@gf.cell_with_module_name
def delay_snake_sbend(
    length: float = 100.0,
    length1: float = 0.0,
    length4: float = 0.0,
    radius: float = 5.0,
    waveguide_spacing: float = 5.0,
    bend: ComponentSpec = "bend_euler",
    sbend: ComponentSpec = "bend_s",
    sbend_xsize: float = 100.0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""Returns compact Snake with sbend in the middle.

    Input port faces west and output port faces east.

    Args:
        length: total length.
        length1: first straight section length in um.
        length4: fourth straight section length in um.
        radius: u bend radius in um.
        waveguide_spacing: waveguide pitch in um.
        bend: bend spec.
        sbend: sbend spec.
        sbend_xsize: sbend size.
        cross_section: cross_section spec.

    .. code::

                         length1
         <----------------------------
               length2    spacing    |
                _______              |
               |        \            |
               |          \          | bend1 radius
               |            \sbend   |
          bend2|              \      |
               |                \    |
               |                  \__|
               |
               ---------------------->----------->
                   length3              length4

        We adjust length2 and length3
    """
    c = Component()

    bend180_radius = (radius + waveguide_spacing) / 2
    bend = gf.get_component(
        bend,
        radius=bend180_radius,
        angle=180,
        cross_section=cross_section,
    )
    sbend = gf.get_component(
        sbend,
        size=(sbend_xsize, radius),
        cross_section=cross_section,
    )

    b1 = c << bend
    b2 = c << bend
    bs = c << sbend
    bs.dmirror()

    length23 = (
        length - (2 * bend.info["length"] - sbend.info["length"]) - length1 - length4
    )
    length2 = length23 / 2
    length3 = length23 / 2

    if length2 < 0:
        raise ValueError(
            f"length2 = {length2} < 0. You need to reduce length1 = {length1} "
            f"or length3 = {length3} or increase length = {length}\n" + diagram
        )

    straight1 = straight(length=length1, cross_section=cross_section)
    straight2 = straight(length=length2, cross_section=cross_section)
    straight3 = straight(length=length3, cross_section=cross_section)
    straight4 = straight(length=length4, cross_section=cross_section)

    # sequence = ["s1", "b1", "bs", "s2", "b2", "s3", "s4"]
    # for i_straight, component in enumerate(straight1, straight2, straight3, straight4):
    #     inst_name = f"s{i_straight+1}"
    #     if component.settings["length"] != 0:
    #         sequence.remove()
    s1 = c.add_ref(straight1, "s1")
    s2 = c.add_ref(straight2, "s2")
    s3 = c.add_ref(straight3, "s3")
    s4 = c.add_ref(straight4, "s4")
    # for inst_name in sequence:

    b1.connect("o2", s1.ports["o2"])
    bs.connect("o2", b1.ports["o1"])

    s2.connect("o2", bs.ports["o1"])

    b2.connect("o1", s2.ports["o1"])
    s3.connect("o1", b2.ports["o2"])
    s4.connect("o1", s3.ports["o2"])

    c.add_port("o1", port=s1.ports["o1"])
    c.add_port("o2", port=s4.ports["o2"])

    c.info["min_bend_radius"] = float(sbend.info["min_bend_radius"])
    c.info["bend180_radius"] = bend180_radius

    # delete any straights with zero length
    for inst in s1, s2, s3, s4:
        if inst.cell.settings["length"] == 0:
            del c.insts[inst]

    return c


if __name__ == "__main__":
    # test_delay_snake_sbend_length()
    # c = gf.grid(
    #     [
    #         delay_snake_sbend(length=length, cross_section="rib")
    #         for length in [500, 3000]
    #     ]
    # )
    c = delay_snake_sbend(length=200, cross_section="strip")
    c.show()
