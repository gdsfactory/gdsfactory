import numpy as np

import gdsfactory as gf


@gf.cell
def coupler_bent_half(
    gap: float = 0.200,
    radius: float = 26,
    length: float = 8.6,
    width1: float = 0.400,
    width2: float = 0.400,
    length_straight: float = 10,
    cross_section: str = "xs_sc",
) -> gf.Component:
    """Returns Broadband SOI curved / straight directional coupler.

    Args:
        gap: gap.
        radius: radius coupling.
        length: coupler_length.
        width1: width1.
        width2: width2.
        length_straight: input and output straight length.
        cross_section: cross_section.
    """
    R1 = radius + (width1 + gap) / 2
    R2 = radius - (width2 + gap) / 2
    alpha = round(np.rad2deg(length / (2 * radius)), 4)
    beta = alpha

    c = gf.Component()

    xs = gf.get_cross_section(cross_section)
    xs1 = xs.copy(radius=R1, width=width1)
    xs2 = xs.copy(radius=R2, width=width2)

    outer_bend = c << gf.components.bend_circular(
        angle=np.round(-alpha, 3), cross_section=xs1, add_pins=False
    )

    inner_bend = c << gf.components.bend_circular(
        angle=-alpha, cross_section=xs2, add_pins=False
    )

    outer_bend.movey(+(width1 + gap) / 2)
    inner_bend.movey(-(width2 + gap) / 2)

    outer_straight = c << gf.components.straight(
        length=length, cross_section=xs1, npoints=100, add_pins=False
    )

    inner_straight = c << gf.components.straight(
        length=length, cross_section=xs2, npoints=100, add_pins=False
    )

    outer_straight.connect(port="o1", destination=outer_bend.ports["o2"])
    inner_straight.connect(port="o1", destination=inner_bend.ports["o2"])

    outer_exit_bend = c << gf.components.bend_circular(
        angle=alpha, cross_section=xs1, add_pins=False
    )

    inner_exit_bend_down = c << gf.components.bend_circular(
        angle=-beta, cross_section=xs2, add_pins=False
    )

    inner_exit_bend_up = c << gf.components.bend_circular(
        angle=alpha + beta, cross_section=xs2, add_pins=False
    )

    outer_exit_bend.connect(port="o1", destination=outer_straight.ports["o2"])
    inner_exit_bend_down.connect(port="o1", destination=inner_straight.ports["o2"])
    inner_exit_bend_up.connect(port="o1", destination=inner_exit_bend_down.ports["o2"])

    inner_exit_straight = c << gf.components.straight(
        length=length_straight,
        cross_section=gf.cross_section.cross_section(width=width2),
        npoints=100,
    )
    inner_exit_straight.connect(port="o1", destination=inner_exit_bend_up.ports["o2"])

    outer_exit_straight = c << gf.components.straight(
        length=abs(inner_exit_straight.ports["o2"].x - outer_exit_bend.ports["o2"].x),
        cross_section=gf.cross_section.cross_section(width=width1),
        npoints=100,
    )
    outer_exit_straight.connect(port="o1", destination=outer_exit_bend.ports["o2"])

    c.add_port("o1", port=outer_bend.ports["o1"])
    c.add_port("o2", port=inner_bend.ports["o1"])
    c.add_port("o3", port=outer_exit_straight.ports["o2"])
    c.add_port("o4", port=inner_exit_straight.ports["o2"])

    c.absorb(outer_exit_bend)
    c.absorb(inner_exit_bend_down)
    c.absorb(inner_exit_bend_up)
    c = c.flatten_offgrid_references()
    return c


@gf.cell
def coupler_bent(
    gap: float = 0.200,
    radius: float = 26,
    length: float = 8.6,
    width1: float = 0.400,
    width2: float = 0.400,
    length_straight: float = 10,
    cross_section: str = "xs_sc",
) -> gf.Component:
    """Returns Broadband SOI curved / straight directional coupler.
    based on: https://doi.org/10.1038/s41598-017-07618-6

    Args:
        gap: gap.
        radius: radius coupling.
        length: coupler_length.
        width1: width1.
        width2: width2.
        length_straight: input and output straight length.
        cross_section: cross_section.
    """
    c = gf.Component()
    xs = gf.get_cross_section(cross_section)

    right_half = c << coupler_bent_half(
        gap=gap,
        radius=radius,
        length=length,
        width1=width1,
        width2=width2,
        length_straight=length_straight,
        cross_section=cross_section,
    )
    left_half = c << coupler_bent_half(
        gap=gap,
        radius=radius,
        length=length,
        width1=width1,
        width2=width2,
        length_straight=length_straight,
        cross_section=cross_section,
    )

    left_half.mirror_x()
    left_half.connect(port="o1", destination=right_half.ports["o1"])
    c = c.flatten()

    c.add_port("o1", port=left_half.ports["o3"])
    c.add_port("o2", port=left_half.ports["o4"])
    c.add_port("o3", port=right_half.ports["o3"])
    c.add_port("o4", port=right_half.ports["o4"])

    xs.add_pins(c)
    return c


if __name__ == "__main__":
    # c = coupler_bent_half()
    c = coupler_bent()
    c.show(show_ports=False)
