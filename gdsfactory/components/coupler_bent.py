import gdsfactory as gf
import numpy as np


@gf.cell
def coupler_bent_half(
    gap: float = 0.200,
    radius: float = 26,
    length: float = 8.6,
    width1: float = 0.400,
    width2: float = 0.400,
    length_straight: float = 10,
) -> gf.Component:
    R1 = radius + (width1 + gap) / 2
    R2 = radius - (width2 + gap) / 2
    alpha = round(np.rad2deg(length / (2 * radius)), 4)
    beta = alpha

    c = gf.Component()

    outer_bend = c << gf.components.bend_circular(
        radius=R1,
        angle=np.round(-alpha, 3),
        cross_section=gf.cross_section.cross_section(width=width1),
        with_bbox=False,
    )

    inner_bend = c << gf.components.bend_circular(
        radius=R2,
        angle=-alpha,
        cross_section=gf.cross_section.cross_section(width=width2),
        with_bbox=True,
    )

    outer_bend.movey((width1 + gap) / 2)
    inner_bend.movey(-(width2 + gap) / 2)

    outer_straight = c << gf.components.straight(
        length=length,
        cross_section=gf.cross_section.cross_section(width=width1),
        npoints=100,
    )

    inner_straight = c << gf.components.straight(
        length=length,
        cross_section=gf.cross_section.cross_section(width=width2),
        npoints=100,
    )

    outer_straight.connect(port="o1", destination=outer_bend.ports["o2"])
    inner_straight.connect(port="o1", destination=inner_bend.ports["o2"])

    outer_exit_bend = c << gf.components.bend_circular(
        radius=R2,
        angle=alpha,
        cross_section=gf.cross_section.cross_section(width=width1),
        bend=gf.components.bend_circular(),
        with_bbox=True,
    )

    inner_exit_bend_down = c << gf.components.bend_circular(
        radius=R2,
        angle=-beta,
        cross_section=gf.cross_section.cross_section(width=width2),
        bend=gf.components.bend_circular(),
        with_bbox=True,
    )

    inner_exit_bend_up = c << gf.components.bend_circular(
        radius=R2,
        angle=alpha + beta,
        cross_section=gf.cross_section.cross_section(width=width2),
        bend=gf.components.bend_circular(),
        with_bbox=True,
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
    return c


@gf.cell
def coupler_bent(
    gap: float = 0.200,
    radius: float = 26,
    length: float = 8.6,
    width1: float = 0.400,
    width2: float = 0.400,
    length_straight: float = 10,
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
    """
    c = gf.Component()

    right_half = c << coupler_bent_half(
        gap=gap,
        radius=radius,
        length=length,
        width1=width1,
        width2=width2,
        length_straight=length_straight,
    )
    left_half = c << coupler_bent_half(
        gap=gap,
        radius=radius,
        length=length,
        width1=width1,
        width2=width2,
        length_straight=length_straight,
    )

    left_half.mirror_x()
    left_half.connect(port="o1", destination=right_half.ports["o1"])

    c.add_port("o1", port=left_half.ports["o3"])
    c.add_port("o2", port=left_half.ports["o4"])
    c.add_port("o3", port=right_half.ports["o3"])
    c.add_port("o4", port=right_half.ports["o4"])

    return c


if __name__ == "__main__":
    c = coupler_bent(length_straight=20)
    c.show(show_ports=True)
