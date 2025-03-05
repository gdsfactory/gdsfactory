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
    length_straight_exit: float = 18,
    cross_section: str = "strip",
) -> gf.Component:
    """Returns Broadband SOI curved / straight directional coupler.

    Args:
        gap: gap.
        radius: radius coupling.
        length: coupler_length.
        width1: width1.
        width2: width2.
        length_straight: input and output straight length.
        length_straight_exit: length straight exit.
        cross_section: cross_section.
    """
    radius_outer = radius + (width1 + gap) / 2
    radius_inner = radius - (width2 + gap) / 2
    alpha = round(np.rad2deg(length / (2 * radius)), 4)
    beta = alpha

    c = gf.Component()

    xs = gf.get_cross_section(cross_section)
    xs1 = xs.copy(radius=radius_outer, width=width1)
    xs2 = xs.copy(radius=radius_inner, width=width2)

    outer_bend = gf.path.arc(angle=-alpha, radius=radius_outer)
    inner_bend = gf.path.arc(angle=-alpha, radius=radius_inner)

    outer_straight = gf.path.straight(length=length, npoints=100)
    inner_straight = gf.path.straight(length=length, npoints=100)

    outer_exit_bend = gf.path.arc(angle=alpha, radius=radius_outer)
    inner_exit_bend_down = gf.path.arc(angle=-beta, radius=radius_inner)
    inner_exit_bend_up = gf.path.arc(angle=alpha + beta, radius=radius_inner)

    inner_exit_straight = gf.path.straight(
        length=length_straight,
        npoints=100,
    )
    outer_exit_straight = gf.path.straight(
        length=length_straight_exit,
        npoints=100,
    )

    outer = outer_bend + outer_straight + outer_exit_bend + outer_exit_straight
    inner = (
        inner_bend
        + inner_straight
        + inner_exit_bend_down
        + inner_exit_bend_up
        + inner_exit_straight
    )

    inner_component = c << inner.extrude(xs2)
    outer_component = c << outer.extrude(xs1)
    outer_component.dmovey(+(width1 + gap) / 2)
    inner_component.dmovey(-(width2 + gap) / 2)

    c.add_port("o1", port=outer_component.ports["o1"])
    c.add_port("o2", port=inner_component.ports["o1"])
    c.add_port("o3", port=outer_component.ports["o2"])
    c.add_port("o4", port=inner_component.ports["o2"])
    c.flatten()
    return c


@gf.cell
def coupler_bent(
    gap: float = 0.200,
    radius: float = 26,
    length: float = 8.6,
    width1: float = 0.400,
    width2: float = 0.400,
    length_straight: float = 10,
    cross_section: str = "strip",
) -> gf.Component:
    """Returns Broadband SOI curved / straight directional coupler.

    based on: https://doi.org/10.1038/s41598-017-07618-6.

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

    left_half.connect(port="o1", other=right_half.ports["o1"], mirror=True)

    c.add_port("o1", port=left_half.ports["o3"])
    c.add_port("o2", port=left_half.ports["o4"])
    c.add_port("o3", port=right_half.ports["o3"])
    c.add_port("o4", port=right_half.ports["o4"])

    c.flatten()
    return c


if __name__ == "__main__":
    c = coupler_bent()
    # c = coupler_bent_half()
    # c = coupler_bent()
    c.show()
