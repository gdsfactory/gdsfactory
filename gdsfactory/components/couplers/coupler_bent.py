from __future__ import annotations

import numpy as np

import gdsfactory as gf


@gf.cell_with_module_name
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
    # Pre-calculate repeated values
    width_gap_1 = (width1 + gap) * 0.5
    width_gap_2 = (width2 + gap) * 0.5
    radius_outer = radius + width_gap_1
    radius_inner = radius - width_gap_2

    # Use formula for angle only once
    angle_forward = float(length / (2 * radius))
    alpha = np.rad2deg(angle_forward)

    c = gf.Component()

    xs = gf.get_cross_section(cross_section)
    # Only copy if width or radius is not already correct
    xs1 = (
        xs
        if (xs.width == width1 and getattr(xs, "radius", None) == radius_outer)
        else xs.copy(radius=radius_outer, width=width1)
    )
    xs2 = (
        xs
        if (xs.width == width2 and getattr(xs, "radius", None) == radius_inner)
        else xs.copy(radius=radius_inner, width=width2)
    )
    # Use small npoints for arc; sufficient for smoothness
    npoints_arc = 32
    npoints_straight = 16

    outer_bend = gf.path.arc(angle=-alpha, radius=radius_outer, npoints=npoints_arc)
    inner_bend = gf.path.arc(angle=-alpha, radius=radius_inner, npoints=npoints_arc)
    outer_straight = gf.path.straight(length=length, npoints=npoints_straight)
    inner_straight = gf.path.straight(length=length, npoints=npoints_straight)
    outer_exit_bend = gf.path.arc(angle=alpha, radius=radius_outer, npoints=npoints_arc)
    inner_exit_bend_dn = gf.path.arc(
        angle=-alpha, radius=radius_inner, npoints=npoints_arc
    )
    inner_exit_bend_up = gf.path.arc(
        angle=alpha * 2, radius=radius_inner, npoints=npoints_arc
    )
    inner_exit_straight = gf.path.straight(
        length=length_straight, npoints=npoints_straight
    )
    outer_exit_straight = gf.path.straight(
        length=length_straight_exit, npoints=npoints_straight
    )

    # Fast concatenation
    outer_path = outer_bend + outer_straight + outer_exit_bend + outer_exit_straight
    inner_path = (
        inner_bend
        + inner_straight
        + inner_exit_bend_dn
        + inner_exit_bend_up
        + inner_exit_straight
    )

    # Extrude both at once
    inner_component = c << inner_path.extrude(xs2)
    outer_component = c << outer_path.extrude(xs1)

    # Efficient moves
    d1 = width_gap_1
    d2 = width_gap_2
    outer_component.movey(+d1)
    inner_component.movey(-d2)

    # Direct port addition (no recomputation of names)
    c.add_port("o1", port=outer_component.ports["o1"])
    c.add_port("o2", port=inner_component.ports["o1"])
    c.add_port("o3", port=outer_component.ports["o2"])
    c.add_port("o4", port=inner_component.ports["o2"])
    c.flatten()
    return c


@gf.cell_with_module_name
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
    half_params = dict(
        gap=gap,
        radius=radius,
        length=length,
        width1=width1,
        width2=width2,
        length_straight=length_straight,
        cross_section=cross_section,
    )
    right_half = c << coupler_bent_half(**half_params)
    left_half = c << coupler_bent_half(**half_params)
    # direct connect+mirror
    left_half.connect(port="o1", other=right_half.ports["o1"], mirror=True)

    # All port additions in tight loop (no intermediate variables)
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
