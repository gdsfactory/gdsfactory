import numpy as np

import gdsfactory as gf
from gdsfactory import Component
from gdsfactory.types import CrossSectionSpec


def _compute_parameters(xs_bend, wrap_angle_deg, radius):

    r_bend = xs_bend.radius
    theta = wrap_angle_deg / 2.0
    size_x, dy = r_bend * np.sin(theta * np.pi / 180), r_bend - r_bend * np.cos(
        theta * np.pi / 180
    )
    bus_length = 2 * radius if (4 * size_x < 2 * radius) else 4 * size_x

    return (r_bend, size_x, dy, bus_length)


def _generate_bends(c, r_bend, wrap_angle_deg, cross_section):

    if wrap_angle_deg != 0:
        input_arc = gf.path.arc(radius=r_bend, angle=-wrap_angle_deg / 2.0)
        bend_middle_arc = gf.path.arc(radius=r_bend, angle=-wrap_angle_deg)

        bend_input_output = input_arc.extrude(cross_section=cross_section)

        bend_input = c << bend_input_output
        bend_middle = c << bend_middle_arc.extrude(cross_section=cross_section)
        bend_middle.rotate(180 + wrap_angle_deg / 2.0, center=c.center)

        bend_input.connect("o2", bend_middle.ports["o2"])

        bend_output = c << bend_input_output
        bend_output.mirror()
        bend_output.connect("o2", bend_middle.ports["o1"])

        return (c, bend_input, bend_middle, bend_output)
    else:
        return (c, None, None, None)


def _generate_straights(c, bus_length, size_x, bend_input, bend_output, cross_section):

    straight_left = c << gf.components.straight(
        length=(bus_length - 4 * size_x) / 2.0, cross_section=cross_section
    )

    straight_right = c << gf.components.straight(
        length=(bus_length - 4 * size_x) / 2.0, cross_section=cross_section
    )

    if None not in (bend_input, bend_output):
        straight_left.connect("o2", bend_input.ports["o1"])

        straight_right.connect("o1", bend_output.ports["o1"])
    else:
        straight_left.connect("o2", straight_right.ports["o1"])

    return (c, straight_left, straight_right)


def _generate_circles(c, radius, xs, bend_middle, straight_left, r_bend, dy):

    circle = c << gf.components.circle(radius=radius, layer=xs.layer)

    circle_cladding = c << gf.components.circle(
        radius=radius + xs.cladding_offsets[0], layer=xs.cladding_layers[0]
    )

    if bend_middle is not None:
        circle.move(
            origin=circle.center,
            destination=(
                (bend_middle.ports["o1"].x + bend_middle.ports["o2"].x) / 2.0,
                straight_left.ports["o2"].y - 2 * dy + r_bend,
            ),
        )
    else:
        circle.move(
            origin=circle.center,
            destination=(straight_left.ports["o2"].center + (0, r_bend),),
        )

    circle_cladding.move(origin=circle_cladding.center, destination=circle.center)

    return (c, circle, circle_cladding)


def _absorb(c, *refs):
    for ref in list(refs):
        if ref is not None:
            c.absorb(ref)
    return c


@gf.cell
def disk(
    radius: float = 10.0,
    gap: float = 0.2,
    wrap_angle_deg: float = 180.0,
    parity: int = 1,
    cross_section: CrossSectionSpec = "strip",
    **kwargs
) -> Component:
    """Disk Resonator.

    Args:
       radius: disk resonator radius.
       gap: Distance between the bus straight and resonator.
       wrap_angle_deg: Angle in degrees between 0 and 180.
        determines how much the bus straight wraps along the resonator.
        0 corresponds to a straight bus straight.
        180 corresponds to a bus straight wrapped around half of the resonator.
       parity (1 or -1): 1, resonator left from bus straight, -1 resonator to the right.
       cross_section: cross_section spec.
       kwargs: cross_section settings.

    """
    if parity not in (1, -1):
        raise ValueError("parity must be 1 or -1")

    if wrap_angle_deg < 0.0 or wrap_angle_deg > 180.0:
        raise ValueError("wrap_angle_deg must be between 0.0 and 180.0")

    c = gf.Component()

    xs = gf.get_cross_section(cross_section=cross_section, radius=radius, **kwargs)
    xs_bend = xs.copy()
    xs_bend.radius = radius + xs.width / 2.0 + gap

    r_bend, size_x, dy, bus_length = _compute_parameters(
        xs_bend, wrap_angle_deg, radius
    )

    c, bend_input, bend_middle, bend_output = _generate_bends(
        c, r_bend, wrap_angle_deg, xs_bend
    )

    c, straight_left, straight_right = _generate_straights(
        c, bus_length, size_x, bend_input, bend_output, xs_bend
    )

    c, circle, circle_cladding = _generate_circles(
        c, radius, xs, bend_middle, straight_left, r_bend, dy
    )

    c = _absorb(
        c,
        circle,
        circle_cladding,
        straight_left,
        straight_right,
        bend_input,
        bend_middle,
        bend_output,
    )

    c.add_port("o1", port=straight_left.ports["o1"], layer="PORT")
    c.add_port("o2", port=straight_right.ports["o2"])

    if xs.add_bbox:
        c = xs.add_bbox(c)

    if xs.info:
        c = xs.info(c)

    if parity == -1:
        c = c.rotate(180)

    c.snap_ports_to_grid()

    return c


if __name__ == "__main__":

    c = disk(wrap_angle_deg=30)
    c.show(show_ports=True)
