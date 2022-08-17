import numpy as np

import gdsfactory as gf
from gdsfactory import Component
from gdsfactory.cross_section import strip
from gdsfactory.types import CrossSectionSpec


@gf.cell
def disk(
    radius: float = 10.0,
    gap: float = 0.2,
    wrap_angle_deg: float = 180.0,
    parity: int = 1,
    cross_section: CrossSectionSpec = strip,
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

    Keyword Args:
        cross_section kwargs.
    """
    if parity not in (1, -1):
        raise ValueError("parity must be 1 or -1")

    if wrap_angle_deg < 0.0 or wrap_angle_deg > 180.0:
        raise ValueError("wrap_angle_deg must be between 0.0 and 180.0")

    c = gf.Component()

    xs = gf.get_cross_section(cross_section=cross_section, radius=radius, **kwargs)
    xs_bend = xs.copy()
    xs_bend.radius = radius + xs.width / 2.0 + gap

    r_bend = xs_bend.radius
    theta = wrap_angle_deg / 2.0
    size_x, dy = r_bend * np.sin(theta * np.pi / 180), r_bend - r_bend * np.cos(
        theta * np.pi / 180
    )
    bus_length = 2 * radius if (4 * size_x < 2 * radius) else 4 * size_x

    input_arc = gf.path.arc(radius=r_bend, angle=-wrap_angle_deg / 2.0)

    bend_input = c << input_arc.extrude(cross_section=xs_bend.copy(width=xs_bend.width))

    bend_middle_arc = gf.path.arc(radius=r_bend, angle=-wrap_angle_deg)

    bend_middle = c << bend_middle_arc.extrude(
        cross_section=xs_bend.copy(width=xs_bend.width)
    )

    bend_middle.rotate(180 + wrap_angle_deg / 2.0, center=c.center)

    bend_input.connect("o2", bend_middle.ports["o2"])

    bend_output = c << bend_input.parent.copy()

    bend_output.x_reflection = True

    bend_output.connect("o2", bend_middle.ports["o1"])

    straight_left = c << gf.components.straight(
        length=(bus_length - 4 * size_x) / 2.0,
        cross_section=xs_bend.copy(width=xs_bend.width),
    )

    straight_left.connect("o2", bend_input.ports["o1"])

    straight_right = c << gf.components.straight(
        length=(bus_length - 4 * size_x) / 2.0,
        cross_section=xs_bend.copy(width=xs_bend.width),
    )

    straight_right.connect("o1", bend_output.ports["o1"])

    circle = c << gf.components.circle(radius=radius, layer=xs.layer)

    circle_cladding = c << gf.components.circle(
        radius=radius + xs.cladding_offsets[0], layer=xs.cladding_layers[0]
    )

    circle.move(
        origin=circle.center,
        destination=(
            (bend_middle.ports["o1"].x + bend_middle.ports["o2"].x) / 2.0,
            straight_left.ports["o2"].y - 2 * dy + r_bend,
        ),
    )

    circle_cladding.move(origin=circle_cladding.center, destination=circle.center)

    c.absorb(circle)
    c.absorb(straight_left)
    c.absorb(straight_right)
    c.absorb(bend_input)
    c.absorb(bend_middle)
    c.absorb(bend_output)

    c.add_port("o1", port=straight_left.ports["o1"])
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
    c.show(show_ports=True, show_subports=True)
