from __future__ import annotations

from typing import Tuple

import gdsfactory as gf
from gdsfactory import Component
from gdsfactory.cross_section import strip
from gdsfactory.typings import CrossSectionSpec


def _generate_fins(c, x, fin_size, bend):
    num_fins = int(x.width // (2 * fin_size[1]))
    y0 = bend.ports["o1"].x, -num_fins * (2 * fin_size[1]) / 2.0 + fin_size[1]

    for i in range(num_fins):
        y = y0 + i * 2 * fin_size[1]
        rectangle = c << gf.components.rectangle(
            size=(fin_size[0], fin_size[1]),
            layer=x.layer,
            centered=True,
            port_type=None,
            port_orientations=None,
        )
        rectangle.movex(
            destination=bend.ports["o1"].x
            - (1 - bend.ports["o1"].orientation / 90.0) * fin_size[0] / 2.0
        )

        rectangle.movey(
            origin=rectangle.y,
            destination=bend.ports["o1"].y + y,
        )
        c.absorb(rectangle)

    return c


def _generate_bends(c, x_top, x_bot, dx, dy, gap):
    input_bend_top = (
        c << gf.components.bend_s(size=(dx, dy), cross_section=x_top.copy()).mirror()
    )

    input_bend_bottom = c << gf.components.bend_s(
        size=(dx, dy), cross_section=x_bot
    ).mirror().mirror(p1=(1, 0))

    dy = input_bend_bottom.ports["o2"].y - input_bend_top.ports["o2"].y

    input_bend_top.movey(destination=dy / 2.0 + gap / 2.0)

    input_bend_bottom.movey(
        destination=-dy / 2.0 - (x_bot.width / 2.0 + x_top.width / 2.0 + gap / 2.0)
    )

    return (c, input_bend_top, input_bend_bottom)


def _generate_straights(c, length, x_top, x_bot, input_bend_top, input_bend_bottom):
    top_straight = c << gf.components.straight(length=length, cross_section=x_top)

    bottom_straight = c << gf.components.straight(length=length, cross_section=x_bot)

    top_straight.movey(destination=input_bend_top.ports["o2"].y)

    bottom_straight.movey(destination=input_bend_bottom.ports["o2"].y)

    return (c, top_straight, bottom_straight)


def _generate_gratings(c, length, period, dc, gap, x, bottom_straight, x_bot):
    num = length // period
    x_size = period * dc
    start = length - (num - 1) * period

    for i in range(int(num)):
        x_start = start + i * period
        rectangle = c << gf.components.rectangle(
            size=(x_size, gap),
            layer=x.layer,
            centered=True,
            port_type=None,
            port_orientations=None,
        )
        rectangle.movex(destination=x_start)
        rectangle.movey(
            destination=bottom_straight.ports["o1"].y + x_bot.width / 2.0 + gap / 2.0
        )

    return c


def _absorb(c, *refs):
    for ref in refs:
        c.absorb(ref)
    return c


@gf.cell
def cdc(
    length: float = 30.0,
    gap: float = 0.5,
    period: float = 0.220,
    dc: float = 0.5,
    dx: float = 10.0,
    dy: float = 5.0,
    width_top: float = 2.0,
    width_bot: float = 0.75,
    fins: bool = False,
    fin_size: Tuple[float, float] = (0.2, 0.05),
    cross_section: CrossSectionSpec = strip,
    **kwargs,
) -> Component:
    """Grating-Assisted Contra-Directional Coupler.

    Args:
       length : Length of the coupling region.
       gap: Distance between the two straights.
       period: Period of the grating.
       dc: Duty cycle of the grating. Must be between 0 and 1.
       width_top: Width of the top straight in the coupling region.
       width_bot: Width of the bottom straight in the coupling region.
       dx: size of bends in x-direction.
       dy: size of bends in y-direction.
       fins: If `True`, adds fins to the input/output straights.
        In this case a different template for the component must be specified.
        This feature is useful when performing electron-beam lithography
        and using different beam currents
        for fine features (helps to reduce stitching errors).
       fin_size: Specifies the x- and y-size of the `fins`. Defaults to 200 nm x 50 nm.
       cross_section: CrossSection spec.

    kwargs:
        cross_section kwargs.
    """
    x = gf.get_cross_section(cross_section, **kwargs)
    x_top = x.copy(width=width_top)
    x_bot = x.copy(width=width_bot)

    c = gf.Component()

    c, input_bend_top, input_bend_bottom = _generate_bends(c, x_top, x_bot, dx, dy, gap)
    c, top_straight, bottom_straight = _generate_straights(
        c, length, x_top, x_bot, input_bend_top, input_bend_bottom
    )

    bend_output_top = c << input_bend_top.parent.mirror()
    bend_output_bottom = c << input_bend_bottom.parent.mirror()

    input_bend_top.connect("o2", top_straight.ports["o1"])
    input_bend_bottom.connect("o2", bottom_straight.ports["o1"])
    bend_output_top.connect("o2", top_straight.ports["o2"])
    bend_output_bottom.connect("o2", bottom_straight.ports["o2"])

    c = _generate_gratings(c, length, period, dc, gap, x, bottom_straight, x_bot)

    if fins:
        c = _generate_fins(c, x_top, fin_size, input_bend_top)
        c = _generate_fins(c, x_bot, fin_size, input_bend_bottom)
        c = _generate_fins(c, x_top, fin_size, bend_output_top)
        c = _generate_fins(c, x_bot, fin_size, bend_output_bottom)

    c = _absorb(
        c,
        input_bend_top,
        input_bend_bottom,
        top_straight,
        bottom_straight,
        bend_output_top,
        bend_output_bottom,
    )

    if x.add_bbox:
        c = x.add_bbox(c)

    c.add_port("o1", port=input_bend_top.ports["o1"])
    c.add_port("o2", port=input_bend_bottom.ports["o1"])
    c.add_port("o3", port=bend_output_top.ports["o1"])
    c.add_port("o4", port=bend_output_bottom.ports["o1"])

    return c


if __name__ == "__main__":
    c = cdc(fins=False)
    print(c.ports.keys())
    c.show(show_ports=True)
