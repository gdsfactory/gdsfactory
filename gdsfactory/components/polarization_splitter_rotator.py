from __future__ import annotations

from typing import Union

from numpy import ndarray

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bezier import bezier
from gdsfactory.components.coupler_straight_asymmetric import (
    coupler_straight_asymmetric,
)
from gdsfactory.components.taper import taper
from gdsfactory.typings import CrossSectionSpec, Float2, Float3


@gf.cell
def polarization_splitter_rotator(
    width_taper_in: Float3 = (0.54, 0.69, 0.83),
    length_taper_in: Union[Float2, Float3] = (4.0, 44.0),
    width_coupler: Float2 = (0.9, 0.405),
    length_coupler: float = 7.0,
    gap: float = 0.15,
    width_out: float = 0.54,
    length_out: float = 14.33,
    dy: float = 5.0,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    """Returns polarization splitter rotator

    "Novel concept for ultracompact polarization splitter-rotator
    based on silicon nanowires." By D. Dai, and J. E. Bowers
    (Optics express vol 19, no. 11 pp. 10940-10949 (2011)).

    Args:
        width_taper_in: Three west widths of the input tapers in um.
        length_taper_in: Two or three length of the bend regions in um.
        width_coupler: Top and bottom widths of the coupling region in um.
        length_coupler: Length of the coupling region in um.
        gap: Distance between the coupler in um.
        width_out: Width of the splitter region in um.
        length_out: Length of the splitter region in um.
        dy: Port-to-port distance between the splitter region in um.
        cross_section: cross-section spec.

    Keyword Args:
        cross_section kwargs.

    Notes:
        The length of third input taper is automatically determined
        if only two lengths are in arguments.
    """

    c = gf.Component()
    x = gf.get_cross_section(cross_section=cross_section, **kwargs)

    w0, w1, w2 = width_taper_in
    w3, w4 = width_coupler
    if len(length_taper_in) == 2:
        l1, l2 = length_taper_in
        l3 = l1 * (w3 - w2) / (w1 - w0)
    else:
        l1, l2, l3 = length_taper_in

    taper_in1 = c << taper(length=l1, width1=w0, width2=w1, cross_section=x)
    taper_in2 = c << taper(length=l2, width1=w1, width2=w2, cross_section=x)
    taper_in3 = c << taper(length=l3, width1=w2, width2=w3, cross_section=x)

    coupler = c << coupler_straight_asymmetric(
        length=length_coupler, gap=gap, width_top=w4, width_bot=w3, cross_section=x
    )

    def bend_s_width(t: ndarray) -> ndarray:
        return w4 + (width_out - w4) * t

    x_bend = x.copy(width=bend_s_width)

    bend_s_var = c << bezier(
        control_points=(
            (0, 0),
            (length_out / 2, 0),
            (length_out / 2, dy),
            (length_out, dy),
        ),
        cross_section=x_bend,
        **kwargs,
    )

    taper_out = c << taper(
        length=length_out, width1=w3, width2=width_out, cross_section=x
    )

    taper_in3.connect("o2", destination=coupler.ports["o1"])
    taper_in2.connect("o2", destination=taper_in3.ports["o1"])
    taper_in1.connect("o2", destination=taper_in2.ports["o1"])
    taper_out.connect("o1", destination=coupler.ports["o4"])
    bend_s_var.connect("o1", destination=coupler.ports["o3"])

    c.add_port("o1", port=taper_in1.ports["o1"])
    c.add_port("o2", port=bend_s_var.ports["o2"])
    c.add_port("o3", port=taper_out.ports["o2"])

    c.absorb(coupler)
    c.absorb(taper_in3)
    c.absorb(taper_in2)
    c.absorb(taper_in1)
    c.absorb(taper_out)
    c.absorb(bend_s_var)

    c.info["length"] = bend_s_var.info["length"]
    c.info["min_bend_radius"] = bend_s_var.info["min_bend_radius"]
    c.auto_rename_ports()

    return c


if __name__ == "__main__":
    c = polarization_splitter_rotator(length_taper_in=(10, 69))
    c.show(show_ports=True)
