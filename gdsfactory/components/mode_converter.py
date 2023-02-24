from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.coupler_straight_asymmetric import (
    coupler_straight_asymmetric as coupler_straight_asymmetric_function,
)
from gdsfactory.components.bend_s import bend_s
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def mode_converter(
    gap: float = 0.3,
    length: float = 10,
    coupler_straight_asymmetric: ComponentSpec = coupler_straight_asymmetric_function,
    bend: ComponentSpec = bend_s,
    taper: ComponentSpec = taper_function,
    mm_width: float = 1.0,
    sm_width: float = 0.5,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    r"""Returns Mode converter from TE0 to TE1.

    By matching the effective indices of two waveguides with different widths,
    light can couple from different transverse modes e.g. TE0 <-> TE1.
    https://doi.org/10.1109/JPHOT.2019.2941742

    Args:
        gap: directional coupler gap.
        length: coupler length interaction.
        coupler_straight_asymmetric: spec.
        mm_width: multimode waveguide width.
        sm_width: single mode waveguide width.
        cross_section: cross_section spec.
        kwargs: cross_section settings.

    .. code::

        o2 ---           --- o4
              \         /
               \       /
                -------
        o1 -----=======----- o3
                |-----|
                length

        = : multimode width
        - : singlemode width
    """

    c = Component()

    coupler = gf.get_component(
        coupler_straight_asymmetric,
        length=length,
        gap=gap,
        width_bot=mm_width,
        width_top=sm_width,
        **kwargs,
    )

    bend = gf.get_component(bend, **kwargs)

    bot_taper = gf.get_component(
        taper,
        width1=mm_width,
        width2=sm_width,
        length=bend.xsize,
        **kwargs,
    )

    # directional coupler
    dc = c << coupler
    c.absorb(dc)

    # straight waveguides at the bottom
    l_bot_straight = c << bot_taper
    r_bot_straight = c << bot_taper

    l_bot_straight.connect("o1", dc.ports["o1"])
    r_bot_straight.connect("o1", dc.ports["o4"])
    c.absorb(l_bot_straight)
    c.absorb(r_bot_straight)

    # top right bend with termination
    r_bend = c << bend
    l_bend = c << bend
    l_bend.mirror()

    l_bend.connect("o1", dc.ports["o2"])
    r_bend.connect("o1", dc.ports["o3"])

    # define ports of mode converter
    c.add_port("o1", port=l_bot_straight.ports["o2"])
    c.add_port("o3", port=r_bot_straight.ports["o2"])
    c.add_port("o2", port=l_bend.ports["o2"])
    c.add_port("o4", port=r_bend.ports["o2"])

    x = gf.get_cross_section(cross_section, **kwargs)
    if x.add_bbox:
        c = x.add_bbox(c)
    if x.add_pins:
        c = x.add_pins(c)
    return c


if __name__ == "__main__":
    c = mode_converter(bbox_offsets=[0.5], bbox_layers=[(111, 0)])
    c.pprint_ports()
    c.show(show_ports=True)
