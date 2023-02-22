from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.coupler_straight_asymmetric import (
    coupler_straight_asymmetric,
)
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.taper import taper
from gdsfactory.typings import ComponentSpec, CrossSectionSpec
import numpy as np


@gf.cell
def mode_converter(
    gap: float = 0.3,
    length: float = 10,
    coupler_straight_asymmetric: ComponentSpec = coupler_straight_asymmetric,
    bend_circular: ComponentSpec = bend_circular,
    taper: ComponentSpec = taper,
    mm_width: float = 1.0,
    sm_width: float = 0.5,
    radius: float = 10,
    angle: float = 45,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    r"""
    Mode converter built upon coupler_straight_asymmetric.
    By matching the effective indices of two waveguides with different widths,
    light can couple from different transverse modes e.g. TE0 <-> TE1.
    https://doi.org/10.1109/JPHOT.2019.2941742

    Args:
        gap: directional coupler gap
        length: interaction length of coupler
        mm_width: multimode waveguide width
        sm_width: single mode waveguide width
        radius: circular bend radius
        angle: circular bend angle

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

    bend = gf.get_component(bend_circular, radius=radius, angle=angle, **kwargs)

    bot_taper = gf.get_component(
        taper,
        width1=mm_width,
        width2=sm_width,
        length=2 * radius * np.sin(np.deg2rad(angle)),
        **kwargs,
    )

    # directional coupler
    dc = c << coupler
    c.absorb(dc)

    # straight waveguides at the bottom
    l_bot_straight = c << bot_taper.mirror()
    r_bot_straight = c << bot_taper

    l_bot_straight.connect("o1", dc.ports["o1"])
    r_bot_straight.connect("o1", dc.ports["o4"])

    # top right bend with termination
    r1_bend = c << bend
    r2_bend = c << bend.mirror()

    r1_bend.connect("o1", dc.ports["o3"])
    r2_bend.connect("o1", r1_bend["o2"])
    c.absorb(r1_bend)

    # top left bend
    l1_bend = c << bend.mirror()
    l2_bend = c << bend.mirror()

    l1_bend.connect("o1", dc.ports["o2"])
    l2_bend.connect("o2", l1_bend.ports["o2"])
    c.absorb(l1_bend)

    # define ports of mode converter
    c.add_port("o1", port=l_bot_straight.ports["o2"])
    c.add_port("o2", port=l2_bend.ports["o1"])
    c.add_port("o3", port=r_bot_straight.ports["o2"])
    c.add_port("o4", port=r2_bend.ports["o2"])
    c.auto_rename_ports()

    x = gf.get_cross_section(cross_section, **kwargs)
    if x.add_bbox:
        c = x.add_bbox(c)
    if x.add_pins:
        c = x.add_pins(c)
    return c


if __name__ == "__main__":
    c = mode_converter(bbox_offsets=[0.5], bbox_layers=[(111, 0)])
    c.show()
