from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bends.bend_s import bend_s
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def mode_converter(
    gap: float = 0.3,
    length: float = 10,
    coupler_straight_asymmetric: ComponentSpec = "coupler_straight_asymmetric",
    bend: ComponentSpec = partial(bend_s, size=(25, 3)),
    taper: ComponentSpec = "taper",
    mm_width: float = 1.2,
    mc_mm_width: float = 1,
    sm_width: float = 0.5,
    taper_length: float = 25,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""Returns Mode converter from TE0 to TE1.

    By matching the effective indices of two waveguides with different widths,
    light can couple from different transverse modes e.g. TE0 <-> TE1.
    https://doi.org/10.1109/JPHOT.2019.2941742

    Args:
        gap: directional coupler gap.
        length: coupler length interaction.
        coupler_straight_asymmetric: spec.
        bend: spec.
        taper: spec.
        mm_width: input/output multimode waveguide width.
        mc_mm_width: mode converter multimode waveguide width
        sm_width: single mode waveguide width.
        taper_length: taper length.
        cross_section: cross_section spec.

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
        width_bot=mc_mm_width,
        width_top=sm_width,
        cross_section=cross_section,
    )

    bend = gf.get_component(bend, cross_section=cross_section)

    bot_taper = gf.get_component(
        taper,
        width1=mc_mm_width,
        width2=mm_width,
        length=taper_length,
        cross_section=cross_section,
    )

    # directional coupler
    dc = c << coupler

    # straight waveguides at the bottom
    l_bot_straight = c << bot_taper
    r_bot_straight = c << bot_taper

    l_bot_straight.connect("o1", dc.ports["o1"])
    r_bot_straight.connect("o1", dc.ports["o4"])

    # top right bend with termination
    r_bend = c << bend
    l_bend = c << bend

    l_bend.connect("o1", dc.ports["o2"], mirror=True)
    r_bend.connect("o1", dc.ports["o3"])

    # define ports of mode converter
    c.add_port("o1", port=l_bot_straight.ports["o2"])
    c.add_port("o3", port=r_bot_straight.ports["o2"])
    c.add_port("o2", port=l_bend.ports["o2"])
    c.add_port("o4", port=r_bend.ports["o2"])
    return c


if __name__ == "__main__":
    # c = mode_converter(bbox_offsets=(0.5,), bbox_layers=((111, 0),))
    c = mode_converter(cross_section="rib")
    c.pprint_ports()
    c.show()
