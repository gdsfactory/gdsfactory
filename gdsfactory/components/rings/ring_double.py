from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def ring_double(
    gap: float = 0.2,
    gap_top: float | None = None,
    gap_bot: float | None = None,
    radius: float = 10.0,
    length_x: float = 0.01,
    length_y: float = 0.01,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    coupler_ring: ComponentSpec = "coupler_ring",
    coupler_ring_top: ComponentSpec | None = None,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns a double bus ring.

    two couplers (ct: top, cb: bottom)
    connected with two vertical straights (sl: left, sr: right)

    Args:
        gap: gap between for coupler.
        gap_top: gap for the top coupler. Defaults to gap.
        gap_bot: gap for the bottom coupler. Defaults to gap.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        bend: 90 degrees bend spec.
        straight: straight spec.
        coupler_ring: ring coupler spec.
        coupler_ring_top: top ring coupler spec. Defaults to coupler_ring.
        cross_section: cross_section spec.

    .. code::

           o2──────▲─────────o3
                   │gap_top
           xx──────▼─────────xxx
          xxx                   xxx
        xxx                       xxx
       xx                           xxx
       x                             xxx
      xx                              xx▲
      xx                              xx│length_y
      xx                              xx▼
      xx                             xx
       xx          length_x          x
        xx     ◄───────────────►    x
         xx                       xxx
           xx                   xxx
            xxx──────▲─────────xxx
                     │gap
             o1──────▼─────────o4
    """
    gap_top = gap_top or gap
    gap_bot = gap_bot or gap
    coupler_component_bot = gf.get_component(
        coupler_ring,
        gap=gap_bot,
        radius=radius,
        length_x=length_x,
        cross_section=cross_section,
        straight=straight,
        bend=bend,
    )
    coupler_component_top = gf.get_component(
        coupler_ring_top or coupler_ring,
        gap=gap_top,
        radius=radius,
        length_x=length_x,
        cross_section=cross_section,
        straight=straight,
        bend=bend,
    )
    straight_component = gf.get_component(
        straight,
        length=length_y,
        cross_section=cross_section,
    )

    c = Component()
    cb = c.add_ref(coupler_component_bot)
    ct = c.add_ref(coupler_component_top)

    length_y = length_y or 0.001

    sl = c << straight_component
    sr = c << straight_component

    sl.connect(port="o1", other=cb.ports["o2"])
    sr.connect(port="o2", other=cb.ports["o3"])
    ct.connect(port="o3", other=sl.ports["o2"])

    c.add_port("o1", port=cb.ports["o1"])
    c.add_port("o2", port=cb.ports["o4"])
    c.add_port("o3", port=ct.ports["o4"])
    c.add_port("o4", port=ct.ports["o1"])
    return c


if __name__ == "__main__":
    c = ring_double(length_y=2, bend="bend_circular", gap_top=0.4)
    c.show()
