from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.coupler_ring import coupler_ring
from gdsfactory.components.straight import straight
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def ring_double(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 0.01,
    length_y: float = 0.01,
    coupler_ring: ComponentSpec = coupler_ring,
    bend: ComponentSpec = "bend_euler",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns a double bus ring.

    two couplers (ct: top, cb: bottom)
    connected with two vertical straights (sl: left, sr: right)

    Args:
        gap: gap between for coupler.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        coupler_ring: ring coupler spec.
        bend: bend spec.
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
    gap = gf.snap.snap_to_grid(gap, grid_factor=2)

    coupler_component = gf.get_component(
        coupler_ring,
        gap=gap,
        radius=radius,
        length_x=length_x,
        bend=bend,
        cross_section=cross_section,
    )
    straight_component = straight(
        length=length_y,
        cross_section=cross_section,
    )

    c = Component()
    cb = c.add_ref(coupler_component)
    ct = c.add_ref(coupler_component)

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
    c = ring_double(length_y=2)
    c.show()
