from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler_ring import coupler_ring as coupler_ring_function
from gdsfactory.components.straight import straight
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def ring_double(
    gap: float = 0.2,
    gap_top: float | None = None,
    radius: float = 10.0,
    length_x: float = 0.01,
    length_y: float = 0.01,
    coupler_ring: ComponentSpec = coupler_ring_function,
    bend: ComponentSpec = bend_euler,
    straight: ComponentSpec = straight,
    cross_section: CrossSectionSpec = "xs_sc",
) -> Component:
    """Returns a double bus ring.

    two couplers (ct: top, cb: bottom)
    connected with two vertical straights (sl: left, sr: right)

    Args:
        gap: gap between for coupler.
        gap_top: optional gap between top waveguides. Defaults to gap.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        coupler: ring coupler spec.
        bend: bend spec.
        straight: straight spec.
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
    gap = gf.snap.snap_to_grid(gap, grid_factor=2)
    gap_top = gf.snap.snap_to_grid(gap_top, grid_factor=2)

    xs = gf.get_cross_section(cross_section)
    radius = radius or xs.radius
    cross_section = xs.copy(radius=radius)

    coupler_component_top = gf.get_component(
        coupler_ring,
        gap=gap_top,
        radius=radius,
        length_x=length_x,
        bend=bend,
        cross_section=cross_section,
    )
    coupler_component_bot = gf.get_component(
        coupler_ring,
        gap=gap,
        radius=radius,
        length_x=length_x,
        bend=bend,
        cross_section=cross_section,
    )
    straight_component = straight(length=length_y, cross_section=cross_section)

    c = Component()
    cb = c.add_ref(coupler_component_bot)
    ct = c.add_ref(coupler_component_top)

    sl = straight_component.ref()
    sr = straight_component.ref()

    if length_y > 0:
        c.add(sl)
        c.add(sr)

    sl.connect(port="o1", destination=cb.ports["o2"])
    ct.connect(port="o3", destination=sl.ports["o2"])
    sr.connect(port="o2", destination=ct.ports["o2"])
    c.add_port("o1", port=cb.ports["o1"])
    c.add_port("o2", port=cb.ports["o4"])
    c.add_port("o3", port=ct.ports["o4"])
    c.add_port("o4", port=ct.ports["o1"])
    return c


if __name__ == "__main__":
    c = ring_double(length_x=10, radius=15, length_y=5)
    c.get_netlist()
    c.show(show_subports=False)
