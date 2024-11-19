from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import AngleInDegrees, ComponentSpec, CrossSectionSpec, Float2


@gf.cell
def ring_double_heater(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 1.0,
    length_y: float = 0.01,
    coupler_ring: ComponentSpec = "coupler_ring",
    coupler_ring_top: ComponentSpec | None = None,
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_euler",
    cross_section_heater: CrossSectionSpec = "heater_metal",
    cross_section_waveguide_heater: CrossSectionSpec = "strip_heater_metal",
    cross_section: CrossSectionSpec = "strip",
    via_stack: ComponentSpec = "via_stack_heater_mtop_mini",
    port_orientation: AngleInDegrees | None = None,
    via_stack_offset: Float2 = (1, 0),
    with_drop: bool = True,
) -> Component:
    """Returns a double bus ring with heater on top.

    two couplers (ct: top, cb: bottom)
    connected with two vertical straights (sl: left, sr: right)

    Args:
        gap: gap between for coupler.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        coupler_ring: ring coupler spec.
        coupler_ring_top: ring coupler spec for coupler away from vias (defaults to coupler_ring)
        straight: straight spec.
        bend: bend spec.
        cross_section_heater: for heater.
        cross_section_waveguide_heater: for waveguide with heater.
        cross_section: for regular waveguide.
        via_stack: for heater to routing metal.
        port_orientation: for electrical ports to promote from via_stack.
        via_stack_offset: x,y offset for via_stack.
        with_drop: adds drop ports.

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

    coupler_ring_top = coupler_ring_top or coupler_ring

    coupler_component = gf.get_component(
        coupler_ring,
        gap=gap,
        radius=radius,
        length_x=length_x,
        bend=bend,
        cross_section=cross_section,
        cross_section_bend=cross_section_waveguide_heater,
    )
    coupler_component_top = gf.get_component(
        coupler_ring_top,
        gap=gap,
        radius=radius,
        length_x=length_x,
        bend=bend,
        cross_section=cross_section,
        cross_section_bend=cross_section_waveguide_heater,
    )
    straight_component = gf.get_component(
        straight,
        length=length_y,
        cross_section=cross_section_waveguide_heater,
    )

    c = Component()

    cb = c.add_ref(coupler_component)
    sl = c.add_ref(straight_component)
    sr = c.add_ref(straight_component)
    c.add_port("o1", port=cb.ports["o1"])
    c.add_port("o2", port=cb.ports["o4"])

    if with_drop:
        ct = c.add_ref(coupler_component_top)
        sl.connect(port="o1", other=cb.ports["o2"])
        ct.connect(port="o3", other=sl.ports["o2"])
        sr.connect(port="o2", other=ct.ports["o2"])
        c.add_port("o3", port=ct.ports["o4"])
        c.add_port("o4", port=ct.ports["o1"])
        heater_top = c << gf.get_component(
            straight,
            length=length_x,
            cross_section=cross_section_heater,
        )
        heater_top.connect("e1", ct["e1"])

    else:
        straight_top = gf.get_component(
            straight,
            length=length_x,
            cross_section=cross_section_waveguide_heater,
        )
        bend = gf.get_component(
            bend,
            radius=radius,
            cross_section=cross_section_waveguide_heater,
        )
        bl = c << bend
        br = c << bend
        st = c << straight_top

        sl.connect(port="o1", other=cb.ports["o2"])
        bl.connect(port="o2", other=sl.ports["o2"])

        st.connect(port="o2", other=bl.ports["o1"])
        br.connect(port="o2", other=st.ports["o1"])
        sr.connect(port="o1", other=br.ports["o1"])
        sr.connect(port="o2", other=cb.ports["o3"])

    via = gf.get_component(via_stack)
    c1 = c << via
    c2 = c << via
    c1.dxmax = -length_x / 2 + cb.dx - via_stack_offset[0]
    c2.dxmin = +length_x / 2 + cb.dx + via_stack_offset[0]
    c1.dmovey(via_stack_offset[1])
    c2.dmovey(via_stack_offset[1])

    p1 = c1.ports.filter(orientation=port_orientation)
    p2 = c2.ports.filter(orientation=port_orientation)
    valid_orientations = {p.orientation for p in via.ports}

    if not p1:
        raise ValueError(
            f"No ports found for port_orientation {port_orientation} in {valid_orientations}"
        )

    c.add_ports(p1, prefix="l_")
    c.add_ports(p2, prefix="r_")
    return c


ring_single_heater = partial(ring_double_heater, with_drop=False)


if __name__ == "__main__":
    c = ring_single_heater()
    c.pprint_ports()
    c.show()
