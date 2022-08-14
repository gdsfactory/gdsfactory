from typing import Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.coupler_ring import coupler_ring as coupler_ring_function
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.via_stack import via_stack_heater_m3
from gdsfactory.cross_section import strip
from gdsfactory.types import ComponentSpec, CrossSectionSpec, Float2

via_stack_heater_m3_mini = gf.partial(via_stack_heater_m3, size=(4, 4))


@gf.cell
def ring_double_heater(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 0.01,
    length_y: float = 0.01,
    coupler_ring: ComponentSpec = coupler_ring_function,
    straight: ComponentSpec = straight_function,
    bend: Optional[ComponentSpec] = None,
    cross_section_heater: CrossSectionSpec = gf.cross_section.heater_metal,
    cross_section_waveguide_heater: CrossSectionSpec = gf.cross_section.strip_heater_metal,
    cross_section: CrossSectionSpec = strip,
    via_stack: gf.types.ComponentSpec = via_stack_heater_m3_mini,
    port_orientation: float = 90,
    via_stack_offset: Float2 = (0, 0),
    **kwargs
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
        straight: straight spec.
        bend: bend spec.
        cross_section_heater: for heater.
        cross_section_waveguide_heater: for waveguide with heater.
        cross_section: for regular waveguide.
        via_stack: for heater to routing metal.
        port_orientation: for electrical ports to promote from via_stack.
        via_stack_offset: x,y offset for via_stack.
        kwargs: cross_section settings.

    .. code::

         --==ct==--
          |      |
          sl     sr length_y
          |      |
         --==cb==-- gap

          length_x
    """
    gap = gf.snap.snap_to_grid(gap, nm=2)

    coupler_component = gf.get_component(
        coupler_ring,
        gap=gap,
        radius=radius,
        length_x=length_x,
        bend=bend,
        cross_section=cross_section,
        bend_cross_section=cross_section_waveguide_heater,
        **kwargs
    )
    straight_component = gf.get_component(
        straight,
        length=length_y,
        cross_section=cross_section_waveguide_heater,
        **kwargs
    )

    c = Component()
    cb = c.add_ref(coupler_component)
    ct = c.add_ref(coupler_component)
    sl = c.add_ref(straight_component)
    sr = c.add_ref(straight_component)

    sl.connect(port="o1", destination=cb.ports["o2"])
    ct.connect(port="o3", destination=sl.ports["o2"])
    sr.connect(port="o2", destination=ct.ports["o2"])
    c.add_port("o1", port=cb.ports["o1"])
    c.add_port("o2", port=cb.ports["o4"])
    c.add_port("o3", port=ct.ports["o4"])
    c.add_port("o4", port=ct.ports["o1"])

    via = gf.get_component(via_stack)
    c1 = c << via
    c2 = c << via
    c1.xmax = -length_x / 2 + cb.x - via_stack_offset[0]
    c2.xmin = +length_x / 2 + cb.x + via_stack_offset[0]
    c1.movey(via_stack_offset[1])
    c2.movey(via_stack_offset[1])
    c.add_ports(c1.get_ports_list(orientation=port_orientation), prefix="e1")
    c.add_ports(c2.get_ports_list(orientation=port_orientation), prefix="e2")

    heater_top = c << gf.get_component(
        straight,
        length=length_x,
        cross_section=cross_section_heater,
    )
    heater_top.connect("e1", ct.ports["e1"])

    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    # c = ring_double_heater(width=1, layer=(2, 0), length_y=3)
    c = ring_double_heater(length_x=5)
    c.show(show_ports=True)
    # c.pprint()
