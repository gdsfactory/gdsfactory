from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler_ring import coupler_ring
from gdsfactory.components.straight import straight
from gdsfactory.components.via_stack import via_stack_heater_m3
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Float2

via_stack_heater_m3_mini = partial(via_stack_heater_m3, size=(4, 4))


@gf.cell
def ring_double_heater(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 0.01,
    length_y: float = 0.01,
    coupler_ring: ComponentSpec = coupler_ring,
    coupler_ring_top: ComponentSpec | None = None,
    straight: ComponentSpec = straight,
    bend: ComponentSpec = bend_euler,
    cross_section_heater: CrossSectionSpec = "heater_metal",
    cross_section_waveguide_heater: CrossSectionSpec = "strip_heater_metal",
    cross_section: CrossSectionSpec = "strip",
    via_stack: ComponentSpec = via_stack_heater_m3_mini,
    port_orientation: float | None = None,
    via_stack_offset: Float2 = (0, 0),
    **kwargs,
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
        kwargs: cross_section settings.

    .. code::

         --==ct==--
          |      |
          sl     sr length_y
          |      |
         --==cb==-- gap

          length_x
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
        bend_cross_section=cross_section_waveguide_heater,
        **kwargs,
    )
    coupler_component_top = gf.get_component(
        coupler_ring_top,
        gap=gap,
        radius=radius,
        length_x=length_x,
        bend=bend,
        cross_section=cross_section,
        bend_cross_section=cross_section_waveguide_heater,
        **kwargs,
    )
    straight_component = gf.get_component(
        straight,
        length=length_y,
        cross_section=cross_section_waveguide_heater,
        **kwargs,
    )

    c = Component()
    cb = c.add_ref(coupler_component)
    ct = c.add_ref(coupler_component_top)
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

    p1 = c1.get_ports_list(orientation=port_orientation)
    p2 = c2.get_ports_list(orientation=port_orientation)
    valid_orientations = {p.orientation for p in via.ports.values()}

    if not p1:
        raise ValueError(
            f"No ports found for port_orientation {port_orientation} in {valid_orientations}"
        )

    c.add_ports(p1, prefix="l_")
    c.add_ports(p2, prefix="r_")

    heater_top = c << gf.get_component(
        straight,
        length=length_x,
        cross_section=cross_section_heater,
    )
    heater_top.connect("e1", ct.ports["e1"])
    return c


if __name__ == "__main__":
    pass
    # c1 = ring_double_heater(via_stack="via_stack")
    # c1.pprint_ports()

    # c2 = ring_double_heater(via_stack="via_stack_slot")
    # c2.pprint_ports()
    # c = ring_double_heater(width=1, layer=(2, 0), length_y=3)
    # c = ring_double_heater(
    #     length_x=0,
    #     port_orientation=90,
    #     bend=gf.components.bend_circular,
    #     via_stack_offset=(2, 0),
    #     coupler_ring_top=coupler_ring,
    #     coupler_ring=gf.partial(
    #         coupler_ring_point,
    #         coupler_ring=coupler_ring,
    #         open_layers=("HEATER",),
    #         open_sizes=((5, 7),),
    #     ),
    # )
    # c2.show(show_ports=True)
    # c.pprint()
