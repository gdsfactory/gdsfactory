from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler_ring import coupler_ring as _coupler_ring
from gdsfactory.components.straight import straight
from gdsfactory.components.via_stack import via_stack_heater_mtop
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Float2

via_stack_heater_mtop_mini = partial(via_stack_heater_mtop, size=(4, 4))


@gf.cell
def ring_single_heater(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.6,
    coupler_ring: ComponentSpec = _coupler_ring,
    bend: ComponentSpec = bend_euler,
    cross_section_waveguide_heater: CrossSectionSpec = "strip_heater_metal",
    cross_section: CrossSectionSpec = "strip",
    via_stack: ComponentSpec = via_stack_heater_mtop_mini,
    port_orientation: float | None = None,
    via_stack_offset: Float2 = (0, 0),
    **kwargs,
) -> gf.Component:
    """Returns a single ring with heater on top.

    ring coupler (cb: bottom) connects to two vertical straights (sl: left, sr: right),
    two bends (bl, br) and horizontal straight (wg: top)

    Args:
        gap: gap between for coupler.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        coupler_ring: ring coupler function.
        bend: 90 degrees bend function.
        cross_section_waveguide_heater: for heater.
        cross_section: for regular waveguide.
        via_stack: for heater to routing metal.
        port_orientation: for electrical ports to promote from via_stack.
        via_stack_offset: x,y offset for via_stack.
        kwargs: cross_section settings.

    .. code::

          bl-st-br
          |      |
          sl     sr length_y
          |      |
         --==cb==-- gap

          length_x
    """
    gap = gf.snap.snap_to_grid(gap, grid_factor=2)

    coupler_ring = gf.get_component(
        coupler_ring,
        bend=bend,
        gap=gap,
        radius=radius,
        length_x=length_x,
        cross_section=cross_section,
        bend_cross_section=cross_section_waveguide_heater,
        **kwargs,
    )

    straight_side = straight(
        length=length_y,
        cross_section=cross_section_waveguide_heater,
        **kwargs,
    )
    straight_top = straight(
        length=length_x,
        cross_section=cross_section_waveguide_heater,
        **kwargs,
    )

    bend = gf.get_component(
        bend, radius=radius, cross_section=cross_section_waveguide_heater, **kwargs
    )

    c = gf.Component()
    cb = c << coupler_ring
    sl = c << straight_side
    sr = c << straight_side
    bl = c << bend
    br = c << bend
    st = c << straight_top
    # st.mirror(p1=(0, 0), p2=(1, 0))

    sl.connect(port="o1", destination=cb.ports["o2"])
    bl.connect(port="o2", destination=sl.ports["o2"])

    st.connect(port="o2", destination=bl.ports["o1"])
    br.connect(port="o2", destination=st.ports["o1"])
    sr.connect(port="o1", destination=br.ports["o1"])
    sr.connect(port="o2", destination=cb.ports["o3"])

    c.add_port("o2", port=cb.ports["o4"])
    c.add_port("o1", port=cb.ports["o1"])

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
    return c


if __name__ == "__main__":
    c = ring_single_heater(width=0.5, gap=1, layer=(2, 0), radius=10, length_y=1)
    c.show(show_subports=False)
