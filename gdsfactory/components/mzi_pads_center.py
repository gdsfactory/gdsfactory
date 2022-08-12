import gdsfactory as gf
from gdsfactory.components.mzi import mzi as mzi_function
from gdsfactory.components.pad import pad as pad_function
from gdsfactory.components.straight_heater_metal import straight_heater_metal
from gdsfactory.cross_section import metal3, strip
from gdsfactory.routing.get_route import get_route
from gdsfactory.types import ComponentSpec, CrossSectionSpec


@gf.cell
def mzi_pads_center(
    ps_top: ComponentSpec = straight_heater_metal,
    ps_bot: ComponentSpec = straight_heater_metal,
    mzi: ComponentSpec = mzi_function,
    pad: ComponentSpec = pad_function,
    length_x: float = 500,
    length_y: float = 40,
    mzi_sig_top: str = "e3",
    mzi_gnd_top: str = "e2",
    mzi_sig_bot: str = "e1",
    mzi_gnd_bot: str = "e4",
    pad_sig_bot: str = "e1_1_1",
    pad_sig_top: str = "e3_1_3",
    pad_gnd_bot: str = "e4_1_2",
    pad_gnd_top: str = "e2_1_2",
    delta_length: float = 40.0,
    end_straight_length: float = 5,
    start_straight_length: float = 5,
    metal_route_width: float = 10,
    cross_section: CrossSectionSpec = strip,
    cross_section_metal: ComponentSpec = metal3,
) -> gf.Component:
    """Return Mzi phase shifter with pads in the middle.

    GND is the middle pad
    and is shared between top and bottom phase shifters.

    Args:
        ps_top: phase shifter top.
        ps_bot: phase shifter bottom.
        mzi: interforemeter.
        pad: pad function.
        length_x: horizontal length.
        length_y: vertical length.
        mzi_sig_top: port name for top phase shifter signal.
        mzi_gnd_top: port name for top phase shifter GND.
        mzi_sig_bot: port name for top phase shifter signal.
        mzi_gnd_bot: port name for top phase shifter GND.
        pad_sig_bot: port name for top pad.
        pad_sig_top: port name for top pad.
        pad_gnd_bot: port name for top pad.
        pad_gnd_top: port name for top pad.
        delta_length: mzi length imbalance.
        end_straight_length: for routing metal.
        start_straight_length: for routing metal.
        metal_route_width: for routing metal.
        cross_section: for the mzi.
        cross_section_metal: for routing metal.
    """
    c = gf.Component()

    mzi_ps = mzi(
        length_x=length_x,
        straight_x_top=ps_top,
        straight_x_bot=ps_bot,
        length_y=length_y,
        delta_length=delta_length,
        cross_section=cross_section,
    )

    port_names = list(mzi_ps.ports.keys())
    for port_name in [mzi_sig_top, mzi_gnd_top, mzi_sig_bot, mzi_gnd_bot]:
        if port_name not in port_names:
            raise ValueError(f"port {port_name!r} not in {port_names}")

    m = c << mzi_ps
    pads = c << gf.components.array(component=pad, columns=3, rows=1)
    pads.x = m.x
    pads.y = m.y

    route_sig_bot = get_route(
        m.ports[mzi_sig_bot],
        pads.ports[pad_sig_bot],
        cross_section=cross_section_metal,
        bend=gf.components.wire_corner,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        width=metal_route_width,
    )
    c.add(route_sig_bot.references)

    route_gnd_bot = get_route(
        m.ports[mzi_gnd_bot],
        pads.ports[pad_gnd_bot],
        cross_section=cross_section_metal,
        bend=gf.components.wire_corner,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        width=metal_route_width,
    )
    c.add(route_gnd_bot.references)
    route_gnd_top = get_route(
        m.ports[mzi_gnd_top],
        pads.ports[pad_gnd_top],
        cross_section=cross_section_metal,
        bend=gf.components.wire_corner,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        width=metal_route_width,
    )
    c.add(route_gnd_top.references)

    route_sig_top = get_route(
        m.ports[mzi_sig_top],
        pads.ports[pad_sig_top],
        cross_section=cross_section_metal,
        bend=gf.components.wire_corner,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        width=metal_route_width,
    )
    c.add(route_sig_top.references)
    c.add_ports(m.ports)
    return c


if __name__ == "__main__":
    mzi_ps_fa = gf.compose(gf.routing.add_fiber_array, mzi_pads_center)
    mzi_ps_fs = gf.compose(gf.routing.add_fiber_single, mzi_pads_center)
    # c = mzi_ps_fs()
    c = mzi_pads_center()
    c.show(show_ports=True)
