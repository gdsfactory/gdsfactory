from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name
def mzi_pads_center(
    ps_top: ComponentSpec = "straight_heater_metal",
    ps_bot: ComponentSpec = "straight_heater_metal",
    mzi: ComponentSpec = "mzi",
    pad: ComponentSpec = "pad_small",
    length_x: float = 500,
    length_y: float = 40,
    mzi_sig_top: str | None = "top_r_e2",
    mzi_gnd_top: str | None = "top_l_e2",
    mzi_sig_bot: str | None = "bot_l_e2",
    mzi_gnd_bot: str | None = "bot_r_e2",
    pad_sig_bot: str = "e1_1_1",
    pad_sig_top: str = "e3_1_3",
    pad_gnd_bot: str = "e4_1_2",
    pad_gnd_top: str = "e2_1_2",
    delta_length: float = 40.0,
    cross_section: CrossSectionSpec = "strip",
    cross_section_metal: CrossSectionSpec = "metal_routing",
    pad_pitch: float | str = "pad_pitch",
    **kwargs: Any,
) -> gf.Component:
    """Return Mzi phase shifter with pads in the middle.

    GND is the middle pad and is shared between top and bottom phase shifters.

    Args:
        ps_top: phase shifter top.
        ps_bot: phase shifter bottom.
        mzi: interferometer.
        pad: pad function.
        length_x: horizontal length.
        length_y: vertical length.
        mzi_sig_top: port name for top phase shifter signal. None if no connection.
        mzi_gnd_top: port name for top phase shifter GND. None if no connection.
        mzi_sig_bot: port name for top phase shifter signal. None if no connection.
        mzi_gnd_bot: port name for top phase shifter GND. None if no connection.
        pad_sig_bot: port name for top pad.
        pad_sig_top: port name for top pad.
        pad_gnd_bot: port name for top pad.
        pad_gnd_top: port name for top pad.
        delta_length: mzi length imbalance.
        cross_section: for the mzi.
        cross_section_metal: for routing metal.
        pad_pitch: pad pitch in um.
        kwargs: routing settings.
    """
    c = gf.Component()

    pad_pitch = gf.get_constant(pad_pitch)

    assert isinstance(pad_pitch, float)

    mzi_ps = gf.get_component(
        mzi,
        length_x=length_x,
        straight_x_top=ps_top,
        straight_x_bot=ps_bot,
        length_y=length_y,
        delta_length=delta_length,
        cross_section=cross_section,
        auto_rename_ports=False,
    )

    port_names = [p.name for p in mzi_ps.ports]
    for port_name in [mzi_sig_top, mzi_gnd_top, mzi_sig_bot, mzi_gnd_bot]:
        if port_name and port_name not in port_names:
            raise ValueError(f"port {port_name!r} not in {port_names}")

    m = c << mzi_ps
    pads = c << gf.components.array(
        component=pad, columns=3, rows=1, column_pitch=pad_pitch
    )
    pads.x = m.x
    pads.y = m.y

    if mzi_sig_top is not None:
        gf.routing.route_single_electrical(
            c,
            m.ports[mzi_sig_bot],
            pads.ports[pad_sig_bot],
            cross_section=cross_section_metal,
            **kwargs,
        )

    if mzi_gnd_bot:
        gf.routing.route_single_electrical(
            c,
            m.ports[mzi_gnd_bot],
            pads.ports[pad_gnd_bot],
            cross_section=cross_section_metal,
            **kwargs,
        )

    if mzi_gnd_top:
        gf.routing.route_single_electrical(
            c,
            m.ports[mzi_gnd_top],
            pads.ports[pad_gnd_top],
            cross_section=cross_section_metal,
            **kwargs,
        )

    if mzi_sig_top:
        gf.routing.route_single_electrical(
            c,
            m.ports[mzi_sig_top],
            pads.ports[pad_sig_top],
            cross_section=cross_section_metal,
            **kwargs,
        )

    c.add_ports(m.ports)
    return c


if __name__ == "__main__":
    # mzi_ps_fa = gf.compose(gf.routing.add_fiber_array, mzi_pads_center)
    # mzi_ps_fs = gf.compose(gf.routing.add_fiber_single, mzi_pads_center)
    # c = mzi_ps_fs()
    c = mzi_pads_center()
    c.show()
