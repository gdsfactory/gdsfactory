from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Port, Spacing


@gf.cell
def coh_rx_single_pol(
    bend: ComponentSpec = "bend_euler",
    cross_section: CrossSectionSpec = "strip",
    hybrid_90deg: ComponentSpec = gf.c.mmi_90degree_hybrid,
    detector: ComponentSpec = gf.c.ge_detector_straight_si_contacts,
    det_spacing: Spacing = (60.0, 50.0),
    in_wg_length: float = 20.0,
    lo_input_coupler: ComponentSpec | None = None,
    signal_input_coupler: ComponentSpec | None = None,
    cross_section_metal_top: CrossSectionSpec = "metal3",
    cross_section_metal: CrossSectionSpec = "metal2",
) -> Component:
    r"""Single polarization coherent receiver.

    Args:
        bend: 90 degrees bend library.
        cross_section: for routing.
        hybrid_90deg: generates the 90 degree hybrid.
        detector: generates the detector.
        det_spacing: spacing between 90 degree hybrid and detector and
           vertical spacing between detectors.
        in_wg_length: length of the straight waveguides at the input of the 90 deg hybrid.
        lo_input_coupler: Optional coupler for the LO.
        signal_input_coupler: Optional coupler for the signal.
        cross_section_metal_top: cross_section for the top metal layer.
        cross_section_metal: cross_section for the metal layer.
        cross_section: cross_section for the waveguides.

    .. code::

                               _________
           (lo_in_coupler)---|          |--- detI1 \\ __ i signal
                             |  90 deg  |--- detI2 //
       (signal_in_coupler)---|  hybrid  |--- detQ1 \\ __ q signal
                             |__________|--- detQ2 //
    """
    bend = gf.get_component(bend, cross_section=cross_section)

    # ----- Draw 90 deg hybrid -----

    c = Component()

    hybrid_90deg = gf.get_component(hybrid_90deg)
    hybrid = c << hybrid_90deg

    # ----- Draw input waveguides (and coupler if indicated) ---

    lo_in: ComponentReference | None = None
    signal_in: ComponentReference | None = None
    if in_wg_length > 0.0:
        straight = gf.components.straight(
            length=in_wg_length, cross_section=cross_section
        )
        signal_in = c << straight
        lo_in = c << straight

        signal_in.connect("o2", hybrid.ports["signal_in"])
        lo_in.connect("o2", hybrid.ports["LO_in"])

    if lo_input_coupler is not None:
        lo_in_coupler = gf.get_component(lo_input_coupler)
        in_coup_lo = c << lo_in_coupler

        if in_wg_length > 0.0 and lo_in is not None:
            in_coup_lo.connect("o1", lo_in.ports["o1"])
        else:
            in_coup_lo.connect("o1", hybrid.ports["LO_in"])

    elif in_wg_length > 0.0 and lo_in is not None:
        c.add_port("LO_in", port=lo_in.ports["o1"])
    else:
        c.add_port("LO_in", port=hybrid.ports["LO_in"])

    if signal_input_coupler is not None:
        signal_in_coupler = gf.get_component(signal_input_coupler)
        in_coup_signal = c << signal_in_coupler

        if in_wg_length > 0.0 and signal_in is not None:
            in_coup_signal.connect("o1", signal_in.ports["o1"])
        else:
            in_coup_signal.connect("o1", hybrid.ports["signal_in"])

    elif in_wg_length > 0.0 and signal_in is not None:
        c.add_port("signal_in", port=signal_in.ports["o1"])
    else:
        c.add_port("signal_in", port=hybrid.ports["signal_in"])

    # ---- Draw photodetectors -----

    detector = gf.get_component(detector)

    pd_i1 = c << detector
    pd_i2 = c << detector
    pd_q1 = c << detector
    pd_q2 = c << detector

    pds = [pd_i1, pd_i2, pd_q1, pd_q2]

    # x placement
    for pd in pds:
        pd.dxmin = hybrid.dxmax + det_spacing[0]

    # y placement - we will place them in the same order as the outputs
    # of the 90 degree hybrid to avoid crossings
    hybrid_ports = {"I_out1": pd_i1, "I_out2": pd_i2, "Q_out1": pd_q1, "Q_out2": pd_q2}

    port_names = list(hybrid_ports.keys())
    ports_y_pos = [hybrid.ports[port_name].dy for port_name in port_names]
    inds = np.argsort(ports_y_pos)
    port_names = [port_names[int(i)] for i in inds]

    y_pos = hybrid.dy - 1.5 * det_spacing[1]

    det_ports: list[Port] = []
    ports_hybrid: list[Port] = []
    for port_name in port_names:
        det = hybrid_ports[port_name]
        det.dy = y_pos
        y_pos = y_pos + det_spacing[1]
        det_ports.append(det.ports["o1"])
        ports_hybrid.append(hybrid.ports[port_name])

    gf.routing.route_bundle(c, ports_hybrid, det_ports, cross_section=cross_section)

    # --- Draw metal connections ----
    gf.routing.route_single_electrical(
        c,
        pd_i1.ports["bot_e3"],
        pd_i2.ports["top_e3"],
        cross_section=cross_section_metal_top,
    )

    # Add a port at the center
    x_max = c.dxmax
    c.add_port(
        name="i_out",
        port_type="placement",
        layer="MTOP",
        center=(x_max, (pd_i1.ports["bot_e3"].dy + pd_i2.ports["top_e3"].dy) / 2),
        orientation=0,
        width=2.0,
    )

    gf.routing.route_single_electrical(
        c,
        pd_q1.ports["bot_e3"],
        pd_q2.ports["top_e3"],
        cross_section=cross_section_metal,
    )

    # Add a port
    x_max = c.dxmax
    c.add_port(
        name="q_out",
        port_type="placement",
        layer="M2",
        center=(
            x_max,
            (pd_q1.ports["bot_e3"].dy + pd_q2.ports["top_e3"].dy) / 2 - 15.0,
        ),  # - 20.0 so that the traces for I and Q do not overlap
        orientation=0,
        width=2.0,
    )

    # Create electrical ports. q_out and i_out already exist
    c.add_ports(
        gf.port.get_ports_list(pd_i1, port_type="electrical", prefix="top"),
        prefix="i1vminus",
    )
    c.add_ports(
        gf.port.get_ports_list(pd_q1, port_type="electrical", prefix="top"),
        prefix="q1vminus",
    )
    c.add_ports(
        gf.port.get_ports_list(pd_q2, port_type="electrical", prefix="bot"),
        prefix="q2vplus",
    )
    c.add_ports(
        gf.port.get_ports_list(pd_i2, port_type="electrical", prefix="bot"),
        prefix="i2vplus",
    )

    return c


if __name__ == "__main__":
    c = coh_rx_single_pol()
    c.show()
