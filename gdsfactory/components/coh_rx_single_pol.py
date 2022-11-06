from typing import Optional, Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.ge_detector_straight_si_contacts import (
    ge_detector_straight_si_contacts,
)
from gdsfactory.components.mmi_90degree_hybrid import mmi_90degree_hybrid
from gdsfactory.types import ComponentSpec, CrossSectionSpec


@cell
def coh_rx_single_pol(
    bend: ComponentSpec = "bend_euler",
    cross_section: CrossSectionSpec = "strip",
    hybrid_90deg: ComponentSpec = mmi_90degree_hybrid,
    detector: ComponentSpec = ge_detector_straight_si_contacts,
    det_spacing: Tuple[float, float] = (60.0, 50.0),
    with_pads: bool = True,
    pad_det_spacing: float = 80.0,
    in_wg_length: float = 20.0,
    lo_input_coupler: Optional[ComponentSpec] = None,
    signal_input_coupler: Optional[ComponentSpec] = None,
) -> Component:
    r"""Single polarization coherent receiver.

    Args:
        bend: 90 degrees bend library.
        cross_section: for routing.
        hybrid_90deg: generates the 90 degree hybrid.
        detector: generates the detector.
        det_spacing: spacing between 90 degree hybrid and detector and
           vertical spacing between detectors.
        with_pads: if True, it draws pads for the balanced detectors.
        pad_det_spacing: spacing between the pads and the detectors (if with_pads=True).
        in_wg_length: length of the straight waveguides at the input of the 90 deg hybrid.
        lo_input_coupler: Optional coupler for the LO.
        signal_input_coupler: Optional coupler for the signal.

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

        if in_wg_length > 0.0:
            in_coup_lo.connect("o1", lo_in.ports["o1"])
        else:
            in_coup_lo.connect("o1", hybrid.ports["LO_in"])

    elif in_wg_length > 0.0:
        c.add_port("LO_in", port=lo_in.ports["o1"])
    else:
        c.add_port("LO_in", port=hybrid.ports["LO_in"])

    if signal_input_coupler is not None:
        signal_in_coupler = gf.get_component(signal_input_coupler)
        in_coup_signal = c << signal_in_coupler

        if in_wg_length > 0.0:
            in_coup_signal.connect("o1", signal_in.ports["o1"])
        else:
            in_coup_signal.connect("o1", hybrid.ports["signal_in"])

    elif in_wg_length > 0.0:
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
        pd.xmin = hybrid.xmax + det_spacing[0]

    # y placement - we will place them in the same order as the outputs
    # of the 90 degree hybrid to avoid crossings
    hybrid_ports = {"I_out1": pd_i1, "I_out2": pd_i2, "Q_out1": pd_q1, "Q_out2": pd_q2}

    port_names = hybrid_ports.keys()
    ports_y_pos = [hybrid.ports[port_name].y for port_name in port_names]
    inds = np.argsort(ports_y_pos)
    port_names = list(port_names)
    port_names = [port_names[i] for i in inds]

    y_pos = hybrid.y - 1.5 * det_spacing[1]

    det_ports = []
    ports_hybrid = []
    for port_name in port_names:
        det = hybrid_ports[port_name]
        det.y = y_pos
        y_pos = y_pos + det_spacing[1]
        det_ports.append(det.ports["o1"])
        ports_hybrid.append(hybrid.ports[port_name])

    routes = gf.routing.get_bundle(ports_hybrid, det_ports)
    for route in routes:
        c.add(route.references)

    # --- Draw metal connections ----

    route = gf.routing.get_route_electrical(
        pd_i1.ports["bot_e3"], pd_i2.ports["top_e3"]
    )
    c.add(route.references)

    # Add a port at the center
    x_max = -np.inf
    for ref in route.references:
        if ref.xmax > x_max:
            x_max = ref.xmax
    c.add_port(
        name="i_out",
        port_type="placement",
        layer="M3",
        center=(x_max, (pd_i1.ports["bot_e3"].y + pd_i2.ports["top_e3"].y) / 2),
        orientation=0,
        width=2.0,
    )

    route = gf.routing.get_route_electrical_m2(
        pd_q1.ports["bot_e3"], pd_q2.ports["top_e3"]
    )
    c.add(route.references)

    # Add a port
    x_max = -np.inf
    for ref in route.references:
        if ref.xmax > x_max:
            x_max = ref.xmax
    c.add_port(
        name="q_out",
        port_type="placement",
        layer="M2",
        center=(
            x_max,
            (pd_q1.ports["bot_e3"].y + pd_q2.ports["top_e3"].y) / 2 - 15.0,
        ),  # - 20.0 so that the traces for I and Q do not overlap
        orientation=0,
        width=2.0,
    )

    # --- Draw pads if indicated ----

    if with_pads:

        pad_array = c << gf.components.pad_array(columns=1, rows=4, orientation=0)
        pad_array.xmin = pd_i1.xmax + pad_det_spacing
        pad_array.y = hybrid.y

        # Add labels
        labels = {"e11": "V+", "e21": "Q_out", "e31": "I_out", "e41": "V-"}
        for pad, label in labels.items():
            x_pos, y_pos = pad_array.ports[pad].center
            c << gf.components.text(
                text=label,
                size=14.0,
                position=[x_pos + 55.0, y_pos],
                justify="left",
                layer="M3",
            )

        # Connect to the pads. Need to do it manually to avoid crossings

        # V- pad (connected to positive side of one of the diodes)
        p0x, p0y = pd_i1.ports["top_e2"].center
        p1x, p1y = pad_array.ports["e41"].center
        route = gf.routing.get_route_from_waypoints_electrical(
            [(p0x, p0y), (p0x, p1y), (p1x, p1y)]
        )

        c.add(route.references)

        p0x, p0y = pd_q1.ports["top_e3"].center
        p1x, p1y = pad_array.ports["e41"].center
        route = gf.routing.get_route_from_waypoints_electrical_multilayer(
            [
                (p0x, p0y),
                (p0x + 0.5 * (p1x - p0x), p0y),
                (p0x + 0.5 * (p1x - p0x), p1y),
            ]
        )

        c.add(route.references)

        # V+ pad (connected to negative side of the other diode)
        p0x, p0y = pd_i2.ports["bot_e2"].center
        p1x, p1y = pad_array.ports["e11"].center
        route = gf.routing.get_route_from_waypoints_electrical(
            [(p0x, p0y), (p0x, p1y), (p1x, p1y)]
        )
        c.add(route.references)

        p0x, p0y = pd_q2.ports["bot_e3"].center
        p1x, p1y = pad_array.ports["e11"].center
        route = gf.routing.get_route_from_waypoints_electrical_multilayer(
            [
                (p0x, p0y),
                (p0x + 0.5 * (p1x - p0x), p0y),
                (p0x + 0.5 * (p1x - p0x), p1y),
            ]
        )

        c.add(route.references)

        # I out pad
        route = gf.routing.get_route_electrical(
            pad_array.ports["e31"], c.ports["i_out"]
        )
        c.add(route.references)

        # Q out pad
        route = gf.routing.get_route_electrical_multilayer(
            pad_array.ports["e21"], c.ports["q_out"]
        )
        c.add(route.references)

    else:
        # Create electrical ports. q_out and i_out already exist
        c.add_ports(
            pd_i1.get_ports_list(port_type="electrical", prefix="top"),
            prefix="i1vminus",
        )
        c.add_ports(
            pd_q1.get_ports_list(port_type="electrical", prefix="top"),
            prefix="q1vminus",
        )
        c.add_ports(
            pd_q2.get_ports_list(port_type="electrical", prefix="bot"), prefix="q2vplus"
        )
        c.add_ports(
            pd_i2.get_ports_list(port_type="electrical", prefix="bot"), prefix="i2vplus"
        )

    return c


if __name__ == "__main__":
    c = coh_rx_single_pol(with_pads=False)
    c.show(show_ports=True)
