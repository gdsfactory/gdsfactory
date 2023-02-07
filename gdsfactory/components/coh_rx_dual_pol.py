from __future__ import annotations

from typing import Optional

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coh_rx_single_pol import coh_rx_single_pol
from gdsfactory.routing.get_route import get_route, get_route_from_waypoints
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@cell
def coh_rx_dual_pol(
    bend: ComponentSpec = bend_euler,
    cross_section: CrossSectionSpec = "strip",
    lo_splitter: ComponentSpec = "mmi1x2",
    signal_splitter: Optional[ComponentSpec] = None,
    spol_coh_rx: ComponentSpec = coh_rx_single_pol,
    single_pol_rx_spacing: float = 50.0,
    splitter_coh_rx_spacing: float = 40.0,
    lo_input_coupler: Optional[ComponentSpec] = None,
    signal_input_coupler: Optional[ComponentSpec] = None,
) -> Component:
    """Dual polarization coherent receiver.

    Args:
        bend: 90 degrees bend library.
        cross_section: for routing (splitter to mzms and mzms to combiners).
        lo_splitter: splitter function for the LO input.
        signal_splitter: splitter function for the signal input.
        spol_coh_rx: function generating a coherent rx for a single polarization.
        single_pol_rx_spacing: vertical spacing between each single polarization coherent receiver.
        splitter_coh_rx_spacing: horizontal spacing between the signal splitter and the single pol coh rxs.
        lo_input_coupler: Optional coupler to add before the LO splitter.
        signal_input_coupler: Optional coupler to add before the signal splitter.
    """
    bend_spec = bend
    bend = gf.get_component(bend, cross_section=cross_section)

    spol_coh_rx = gf.get_component(spol_coh_rx)

    # ----- Draw single pol coherent receivers -----

    c = Component()

    single_rx_1 = c << spol_coh_rx
    single_rx_2 = c << spol_coh_rx

    single_rx_1.mirror((1, 0))

    # Separate the two receivers
    single_rx_2.movey(single_rx_1.ymin - single_pol_rx_spacing - single_rx_2.ymax)

    # ------------ Signal splitters and input coupler ---------------

    if signal_splitter is not None:
        signal_splitter = gf.get_component(signal_splitter)
        signal_spl = c << signal_splitter
        signal_spl.xmax = single_rx_1.xmin - splitter_coh_rx_spacing
        signal_spl.y = (single_rx_1.y + single_rx_2.y) / 2

        route = get_route(
            signal_spl.ports["o2"],
            single_rx_1.ports["signal_in"],
            bend=bend_spec,
            cross_section=cross_section,
            with_sbend=False,
        )
        c.add(route.references)

        route = get_route(
            signal_spl.ports["o3"],
            single_rx_2.ports["signal_in"],
            bend=bend_spec,
            cross_section=cross_section,
            with_sbend=False,
        )
        c.add(route.references)

    if signal_input_coupler is not None:
        signal_coupler = gf.get_component(signal_input_coupler)
        signal_coup = c << signal_coupler
        signal_coup.mirror((0, 1))
        if signal_splitter is not None:
            signal_coup.connect("o1", signal_spl.ports["o1"])

        else:
            signal_coup.xmax = single_rx_1.xmin - splitter_coh_rx_spacing
            signal_coup.y = (single_rx_1.y + single_rx_2.y) / 2

            route = get_route(
                signal_coup.ports["o1"],
                single_rx_1.ports["signal_in"],
                bend=bend_spec,
                cross_section=cross_section,
                with_sbend=False,
            )
            c.add(route.references)

            route = get_route(
                signal_coup.ports["o2"],
                single_rx_2.ports["signal_in"],
                bend=bend_spec,
                cross_section=cross_section,
                with_sbend=False,
            )
            c.add(route.references)

    # ------------ LO splitter and input coupler ---------------

    lo_splitter = gf.get_component(lo_splitter)
    lo_split = c << lo_splitter

    if signal_input_coupler is not None:
        xlim = signal_coup.xmin
    elif signal_splitter is not None:
        xlim = signal_spl.xmin
    else:
        xlim = single_rx_1.xmin

    lo_split.xmax = xlim - splitter_coh_rx_spacing
    lo_split.y = (single_rx_1.y + single_rx_2.y) / 2

    p0x, p0y = lo_split.ports["o2"].center
    p1x, p1y = single_rx_1.ports["LO_in"].center

    route = get_route_from_waypoints(
        [
            (p0x, p0y),
            (p0x + splitter_coh_rx_spacing / 4, p0y),
            (p0x + splitter_coh_rx_spacing / 4, p1y),
            (p1x, p1y),
        ],
        bend=bend_spec,
        cross_section=cross_section,
    )
    c.add(route.references)

    p0x, p0y = lo_split.ports["o3"].center
    p1x, p1y = single_rx_2.ports["LO_in"].center

    route = get_route_from_waypoints(
        [
            (p0x, p0y),
            (p0x + splitter_coh_rx_spacing / 4, p0y),
            (p0x + splitter_coh_rx_spacing / 4, p1y),
            (p1x, p1y),
        ],
        bend=bend_spec,
        cross_section=cross_section,
    )
    c.add(route.references)

    if lo_input_coupler is not None:
        lo_coupler = gf.get_component(lo_input_coupler)
        lo_coup = c << lo_coupler
        lo_coup.connect("o1", lo_split.ports["o1"])

    # ------ Extract electrical ports (if no pads) -------
    c.add_ports(single_rx_1.get_ports_list(port_type="electrical"), prefix="pol1")
    c.add_ports(single_rx_2.get_ports_list(port_type="electrical"), prefix="pol2")
    c.auto_rename_ports()

    return c


if __name__ == "__main__":
    # c = coh_rx_dual_pol()
    c = coh_rx_single_pol()
    c.show(show_ports=True)
