from functools import partial
from typing import Optional

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coh_tx_single_pol import coh_tx_single_pol
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.routing.get_route import get_route
from gdsfactory.types import ComponentSpec, CrossSectionSpec

default_single_pol_coh_tx = partial(coh_tx_single_pol)


@cell
def coh_tx_dual_pol(
    bend: ComponentSpec = bend_euler,
    straight: ComponentSpec = straight_function,
    cross_section: CrossSectionSpec = "strip",
    splitter: ComponentSpec = "mmi1x2",
    combiner: Optional[ComponentSpec] = None,
    spol_coh_tx: ComponentSpec = default_single_pol_coh_tx,
    single_pol_tx_spacing: float = 10.0,
    splitter_coh_tx_spacing: float = 40.0,
    input_coupler: Optional[ComponentSpec] = None,
    output_coupler: Optional[ComponentSpec] = None,
) -> Component:
    """Dual polarization coherent transmitter.

    Args:
        bend: 90 degrees bend library.
        straight: straight function.
        cross_section: for routing (splitter to mzms and mzms to combiners).
        splitter: splitter function.
        combiner: combiner function.
        spol_coh_tx: function generating a coherent tx for a single polarization
        single_pol_coh_tx_spacing: vertical spacing between each single polarization coherent transmitter
        splitter_coh_tx_spacing: horizontal spacing between the splitter and combiner and the single pol coh txs
        input_coupler: Optional coupler to add before the splitter
        output_coupler: Optioncal coupler to add after the combiner
     .. code::

                             ___ single_pol_tx__
                             |                  |
                             |                  |
                             |                  |
    (in_coupler)---splitter==|                  |==combiner---(out_coupler)
                             |                  |
                             |                  |
                             |___ single_pol_tx_|
    """
    bend_spec = bend
    bend = gf.get_component(bend, cross_section=cross_section)
    straight_spec = straight

    spol_coh_tx = gf.get_component(spol_coh_tx)

    # ----- Draw single pol coherent transmitters -----

    # Add MZM 1
    c = Component()

    single_tx_1 = c << spol_coh_tx
    single_tx_2 = c << spol_coh_tx

    # Separate the two receivers
    single_tx_2.movey(single_tx_1.ymin - single_pol_tx_spacing - single_tx_2.ymax)

    # ------------ Splitters and combiners ---------------

    splitter = gf.get_component(splitter)
    sp = c << splitter
    sp.x = single_tx_1.xmin - splitter_coh_tx_spacing
    sp.y = (single_tx_1.ports["o1"].y + single_tx_2.ports["o1"].y) / 2

    route = get_route(
        sp.ports["o2"],
        single_tx_1.ports["o1"],
        straight=straight_spec,
        bend=bend_spec,
        cross_section=cross_section,
        with_sbend=False,
    )
    c.add(route.references)

    route = get_route(
        sp.ports["o3"],
        single_tx_2.ports["o1"],
        straight=straight_spec,
        bend=bend_spec,
        cross_section=cross_section,
        with_sbend=False,
    )
    c.add(route.references)

    if combiner is not None:
        combiner = gf.get_component(combiner)
        comb = c << combiner
        comb.mirror()

        comb.x = single_tx_1.xmax + splitter_coh_tx_spacing
        comb.y = (single_tx_1.ports["o2"].y + single_tx_2.ports["o2"].y) / 2

        route = get_route(
            comb.ports["o2"],
            single_tx_1.ports["o2"],
            straight=straight_spec,
            bend=bend_spec,
            cross_section=cross_section,
            with_sbend=False,
        )
        c.add(route.references)

        route = get_route(
            comb.ports["o3"],
            single_tx_2.ports["o2"],
            straight=straight_spec,
            bend=bend_spec,
            cross_section=cross_section,
            with_sbend=False,
        )
        c.add(route.references)

    # ------- In and out couplers (if indicated) -----

    if input_coupler is not None:
        # Add input coupler
        in_coupler = gf.get_component(input_coupler)
        in_coup = c << in_coupler
        in_coup.connect("o1", sp.ports["o1"])

    else:
        c.add_port("o1", port=sp.ports["o1"])

    if output_coupler is not None:

        output_coupler = gf.get_component(output_coupler)
        out_coup = c << output_coupler
        if combiner is not None:
            # Add output coupler
            out_coup.connect("o1", comb.ports["o1"])
        else:
            # Directly connect the output coupler to the branches.
            # Assumes the output couplers has ports "o1" and "o2"

            out_coup.y = (single_tx_1.y + single_tx_2.y) / 2
            out_coup.xmin = single_tx_1.xmax + 40.0

            route = get_route(
                single_tx_1.ports["o2"],
                out_coup.ports["o1"],
                straight=straight_spec,
                bend=bend_spec,
                cross_section=cross_section,
                with_sbend=False,
            )
            c.add(route.references)

            route = get_route(
                single_tx_2.ports["o2"],
                out_coup.ports["o2"],
                straight=straight_spec,
                bend=bend_spec,
                cross_section=cross_section,
                with_sbend=False,
            )
            c.add(route.references)
    else:
        c.add_port("o2", port=comb.ports["o1"])

    # ------ Extract electrical ports (if no pads) -------

    c.add_ports(single_tx_1.get_ports_list(port_type="electrical"), prefix="pol1")
    c.add_ports(single_tx_2.get_ports_list(port_type="electrical"), prefix="pol2")
    c.auto_rename_ports()

    return c
