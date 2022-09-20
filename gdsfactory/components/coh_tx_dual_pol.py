from typing import Optional

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.routing.get_route import get_route
from gdsfactory.types import ComponentSpec, CrossSectionSpec


@cell
def coh_tx_dual_pol(
    splitter: ComponentSpec = "mmi1x2",
    combiner: Optional[ComponentSpec] = None,
    spol_coh_tx: ComponentSpec = "coh_tx_single_pol",
    yspacing: float = 10.0,
    xspacing: float = 40.0,
    input_coupler: Optional[ComponentSpec] = None,
    output_coupler: Optional[ComponentSpec] = None,
    cross_section: CrossSectionSpec = "strip",
    **kwargs
) -> Component:
    """Dual polarization coherent transmitter.

    Args:
        splitter: splitter function.
        combiner: combiner function.
        spol_coh_tx: function generating a coherent tx for a single polarization.
        yspacing: vertical spacing between each single polarization coherent tx.
        xspacing: horizontal spacing between splitter and combiner.
        input_coupler: Optional coupler to add before the splitter.
        output_coupler: Optioncal coupler to add after the combiner.
        cross_section: for routing (splitter to mzms and mzms to combiners).
        kwargs: cross_section settings.

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
    spol_coh_tx = gf.get_component(spol_coh_tx)

    # ----- Draw single pol coherent transmitters -----

    # Add MZM 1
    c = Component()

    single_tx_1 = c << spol_coh_tx
    single_tx_2 = c << spol_coh_tx

    # Separate the two receivers
    single_tx_2.movey(single_tx_1.ymin - yspacing - single_tx_2.ymax)

    # ------------ Splitters and combiners ---------------
    splitter = gf.get_component(splitter)
    sp = c << splitter
    sp.x = single_tx_1.xmin - xspacing
    sp.y = (single_tx_1.ports["o1"].y + single_tx_2.ports["o1"].y) / 2

    route = get_route(
        sp.ports["o2"],
        single_tx_1.ports["o1"],
        cross_section=cross_section,
        with_sbend=False,
        **kwargs
    )
    c.add(route.references)

    route = get_route(
        sp.ports["o3"],
        single_tx_2.ports["o1"],
        cross_section=cross_section,
        with_sbend=False,
        **kwargs
    )
    c.add(route.references)

    if combiner:
        combiner = gf.get_component(combiner)
        comb = c << combiner
        comb.mirror()

        comb.x = single_tx_1.xmax + xspacing
        comb.y = (single_tx_1.ports["o2"].y + single_tx_2.ports["o2"].y) / 2

        route = get_route(
            comb.ports["o2"],
            single_tx_1.ports["o2"],
            cross_section=cross_section,
            with_sbend=False,
            **kwargs
        )
        c.add(route.references)

        route = get_route(
            comb.ports["o3"],
            single_tx_2.ports["o2"],
            cross_section=cross_section,
            with_sbend=False,
            **kwargs
        )
        c.add(route.references)

    # ------- In and out couplers (if indicated) -----

    if input_coupler:
        # Add input coupler
        in_coupler = gf.get_component(input_coupler)
        in_coup = c << in_coupler
        in_coup.connect("o1", sp.ports["o1"])

    else:
        c.add_port("o1", port=sp.ports["o1"])

    if output_coupler:
        output_coupler = gf.get_component(output_coupler)
        out_coup = c << output_coupler
        if combiner:
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
                cross_section=cross_section,
                with_sbend=False,
                **kwargs
            )
            c.add(route.references)

            route = get_route(
                single_tx_2.ports["o2"],
                out_coup.ports["o2"],
                cross_section=cross_section,
                with_sbend=False,
                **kwargs
            )
            c.add(route.references)
    else:
        c.add_port("o2", port=single_tx_1.ports["o2"])
        c.add_port("o3", port=single_tx_2.ports["o2"])

    c.add_ports(single_tx_1.get_ports_list(port_type="electrical"), prefix="pol1")
    c.add_ports(single_tx_2.get_ports_list(port_type="electrical"), prefix="pol2")
    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = coh_tx_dual_pol()
    c.show(show_ports=True)
