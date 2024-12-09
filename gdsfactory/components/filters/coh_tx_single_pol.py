from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.routing.route_single import route_single
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def coh_tx_single_pol(
    balanced_phase_shifters: bool = False,
    mzm_y_spacing: float = 50.0,
    phase_shifter: ComponentSpec = "straight_pin",
    phase_shifter_length: float = 100.0,
    mzm_ps_spacing: float = 40.0,
    splitter: ComponentSpec = "mmi1x2",
    combiner: ComponentSpec | None = None,
    mzm: ComponentSpec = "mzi_pin",
    mzm_length: float = 200.0,
    xspacing: float = 40.0,
    input_coupler: ComponentSpec | None = None,
    output_coupler: ComponentSpec | None = None,
    pad_array: ComponentSpec = "pad_array",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """MZM-based single polarization coherent transmitter.

    Args:
        balanced_phase_shifters: True adds phase sifters after the MZM at both the I and Q arms.
            False, only adds Q arm has a phase shifter.
        mzm_y_spacing: vertical spacing between the bottom of the I MZM and the top of the Q MZM.
        phase_shifter: phase_shifter spec.
        phase_shifter_length: length of the phase shifter.
        mzm_ps_spacing: spacing between the end of the mzm and the phase shifter.
        splitter: splitter spec.
        combiner: combiner spec.
        mzm: Mach-Zehnder modulator spec.
        mzm_length: length of the MZMs.
        xspacing: horizontal spacing between the splitter and combiner and the mzm.
        input_coupler: Optional coupler to add before the splitter.
        output_coupler: Optional coupler to add after the combiner.
        pad_array: array of pads spec.
        cross_section: for routing (splitter to mzms and mzms to combiners).

    .. code::

                                ___ mzm_i __ ps_i__
                                |                  |
                                |                  |
                                |                  |
       (in_coupler)---splitter==|                  |==combiner---(out_coupler)
                                |                  |
                                |                  |
                                |___ mzm_q __ ps_q_|
    """
    from gdsfactory.components.straight import straight as straight_function

    combiner = combiner or splitter
    pad_array = dict(component=pad_array, settings=dict(columns=1, rows=1))

    c = Component()
    mzm_mod_p = gf.get_component(mzm, length_x=mzm_length)
    mzm_i = c << mzm_mod_p
    mzm_q = c << mzm_mod_p

    # Separate the two mzms so they don't overlap
    mzm_q.dmovey(mzm_i.dymin - mzm_y_spacing - mzm_q.dymax)
    phase_shifter = gf.get_component(phase_shifter, length=phase_shifter_length)

    if balanced_phase_shifters:
        ps_i = c << phase_shifter
    else:
        # only the q arm has a phase shifter
        straight = straight_function(
            length=phase_shifter_length,
            cross_section=cross_section,
        )
        ps_i = c << straight
    ps_q = c << phase_shifter

    # Connect to the right ports
    if mzm_ps_spacing <= 0.0:
        ps_i.connect("o1", mzm_i.ports["o2"])
        ps_q.connect("o1", mzm_q.ports["o2"])
    else:
        straight_conn = straight_function(
            length=mzm_ps_spacing,
            cross_section=cross_section,
        )
        straight_i = c << straight_conn
        straight_q = c << straight_conn
        straight_i.connect("o1", mzm_i.ports["o2"])
        straight_q.connect("o1", mzm_q.ports["o2"])
        ps_i.connect("o1", straight_i.ports["o2"])
        ps_q.connect("o1", straight_q.ports["o2"])

    # ------------ Splitters and combiners ---------------

    splitter = gf.get_component(splitter)
    sp = c << splitter
    sp.dx = mzm_q.dxmin - xspacing
    sp.dy = (mzm_i.ports["o1"].dy + mzm_q.ports["o1"].dy) / 2

    route_single(
        c,
        sp.ports["o2"],
        mzm_i.ports["o1"],
        cross_section=cross_section,
    )

    route_single(
        c,
        sp.ports["o3"],
        mzm_q.ports["o1"],
        cross_section=cross_section,
    )

    combiner = gf.get_component(combiner)
    comb = c << combiner
    comb.mirror_x()

    comb.dx = ps_q.dxmax + xspacing
    comb.dy = (mzm_i.ports["o2"].dy + mzm_q.ports["o2"].dy) / 2

    route_single(
        c,
        comb.ports["o2"],
        ps_i.ports["o2"],
        cross_section=cross_section,
    )

    route_single(
        c,
        comb.ports["o3"],
        ps_q.ports["o2"],
        cross_section=cross_section,
    )

    if input_coupler is not None:
        # Add input coupler
        in_coupler = gf.get_component(input_coupler)
        in_coup = c << in_coupler
        in_coup.connect("o1", sp.ports["o1"])

    else:
        c.add_port("o1", port=sp.ports["o1"])

    if output_coupler is not None:
        # Add output coupler
        output_coupler = gf.get_component(output_coupler)
        out_coup = c << output_coupler
        out_coup.connect("o1", comb.ports["o1"])
    else:
        c.add_port("o2", port=comb.ports["o1"])

    c.add_ports(ps_i.ports.filter(port_type="electrical"), prefix="ps_i_")
    c.add_ports(ps_q.ports.filter(port_type="electrical"), prefix="ps_q_")
    c.add_ports(mzm_i.ports.filter(port_type="electrical"), prefix="mzm_i_")
    c.add_ports(mzm_q.ports.filter(port_type="electrical"), prefix="mzm_q_")
    return c


if __name__ == "__main__":
    c = coh_tx_single_pol()
    c.show()
