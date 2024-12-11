from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.tapers.taper import taper as taper_function
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def mmi_90degree_hybrid(
    width: float = 0.5,
    width_taper: float = 1.7,
    length_taper: float = 40.0,
    length_mmi: float = 175.0,
    width_mmi: float = 10.0,
    gap_mmi: float = 0.8,
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""90 degree hybrid based on a 4x4 MMI.

    Default values from Watanabe et al.,
    "Coherent few mode demultiplexer realized as a
    2D grating coupler array in silicon", Optics Express 28(24), 2020

    It could be interesting to consider the design in Guan et al.,
    "Compact and low loss 90° optical hybrid on a silicon-on-insulator
    platform", Optics Express 25(23), 2017

    Args:
        width: input and output straight width.
        width_taper: interface between input straights and mmi region.
        length_taper: into the mmi region.
        length_mmi: in x direction.
        width_mmi: in y direction.
        gap_mmi: (width_taper + gap between tapered wg)/2.
        straight: straight function.
        with_bbox: box in bbox_layers and bbox_offsets avoid DRC sharp edges.
        cross_section: spec.


    .. code::

                   length_mmi
                    <------>
                    ________
                   |        |
                __/          \__
     signal_in  __            __  I_out1
                  \          /_ _ _ _
                  |         | _ _ _ _| gap_mmi
                  |          \__
                  |           __  Q_out1
                  |          /
                  |        |
                  |
                __/          \__
        LO_in   __            __  Q_out2
                  \          /_ _ _ _
                  |         | _ _ _ _| gap_mmi
                  |          \__
                  |           __  I_out2
                  |          /
                  | ________|


                 <->
            length_taper
    """
    c = gf.Component()

    gap_mmi = gf.snap.snap_to_grid(gap_mmi, grid_factor=2)
    w_mmi = width_mmi
    w_taper = width_taper

    taper = taper_function(
        length=length_taper,
        width1=width,
        width2=w_taper,
        cross_section=cross_section,
    )

    x = gf.get_cross_section(cross_section)

    _ = c << gf.get_component(
        straight,
        length=length_mmi,
        width=w_mmi,
        cross_section=cross_section,
    )

    y_signal_in = gap_mmi * 3 / 2 + width_taper * 3 / 2
    y_lo_in = -gap_mmi / 2 - width_taper / 2

    ports = [
        # Inputs
        gf.Port(
            "signal_in",
            orientation=180,
            center=(0, y_signal_in),
            width=w_taper,
            cross_section=x,
        ),
        gf.Port(
            "LO_in",
            orientation=180,
            center=(0, y_lo_in),
            width=w_taper,
            cross_section=x,
        ),
        # Outputs
        gf.Port(
            "I_out1",
            orientation=0,
            center=(length_mmi, y_signal_in),
            width=w_taper,
            cross_section=x,
        ),
        gf.Port(
            "Q_out1",
            orientation=0,
            center=(length_mmi, y_signal_in - gap_mmi - w_taper),
            width=w_taper,
            cross_section=x,
        ),
        gf.Port(
            "Q_out2",
            orientation=0,
            center=(length_mmi, y_lo_in),
            width=w_taper,
            cross_section=x,
        ),
        gf.Port(
            "I_out2",
            orientation=0,
            center=(length_mmi, y_lo_in - gap_mmi - w_taper),
            width=w_taper,
            cross_section=x,
        ),
    ]

    for port in ports:
        taper_ref = c << taper
        taper_ref.connect(port="o2", other=port)
        c.add_port(name=port.name, port=taper_ref.ports["o1"])

    c.flatten()
    x.add_bbox(c)
    return c


if __name__ == "__main__":
    c = mmi_90degree_hybrid()
    c.show()
