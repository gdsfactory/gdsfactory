from __future__ import annotations

__all__ = ["coupler_broadband"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import coupler_schematic


@gf.cell_with_module_name(schematic_function=coupler_schematic, tags=["couplers"])
def coupler_broadband(
    w_sc: float = 0.5,  # width of waveguides in the symmetric coupler section
    gap_sc: float = 0.2,  # gap size between the waveguides in the symmetric coupler section
    w_top: float = 0.6,  # width of the top waveguide in the phase control section
    gap_pc: float = 0.3,  # gap size in the phase control section
    legnth_taper: float = 1.0,  # length of the tapers
    bend: ComponentSpec = "bend_euler",
    coupler_straight: ComponentSpec = "coupler_straight",
    length_coupler_straight: float = 12.4,  # optimal L_1 from the 3d fdtd analysis
    lenght_coupler_big_gap: float = 4.7,  # optimal L_2 from the 3d fdtd analysis
    cross_section: CrossSectionSpec = "strip",
    radius: float = 10.0,
) -> Component:
    """Returns broadband coupler component.

    <https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/BroadbandDirectionalCoupler.html>
    proposed in Zeqin Lu, Han Yun, Yun Wang, Zhitian Chen, Fan Zhang, Nicolas A. F. Jaeger, and Lukas Chrostowski,
    "Broadband silicon photonic directional coupler using asymmetric-waveguide based phase control,"
    Opt. Express 23, 3795-3808 (2015), DOI: 10.1364/OE.23.003795.

    Args:
        w_sc: width of waveguides in the symmetric coupler section.
        gap_sc: gap size between the waveguides in the symmetric coupler section.
        w_top: width of the top waveguide in the phase control section.
        gap_pc: gap size in the phase control section.
        legnth_taper: length of the tapers.
        bend: bend factory.
        coupler_straight: coupler_straight factory.
        length_coupler_straight: optimal L_1 from the 3d fdtd analysis.
        lenght_coupler_big_gap: optimal L_2 from the 3d fdtd analysis.
        cross_section: cross_section of the waveguides.
        radius: bend radius.
    """
    return cf.coupler_broadband(
        w_sc=w_sc,
        gap_sc=gap_sc,
        w_top=w_top,
        gap_pc=gap_pc,
        legnth_taper=legnth_taper,
        bend=bend,
        coupler_straight=coupler_straight,
        length_coupler_straight=length_coupler_straight,
        lenght_coupler_big_gap=lenght_coupler_big_gap,
        cross_section=cross_section,
        radius=radius,
    )
