from __future__ import annotations

__all__ = ["coupler_full"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, Delta

from .._schematic import coupler_schematic


@gf.cell_with_module_name(schematic_function=coupler_schematic, tags=["couplers"])
def coupler_full(
    coupling_length: float = 40.0,
    dx: Delta = 10.0,
    dy: Delta = 4.8,
    gap: float = 0.5,
    dw: float = 0.1,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
) -> Component:
    """Adiabatic Full coupler.

    Design based on asymmetric adiabatic full
    coupler designs, such as the one reported in 'Integrated Optic Adiabatic
    Devices on Silicon' by Y. Shani, et al (IEEE Journal of Quantum
    Electronics, Vol. 27, No. 3 March 1991).

    1. is the first half of the input S-bend straight where the
    input straights widths taper by +dw and -dw,
    2. is the second half of the S-bend straight with constant,
    unbalanced widths,
    3. is the coupling region where the straights from unbalanced widths to
    balanced widths to reverse polarity unbalanced widths,
    4. is the fixed width straight that curves away from the coupling region,
    5.is the final curve where the straights taper back to the regular width
    specified in the straight template.

    Args:
        coupling_length: Length of the coupling region in um.
        dx: Length of the bend regions in um.
        dy: Port-to-port distance between the bend regions in um.
        gap: Distance between the two straights in um.
        dw: delta width. Top arm tapers to width - dw, bottom to width + dw in um.
        cross_section: cross-section spec.
        width: width of the waveguide. If None, it will use the width of the cross_section.

    """
    return cf.coupler_full(
        coupling_length=coupling_length,
        dx=dx,
        dy=dy,
        gap=gap,
        dw=dw,
        cross_section=cross_section,
        width=width,
    )
