from __future__ import annotations

__all__ = ["polarization_splitter_rotator"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, Delta, Float2, Float3


@gf.cell_with_module_name(tags=["filters"])
def polarization_splitter_rotator(
    width_taper_in: Float3 = (0.54, 0.69, 0.83),
    length_taper_in: Float2 | Float3 = (4.0, 44.0),
    width_coupler: Float2 = (0.9, 0.404),
    length_coupler: float = 7.0,
    gap: float = 0.15,
    width_out: float = 0.54,
    length_out: float = 14.33,
    dy: Delta = 5.0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns polarization splitter rotator.

    "Novel concept for ultracompact polarization splitter-rotator
    based on silicon nanowires." By D. Dai, and J. E. Bowers
    (Optics express vol 19, no. 11 pp. 10940-10949 (2011)).

    Args:
        width_taper_in: Three west widths of the input tapers in um.
        length_taper_in: Two or three length of the bend regions in um.
        width_coupler: Top and bottom widths of the coupling region in um.
        length_coupler: Length of the coupling region in um.
        gap: Distance between the coupler in um.
        width_out: Width of the splitter region in um.
        length_out: Length of the splitter region in um.
        dy: Port-to-port distance between the splitter region in um.
        cross_section: cross-section spec.


    Notes:
        The length of third input taper is automatically determined
        if only two lengths are in arguments.
    """
    return cf.polarization_splitter_rotator(
        width_taper_in=width_taper_in,
        length_taper_in=length_taper_in,
        width_coupler=width_coupler,
        length_coupler=length_coupler,
        gap=gap,
        width_out=width_out,
        length_out=length_out,
        dy=dy,
        cross_section=cross_section,
    )
