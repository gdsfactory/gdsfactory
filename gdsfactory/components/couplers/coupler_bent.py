__all__ = ["coupler_bent", "coupler_bent_half"]

import gdsfactory as gf
from gdsfactory import component_functions as cf

from .._schematic import coupler_schematic


@gf.cell_with_module_name(tags=["couplers"])
def coupler_bent_half(
    gap: float = 0.200,
    radius: float = 26,
    length: float = 8.6,
    width1: float = 0.400,
    width2: float = 0.400,
    length_straight: float = 10,
    length_straight_exit: float = 18,
    cross_section: str = "strip",
) -> gf.Component:
    """Returns Broadband SOI curved / straight directional coupler.

    Args:
        gap: gap.
        radius: radius coupling.
        length: coupler_length.
        width1: width1.
        width2: width2.
        length_straight: input and output straight length.
        length_straight_exit: length straight exit.
        cross_section: cross_section.
    """
    return cf.coupler_bent_half(
        gap=gap,
        radius=radius,
        length=length,
        width1=width1,
        width2=width2,
        length_straight=length_straight,
        length_straight_exit=length_straight_exit,
        cross_section=cross_section,
    )


@gf.cell_with_module_name(schematic_function=coupler_schematic, tags=["couplers"])
def coupler_bent(
    gap: float = 0.200,
    radius: float = 26,
    length: float = 8.6,
    width1: float = 0.400,
    width2: float = 0.400,
    length_straight: float = 10,
    cross_section: str = "strip",
) -> gf.Component:
    """Returns Broadband SOI curved / straight directional coupler.

    based on: <https://doi.org/10.1038/s41598-017-07618-6>.

    Args:
        gap: gap.
        radius: radius coupling.
        length: coupler_length.
        width1: width1.
        width2: width2.
        length_straight: input and output straight length.
        cross_section: cross_section.
    """
    return cf.coupler_bent(
        gap=gap,
        radius=radius,
        length=length,
        width1=width1,
        width2=width2,
        length_straight=length_straight,
        cross_section=cross_section,
    )
