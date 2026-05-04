"""This module contains the building blocks for the CSPDK PDK."""

from functools import partial

import gdsfactory as gf
from gdsfactory.typings import (
    CrossSectionSpec,
)

################
# MMIs
################


@gf.cell
def mmi1x2(
    width: float | None = None,
    width_taper: float = 1.5,
    length_taper: float = 20.0,
    length_mmi: float = 31.0,
    width_mmi: float = 6.0,
    gap_mmi: float = 1.64,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """An mmi1x2.

    An mmi1x2 is a splitter that splits a single input to two outputs

    Args:
        width: the width of the waveguides connecting at the mmi ports.
        width_taper: the width at the base of the mmi body.
        length_taper: the length of the tapers going towards the mmi body.
        length_mmi: the length of the mmi body.
        width_mmi: the width of the mmi body.
        gap_mmi: the gap between the tapers at the mmi body.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.mmi1x2(
        width=width,
        width_taper=width_taper,
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        cross_section=cross_section,
    )


mmi1x2_rib = partial(mmi1x2, length_mmi=32.7, gap_mmi=1.64, cross_section="rib")


@gf.cell
def mmi2x2(
    width: float | None = None,
    width_taper: float = 1.5,
    length_taper: float = 20.0,
    length_mmi: float = 42.5,
    width_mmi: float = 6.0,
    gap_mmi: float = 0.5,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """An mmi2x2.

    An mmi2x2 is a 2x2 splitter

    Args:
        width: the width of the waveguides connecting at the mmi ports
        width_taper: the width at the base of the mmi body
        length_taper: the length of the tapers going towards the mmi body
        length_mmi: the length of the mmi body
        width_mmi: the width of the mmi body
        gap_mmi: the gap between the tapers at the mmi body
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.mmi2x2(
        width=width,
        width_taper=width_taper,
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        cross_section=cross_section,
    )


mmi2x2_rib = partial(mmi2x2, length_mmi=44.8, gap_mmi=0.53, cross_section="rib")
