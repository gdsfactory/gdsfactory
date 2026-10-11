from __future__ import annotations

__all__ = ["grating_coupler_tree"]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name(tags=["grating_couplers"])
def grating_coupler_tree(
    n: int = 4,
    straight_spacing: float = 4.0,
    grating_coupler: ComponentSpec = "grating_coupler_elliptical_te",
    with_loopback: bool = False,
    bend: ComponentSpec = "bend_euler",
    fanout_length: float = 0.0,
    cross_section: CrossSectionSpec = "strip",
    **kwargs: Any,
) -> Component:
    """Array of straights connected with grating couplers.

    useful to align the 4 corners of the chip

    Args:
        n: number of gratings.
        straight_spacing: in um.
        grating_coupler: spec.
        with_loopback: adds loopback.
        bend: bend spec.
        fanout_length: in um.
        cross_section: cross_section function.
        kwargs: additional arguments.
    """
    return cf.grating_coupler_tree(
        n=n,
        straight_spacing=straight_spacing,
        grating_coupler=grating_coupler,
        with_loopback=with_loopback,
        bend=bend,
        fanout_length=fanout_length,
        cross_section=cross_section,
        **kwargs,
    )
