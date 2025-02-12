from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
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
    c = gf.c.straight_array(
        n=n,
        spacing=straight_spacing,
    )

    return gf.routing.add_fiber_array(
        component=c,
        with_loopback=with_loopback,
        grating_coupler=grating_coupler,
        fanout_length=fanout_length,
        bend=bend,
        cross_section=cross_section,
        **kwargs,
    )


if __name__ == "__main__":
    c = grating_coupler_tree()
    # print(c.settings)
    c.show()
