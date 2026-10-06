from __future__ import annotations

__all__ = ["grating_coupler_array"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name(tags=["grating_couplers"])
def grating_coupler_array(
    grating_coupler: ComponentSpec = "grating_coupler_elliptical",
    pitch: float = 127.0,
    n: int = 6,
    port_name: str = "o1",
    rotation: int = -90,
    with_loopback: bool = False,
    cross_section: CrossSectionSpec = "strip",
    straight_to_grating_spacing: float = 10.0,
    centered: bool = True,
    radius: float | None = None,
    bend: ComponentSpec = "bend_euler",
    mirror_grating_coupler: bool = False,
) -> Component:
    """Array of grating couplers.

    Args:
        grating_coupler: ComponentSpec.
        pitch: x spacing.
        n: number of grating couplers.
        port_name: port name.
        rotation: rotation angle for each reference.
        with_loopback: if True, adds a loopback between edge GCs. Only works for rotation = 90 for now.
        cross_section: cross_section for the routing.
        straight_to_grating_spacing: spacing between the last grating coupler and the loopback.
        centered: if True, centers the array around the origin.
        radius: optional radius for routing the loopback.
        bend: ComponentSpec for the bend used in the loopback.
        mirror_grating_coupler: if True, mirrors the grating coupler.
    """
    return cf.grating_coupler_array(
        grating_coupler=grating_coupler,
        pitch=pitch,
        n=n,
        port_name=port_name,
        rotation=rotation,
        with_loopback=with_loopback,
        cross_section=cross_section,
        straight_to_grating_spacing=straight_to_grating_spacing,
        centered=centered,
        radius=radius,
        bend=bend,
        mirror_grating_coupler=mirror_grating_coupler,
    )
