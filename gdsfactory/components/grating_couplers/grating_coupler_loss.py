from __future__ import annotations

__all__ = ["grating_coupler_loss"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name(tags=["grating_couplers"])
def grating_coupler_loss(
    pitch: float = 127.0,
    grating_coupler: ComponentSpec = "grating_coupler_elliptical_trenches",
    cross_section: CrossSectionSpec = "strip",
    port_name: str = "o1",
    rotation: float = -90,
    nfibers: int = 10,
    grating_coupler_spacing: float = 5.0,
) -> Component:
    """Grating coupler test structure for de-embeding fiber array.

    Connects channel 1->3, 1->5 ... 1->nfibers with grating couplers.

    Only odd channels are connected to the grating couplers as even channels in the align_tree.

    Args:
        pitch: um.
        grating_coupler: spec.
        cross_section: spec.
        port_name: for the grating_coupler port.
        rotation: degrees.
        nfibers: number of fibers to connect.
        grating_coupler_spacing: um.
    """
    return cf.grating_coupler_loss(
        pitch=pitch,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
        port_name=port_name,
        rotation=rotation,
        nfibers=nfibers,
        grating_coupler_spacing=grating_coupler_spacing,
    )
