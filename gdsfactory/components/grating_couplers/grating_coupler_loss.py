from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.routing.route_single import route_single
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
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
    gc = gf.get_component(grating_coupler)
    c = gf.Component()
    xmin = 0.0

    for i in range(2, nfibers - 1, 2):
        g1 = c << gc
        g1.drotate(rotation)
        g1.dx = xmin

        g2 = c << gc
        g2.drotate(rotation)
        g2.dx = xmin + i * pitch

        route_single(
            c,
            g1[port_name],
            g2[port_name],
            start_straight_length=40.0,
            cross_section=cross_section,
        )

        xmin = g2.dxmax + grating_coupler_spacing + gc.dxsize / 2

    return c


if __name__ == "__main__":
    c = grating_coupler_loss()
    c.show()
