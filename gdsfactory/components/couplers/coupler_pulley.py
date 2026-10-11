from __future__ import annotations

__all__ = ["coupler_pulley"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec


@gf.cell_with_module_name(tags=["couplers"])
def coupler_pulley(
    radius: float = 10.0,
    ring_width: float | None = None,
    gap: float = 0.2,
    coupling_angle: float = 60.0,
    wg_length: float = 40.0,
    wg_height: float = 10.0,
    n_segments: int = 128,
    cross_section: CrossSectionSpec = "strip",
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a disc or ring with a pulley-coupled waveguide.

    A waveguide wraps symmetrically around the top of a disc/ring
    over coupling_angle degrees. Bezier S-curves (following the CNST
    discPulley construction, eq. 2.22) route the waveguide from the
    coupling arc down to horizontal exits on both sides.

    Args:
        radius: radius of the disc, or outer radius of the ring.
        ring_width: width of the ring annulus. None for a solid disc.
        gap: gap between the waveguide inner edge and the disc/ring.
        coupling_angle: total wrap angle in degrees (symmetric about top).
        wg_length: horizontal half-length of the waveguide from the disc
            center to the exit end. Controls S-curve extent. Corresponds
            to CNST parameter L.
        wg_height: vertical drop from disc center to exit waveguide level.
            Corresponds to CNST parameter H.
        n_segments: number of points for each curved section.
        cross_section: cross-section spec for the coupling waveguide.
        layer: layer spec for the disc/ring.
    """
    return cf.coupler_pulley(
        radius=radius,
        ring_width=ring_width,
        gap=gap,
        coupling_angle=coupling_angle,
        wg_length=wg_length,
        wg_height=wg_height,
        n_segments=n_segments,
        cross_section=cross_section,
        layer=layer,
    )


if __name__ == "__main__":
    c = coupler_pulley()
    c.pprint_ports()
    c.show()
