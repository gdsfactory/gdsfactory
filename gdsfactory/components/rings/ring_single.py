from __future__ import annotations

__all__ = ["ring_single"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import ring_single_schematic


@gf.cell_with_module_name(schematic_function=ring_single_schematic, tags=["rings"])
def ring_single(
    gap: float = 0.2,
    radius: float | None = None,
    length_x: float = 4.0,
    length_y: float = 0.6,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    coupler_ring: ComponentSpec = "coupler_ring",
    cross_section: CrossSectionSpec = "strip",
    length_extension: float | None = None,
) -> gf.Component:
    """Returns a single ring resonator with a directional coupler.

    This component creates a ring resonator that consists of:
    - A directional coupler (cb) at the bottom
    - Two vertical straights (sl, sr) on the left and right sides
    - Two bends (bl, br) connecting the vertical straights
    - A horizontal straight (st) at the top

    The ring resonator is commonly used in photonic integrated circuits for:
    - Wavelength filtering
    - Optical modulation
    - Sensing applications
    - Optical switching

    Args:
        gap: Gap between the ring and the straight waveguide in the coupler (μm).
        radius: Radius of the ring bends (μm). If None, it will use the radius from the cross section.
        length_x: Length of the horizontal straight section (μm).
        length_y: Length of the vertical straight sections (μm).
        bend: Component spec for the 90-degree bends. Default is "bend_euler".
        straight: Component spec for the straight waveguides. Default is "straight".
        coupler_ring: Component spec for the ring coupler. Default is "coupler_ring".
        cross_section: Cross section spec for all waveguides. Default is "strip".
        length_extension: straight length extension at the end of the coupler bottom ports.

    Returns:
        Component: A gdsfactory Component containing the ring resonator with:
            - Two ports: "o1" (input) and "o2" (through)
            - All waveguide sections properly connected
            - Cross section applied to all waveguides

    Raises:
        ValueError: If length_x or length_y is negative.

    ```text
                    xxxxxxxxxxxxx
                xxxxx           xxxx
              xxx                   xxx
            xxx                       xxx
           xx                           xxx
           x                             xxx
          xx                              xx▲
          xx                              xx│length_y
          xx                              xx▼
          xx                             xx
           xx          length_x          x
            xx     ◄───────────────►    x
             xx                       xxx
               xx                   xxx
                xxx──────▲─────────xxx
                         │gap
                 o1──────▼─────────o2◄──────────────►
                                     length_extension
    ```
    """
    return cf.ring_single(
        gap=gap,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        bend=bend,
        straight=straight,
        coupler_ring=coupler_ring,
        cross_section=cross_section,
        length_extension=length_extension,
    )
