from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name
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


    .. code::

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
    """
    if length_y < 0:
        raise ValueError(f"length_y={length_y} must be >= 0")

    if length_x < 0:
        raise ValueError(f"length_x={length_x} must be >= 0")

    # Create main component
    c = gf.Component()

    settings = dict(
        gap=gap,
        radius=radius,
        length_x=length_x,
        cross_section=cross_section,
        bend=bend,
        straight=straight,
    )

    if length_extension is not None:
        settings["length_extension"] = length_extension

    # Create and place the coupler
    cb = c << gf.get_component(coupler_ring, settings=settings)

    # Create waveguide components
    sy = gf.get_component(straight, length=length_y, cross_section=cross_section)
    b = gf.get_component(bend, cross_section=cross_section, radius=radius)
    sx = gf.get_component(straight, length=length_x, cross_section=cross_section)

    # Place waveguide components
    sl = c << sy  # Left vertical straight
    sr = c << sy  # Right vertical straight
    st = c << sx  # Top horizontal straight
    bl = c << b  # Left bend
    br = c << b  # Right bend

    # Connect all components
    sl.connect(port="o1", other=cb.ports["o2"])
    bl.connect(port="o2", other=sl.ports["o2"])
    st.connect(port="o2", other=bl.ports["o1"])
    br.connect(port="o2", other=st.ports["o1"])
    sr.connect(port="o1", other=br.ports["o1"])
    sr.connect(port="o2", other=cb.ports["o3"])

    # Add ports
    c.add_port("o2", port=cb.ports["o4"])
    c.add_port("o1", port=cb.ports["o1"])
    return c
