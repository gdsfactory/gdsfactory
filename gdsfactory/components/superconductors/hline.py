from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def hline(
    length: float = 10.0,
    width: float = 0.5,
    layer: LayerSpec = "WG",
    port_type: str = "optical",
) -> Component:
    """Creates a horizontal straight line component with ports on east and west sides.

    This component is commonly used in photonic and superconducting circuits as a basic
    waveguide or transmission line element. It creates a rectangular polygon with ports
    at both ends for easy connection to other components.

    Args:
        length: Length of the line in microns. Must be positive.
        width: Width of the line in microns. Must be positive.
        layer: Layer specification for the line (default: "WG"). Can be a string or a tuple of (layer, datatype).
        port_type: Type of port to create (default: "optical"). Common values are "optical" for photonic circuits
            or "electrical" for superconducting circuits.

    Returns:
        Component: A gdsfactory Component object containing:
            - A rectangular polygon representing the line
            - Two ports named "o1" (west) and "o2" (east)
            - Component info with width and length values

    Note:
        - The line is centered vertically at y=0
        - Port "o1" is at x=0 with orientation 180° (west)
        - Port "o2" is at x=length with orientation 0° (east)
        - If length or width is 0 or negative, no polygon is created but ports are still added
    """
    c = gf.Component()
    if length > 0 and width > 0:
        a = width / 2
        c.add_polygon([(0, -a), (length, -a), (length, a), (0, a)], layer=layer)

    c.add_port(
        name="o1",
        center=(0.0, 0.0),
        width=width,
        orientation=180,
        layer=layer,
        port_type=port_type,
    )
    c.add_port(
        name="o2",
        center=(length, 0.0),
        width=width,
        orientation=0,
        layer=layer,
        port_type=port_type,
    )

    c.info["width"] = width
    c.info["length"] = length
    return c


if __name__ == "__main__":
    c = hline(width=10)
    print(c)
    c.show()
