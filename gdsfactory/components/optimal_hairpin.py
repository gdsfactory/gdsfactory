from __future__ import annotations

import numpy as np

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@cell
def optimal_hairpin(
    width: float = 0.2,
    pitch: float = 0.6,
    length: float = 10,
    turn_ratio: float = 4,
    num_pts: int = 50,
    layer: LayerSpec = (1, 0),
) -> Component:
    """Returns an optimally-rounded hairpin geometry, with a 180 degree turn \
    on the right end of the polygon connected to two prongs extending towards \
    ports on the left end.

    based on phidl.geometry

    Args:
        width : int or float
            Width of the hairpin leads.
        pitch : int or float
            Distance between the two hairpin leads. Must be greater than width.
        length : int or float
            Length of the hairpin from the connectors to the opposite end of the
            curve.
        turn_ratio: int or float
            Specifies how much of the hairpin is dedicated to the 180 degree turn.
            A turn_ratio of 10 will result in 20% of the hairpin being comprised of the turn.
        num_pts : int
            Number of points constituting the 180 degree turn.
        layer : int, array-like[2], or set
            Specific layer(s) to put polygon geometry on.

    Notes:
        Hairpin pitch must be greater than width.

        Optimal structure from https://doi.org/10.1103/PhysRevB.84.174510
        Clem, J., & Berggren, K. (2011). Geometry-dependent critical currents in
        superconducting nanocircuits. Physical Review B, 84(17), 1–27.
    """
    # ==========================================================================
    #  Create the basic geometry
    # ==========================================================================
    a = (pitch + width) / 2
    y = -(pitch - width) / 2
    x = -pitch
    dl = width / (num_pts * 2)
    n = 0

    # Get points of ideal curve from conformal mapping
    # TODO This is an inefficient way of finding points that you need
    xpts = [x]
    ypts = [y]
    while (y < 0) & (n < 1e6):
        s = x + 1j * y
        w = np.sqrt(1 - np.exp(np.pi * s / a))
        wx = np.real(w)
        wy = np.imag(w)
        wx = wx / np.sqrt(wx**2 + wy**2)
        wy = wy / np.sqrt(wx**2 + wy**2)
        x = x + wx * dl
        y = y + wy * dl
        xpts.append(x)
        ypts.append(y)
        n += 1
    ypts[-1] = 0  # Set last point be on the x=0 axis for sake of cleanliness
    ds_factor = len(xpts) // num_pts
    xpts = xpts[::-ds_factor]
    xpts = xpts[::-1]  # This looks confusing, but it's just flipping the arrays around
    ypts = ypts[::-ds_factor]
    ypts = ypts[::-1]  # so the last point is guaranteed to be included when downsampled

    # Add points for the rest of meander
    xpts.append(xpts[-1] + turn_ratio * width)
    ypts.append(0)
    xpts.append(xpts[-1])
    ypts.append(-a)
    xpts.append(xpts[0])
    ypts.append(-a)
    xpts.append(max(xpts) - length)
    ypts.append(-a)
    xpts.append(xpts[-1])
    ypts.append(-a + width)
    xpts.append(xpts[0])
    ypts.append(ypts[0])

    xpts = np.array(xpts)
    ypts = np.array(ypts)

    # ==========================================================================
    #  Create a blank device, add the geometry, and define the ports
    # ==========================================================================
    D = Component()
    D.add_polygon([xpts, ypts], layer=layer)
    D.add_polygon([xpts, -ypts], layer=layer)

    xports = min(xpts)
    yports = -a + width / 2
    D.add_port(
        name="e1", center=[xports, -yports], width=width, orientation=180, layer=layer
    )
    D.add_port(
        name="e2", center=[xports, yports], width=width, orientation=180, layer=layer
    )
    return D


if __name__ == "__main__":
    c = optimal_hairpin()
    c.show(show_ports=True)
