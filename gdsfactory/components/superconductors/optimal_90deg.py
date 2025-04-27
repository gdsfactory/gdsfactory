from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def optimal_90deg(
    width: float = 100,
    num_pts: int = 15,
    length_adjust: float = 1,
    layer: LayerSpec = (1, 0),
) -> Component:
    """Returns optimally-rounded 90 degree bend that is sharp on the outer corner.

    Args:
        width: Width of the ports on either side of the bend.
        num_pts: The number of points comprising the curved section of the bend.
        length_adjust: Adjusts the length of the non-curved portion of the bend.
        layer: Specific layer(s) to put polygon geometry on.

    Notes:
        Optimal structure from https://doi.org/10.1103/PhysRevB.84.174510
        Clem, J., & Berggren, K. (2011). Geometry-dependent critical currents in
        superconducting nanocircuits. Physical Review B, 84(17), 1-27.
    """
    D = Component()

    # Get points of ideal curve
    a = 2 * width
    v = np.logspace(-length_adjust, length_adjust, num_pts)
    xi = (
        a
        / 2.0
        * ((1 + 2 / np.pi * np.arcsinh(1 / v)) + 1j * (1 + 2 / np.pi * np.arcsinh(v)))
    )
    xpts = list(np.real(xi))
    ypts = list(np.imag(xi))

    # Add points for the rest of curve
    d = 2 * xpts[0]  # Farthest point out * 2, rounded to nearest 100
    xpts.append(width)
    ypts.append(d)
    xpts.append(0)
    ypts.append(d)
    xpts.append(0)
    ypts.append(0)
    xpts.append(d)
    ypts.append(0)
    xpts.append(d)
    ypts.append(width)
    xpts.append(xpts[0])
    ypts.append(ypts[0])

    D.add_polygon(list(zip(xpts, ypts)), layer=layer)

    D.add_port(name="e1", center=(a / 4, d), width=a / 2, orientation=90, layer=layer)
    D.add_port(name="e2", center=(d, a / 4), width=a / 2, orientation=0, layer=layer)
    return D


if __name__ == "__main__":
    c = optimal_90deg()
    c.show()
