from __future__ import annotations

__all__ = ["optimal_90deg"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["superconductors"])
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
        Optimal structure from <https://doi.org/10.1103/PhysRevB.84.174510>
        Clem, J., & Berggren, K. (2011). Geometry-dependent critical currents in
        superconducting nanocircuits. Physical Review B, 84(17), 1-27.
    """
    return cf.optimal_90deg(
        width=width,
        num_pts=num_pts,
        length_adjust=length_adjust,
        layer=layer,
    )
