from __future__ import annotations

__all__ = ["optimal_hairpin"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["superconductors"])
def optimal_hairpin(
    width: float = 0.2,
    pitch: float = 0.6,
    length: float = 10,
    turn_ratio: float = 4,
    num_pts: int = 50,
    layer: LayerSpec = (1, 0),
) -> Component:
    """Returns an optimally-rounded hairpin geometry, with a 180 degree turn.

    based on phidl.geometry

    Args:
        width: Width of the hairpin leads.
        pitch: Distance between the two hairpin leads. Must be greater than width.
        length: Length of the hairpin from the connectors to the opposite end of the curve.
        turn_ratio: int or float
            Specifies how much of the hairpin is dedicated to the 180 degree turn.
            A turn_ratio of 10 will result in 20% of the hairpin being comprised of the turn.
        num_pts: Number of points constituting the 180 degree turn.
        layer: Specific layer(s) to put polygon geometry on.

    Notes:
        Hairpin pitch must be greater than width.

        Optimal structure from <https://doi.org/10.1103/PhysRevB.84.174510>
        Clem, J., & Berggren, K. (2011). Geometry-dependent critical currents in
        superconducting nanocircuits. Physical Review B, 84(17), 1-27.
    """
    return cf.optimal_hairpin(
        width=width,
        pitch=pitch,
        length=length,
        turn_ratio=turn_ratio,
        num_pts=num_pts,
        layer=layer,
    )
