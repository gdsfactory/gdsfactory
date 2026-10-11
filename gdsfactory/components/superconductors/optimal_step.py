from __future__ import annotations

__all__ = ["optimal_step"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["superconductors"])
def optimal_step(
    start_width: float = 10,
    end_width: float = 22,
    num_pts: int = 50,
    width_tol: float = 1e-3,
    anticrowding_factor: float = 1.2,
    symmetric: bool = False,
    layer: LayerSpec = (1, 0),
) -> Component:
    """Returns an optimally-rounded step geometry.

    Args:
        start_width: Width of the connector on the left end of the step.
        end_width: Width of the connector on the right end of the step.
        num_pts: number of points comprising the entire step geometry.
        width_tol: Point at which to terminate the calculation of the optimal step
        anticrowding_factor: Factor to reduce current crowding by elongating
            the structure and reducing the curvature
        symmetric: If True, adds a mirrored copy of the step across the x-axis to the
            geometry and adjusts the width of the ports.
        layer: layer spec to put polygon geometry on.

    based on phidl.geometry
    Optimal structure from <https://doi.org/10.1103/PhysRevB.84.174510>
    Clem, J., & Berggren, K. (2011). Geometry-dependent critical currents in
    superconducting nanocircuits. Physical Review B, 84(17), 1-27.
    """
    return cf.optimal_step(
        start_width=start_width,
        end_width=end_width,
        num_pts=num_pts,
        width_tol=width_tol,
        anticrowding_factor=anticrowding_factor,
        symmetric=symmetric,
        layer=layer,
    )
