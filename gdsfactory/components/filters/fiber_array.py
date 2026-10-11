from __future__ import annotations

__all__ = ["fiber_array"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["filters"])
def fiber_array(
    n: int = 8,
    pitch: float = 127.0,
    core_diameter: float = 10,
    cladding_diameter: float = 125,
    layer_core: LayerSpec = "WG",
    layer_cladding: LayerSpec = "WGCLAD",
) -> Component:
    """Returns a fiber array.

    Args:
        n: number of fibers.
        pitch: spacing.
        core_diameter: 10um.
        cladding_diameter: in um.
        layer_core: layer spec for fiber core.
        layer_cladding: layer spec for fiber cladding.

    ```text
        pitch
         <->
        _________
       |         | lid
       | o o o o |
       |         | base
       |_________|
          length
    ```
    """
    return cf.fiber_array(
        n=n,
        pitch=pitch,
        core_diameter=core_diameter,
        cladding_diameter=cladding_diameter,
        layer_core=layer_core,
        layer_cladding=layer_cladding,
    )
