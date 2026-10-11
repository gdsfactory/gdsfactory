from __future__ import annotations

__all__ = ["fiber"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["filters"])
def fiber(
    core_diameter: float = 10,
    cladding_diameter: float = 125,
    layer_core: LayerSpec = "WG",
    layer_cladding: LayerSpec = "WGCLAD",
) -> Component:
    """Returns a fiber.

    Args:
        core_diameter: in um.
        cladding_diameter: in um.
        layer_core: layer spec for fiber core.
        layer_cladding: layer spec for fiber cladding.
    """
    return cf.fiber(
        core_diameter=core_diameter,
        cladding_diameter=cladding_diameter,
        layer_core=layer_core,
        layer_cladding=layer_cladding,
    )
