from __future__ import annotations

__all__ = ["ramp"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec

from .._schematic import taper_schematic


@gf.cell_with_module_name(schematic_function=taper_schematic, tags=["tapers"])
def ramp(
    length: float = 10.0,
    width1: float = 5.0,
    width2: float | None = 8.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Return a ramp component.

    Based on phidl.

    Args:
        length: Length of the ramp section.
        width1: Width of the start of the ramp section.
        width2: Width of the end of the ramp section (defaults to width1).
        layer: Specific layer to put polygon geometry on.
    """
    return cf.ramp(length=length, width1=width1, width2=width2, layer=layer)
