from __future__ import annotations

__all__ = ["C"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec, Size


@gf.cell_with_module_name(tags=["shapes"])
def C(
    width: float = 1.0,
    size: Size = (10.0, 20.0),
    layer: LayerSpec = "WG",
    port_type: str = "electrical",
) -> Component:
    """C geometry with ports on both ends.

    based on phidl.

    Args:
        width: of the line.
        size: length and height of the base.
        layer: layer spec.
        port_type: optical or electrical.

    ```text
         ______
        |       o1
        |   ___
        |  |
        |  |___
        ||<---> size[0]
        |______ o2
    ```
    """
    return cf.C(width=width, size=size, layer=layer, port_type=port_type)
