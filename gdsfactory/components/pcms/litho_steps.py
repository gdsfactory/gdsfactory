from __future__ import annotations

__all__ = ["litho_steps"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["pcms"])
def litho_steps(
    line_widths: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0),
    line_spacing: float = 10.0,
    height: float = 100.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Positive + negative tone linewidth test.

    used for lithography resolution test patterning
    based on phidl

    Args:
        line_widths: in um.
        line_spacing: in um.
        height: in um.
        layer: Specific layer to put the ruler geometry on.
    """
    return cf.litho_steps(
        line_widths=line_widths,
        line_spacing=line_spacing,
        height=height,
        layer=layer,
    )
